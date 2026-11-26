from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BLOCK_WIDTH = 32
_Q4_BLOCK = np.dtype([("scale", "<f2"), ("nibbles", "u1", BLOCK_WIDTH // 2)])
BLOCK_BYTES = _Q4_BLOCK.itemsize


@dataclass(frozen=True)
class Q4Blocks:
    scales: NDArray[np.float16]
    codes: NDArray[np.uint8]

    @classmethod
    def read(cls, payload: bytes, shape: tuple[int, int]) -> "Q4Blocks":
        rows, columns = shape
        if columns % BLOCK_WIDTH or len(payload) != rows * (columns // BLOCK_WIDTH) * BLOCK_BYTES:
            raise ValueError("Q4_0 payload does not match the requested matrix shape")
        blocks = np.frombuffer(payload, _Q4_BLOCK).reshape(rows, columns // BLOCK_WIDTH)
        nibbles = blocks["nibbles"]
        codes = np.concatenate((nibbles & 15, nibbles >> 4), axis=-1)
        return cls(blocks["scale"], codes)

    def pack(self) -> bytes:
        blocks = np.empty(self.scales.shape, _Q4_BLOCK)
        blocks["scale"] = self.scales
        lower, upper = np.split(self.codes, 2, axis=-1)
        blocks["nibbles"] = lower | (upper << 4)
        return blocks.tobytes()

    def dequantize(self) -> NDArray[np.float32]:
        values = (self.codes.astype(np.float32) - 8) * self.scales[..., None]
        return values.reshape(values.shape[0], -1)


def reference_q4(weights: NDArray) -> Q4Blocks:
    groups = _weight_groups(weights)
    scales = _signed_scales(groups)
    codes = _codes(groups, scales[..., None])
    return Q4Blocks(scales.astype("<f2"), codes)


def gptq(weights: NDArray, second_moment: NDArray, damp: float) -> Q4Blocks:
    """Quantize a matrix on a native Q4_0 grid using E[x xᵀ] for its input."""
    groups = _weight_groups(weights)
    columns = weights.shape[1]
    if second_moment.shape != (columns, columns) or not np.isfinite(second_moment).all():
        raise ValueError("Second moment must be finite and square in input-channel order")
    if not np.isfinite(damp) or damp <= 0:
        raise ValueError("Damping must be positive and finite")
    if not np.any(second_moment):
        return reference_q4(weights)

    inverse_factor = _inverse_cholesky(second_moment, damp)
    corrected = groups.copy().reshape(weights.shape)
    scales = np.empty(groups.shape[:-1], dtype="<f2")
    codes = np.empty(groups.shape, dtype=np.uint8)
    for block_index, start in enumerate(range(0, columns, BLOCK_WIDTH)):
        end = start + BLOCK_WIDTH
        block = corrected[:, start:end]
        scales[:, block_index] = _signed_scales(block)
        scale = scales[:, block_index].astype(np.float32)
        errors = _quantize_block(block, scale, inverse_factor[start:end, start:end], codes[:, block_index])
        corrected[:, end:] -= errors @ inverse_factor[start:end, end:]
    return Q4Blocks(scales, codes)


def _weight_groups(weights: NDArray) -> NDArray[np.float32]:
    if weights.ndim != 2 or not weights.size or weights.shape[1] % BLOCK_WIDTH:
        raise ValueError("Weights must be a matrix with a nonzero multiple of 32 input channels")
    if not np.isfinite(weights).all():
        raise ValueError("Weights must be finite")
    return np.asarray(weights, dtype=np.float32).reshape(weights.shape[0], -1, BLOCK_WIDTH)


def _signed_scales(groups: NDArray) -> NDArray[np.float32]:
    peak_indices = np.argmax(np.abs(groups), axis=-1, keepdims=True)
    peaks = np.take_along_axis(groups, peak_indices, axis=-1).squeeze(-1)
    return peaks / -8


def _codes(values: NDArray, scales: NDArray) -> NDArray[np.uint8]:
    inverse = np.divide(1, scales, out=np.zeros_like(scales), where=scales != 0)
    return np.clip(np.floor(values * inverse + 8.5), 0, 15).astype(np.uint8)


def _inverse_cholesky(second_moment: NDArray, damp: float) -> NDArray[np.float32]:
    hessian = np.array(second_moment, dtype=np.float64, copy=True)
    diagonal = np.diag_indices_from(hessian)
    hessian[diagonal] += damp * hessian[diagonal].mean()
    lower = np.linalg.cholesky(hessian)
    lower_inverse = np.linalg.solve(lower, np.eye(len(lower)))
    return np.linalg.cholesky(lower_inverse.T @ lower_inverse).T.astype(np.float32)


def _quantize_block(block: NDArray, scale: NDArray, factor: NDArray, codes: NDArray) -> NDArray:
    errors = np.empty_like(block)
    for column in range(BLOCK_WIDTH):
        codes[:, column] = _codes(block[:, column], scale)
        quantized = (codes[:, column].astype(np.float32) - 8) * scale
        error = (block[:, column] - quantized) / factor[column, column]
        errors[:, column] = error
        block[:, column + 1:] -= error[:, None] * factor[column, column + 1:]
    return errors
