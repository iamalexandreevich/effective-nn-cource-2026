import math
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

from q4 import BLOCK_BYTES, BLOCK_WIDTH

F32 = 0
F16 = 1
Q4_0 = 2


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]
    type_id: int
    offset: int
    size: int


@dataclass(frozen=True)
class Model:
    path: Path
    hparams: tuple[int, ...]
    tensors: dict[str, Tensor]

    @classmethod
    def read(cls, path: Path) -> "Model":
        with path.open("rb") as stream:
            magic, *hparams = _ints(stream, 12)
            if magic != 0x67676D6C:
                raise ValueError("Expected a whisper.cpp GGML file")
            _skip_mel_and_tokens(stream)
            tensors = {}
            file_size = path.stat().st_size
            while stream.tell() < file_size:
                name, tensor = _tensor_header(stream)
                if tensor.offset + tensor.size > file_size:
                    raise ValueError(f"GGML file truncated in tensor {name}")
                if name in tensors:
                    raise ValueError(f"Duplicate GGML tensor {name}")
                tensors[name] = tensor
                stream.seek(tensor.size, 1)
        return cls(path, tuple(hparams), tensors)

    def payload(self, name: str) -> bytes:
        tensor = self.tensors[name]
        with self.path.open("rb") as stream:
            stream.seek(tensor.offset)
            return _read_exact(stream, tensor.size)

    def weights(self, name: str) -> NDArray[np.float32]:
        tensor = self.tensors[name]
        dtype = {F32: "<f4", F16: "<f2"}[tensor.type_id]
        return np.frombuffer(self.payload(name), dtype).reshape(tensor.shape).astype(np.float32)


def replace_q4_payloads(control: Model, replacements: dict[str, bytes], destination: Path) -> None:
    """Write a complete checkpoint, replacing only existing Q4_0 payloads."""
    if destination.exists() and destination.samefile(control.path):
        raise ValueError("Control and destination must be distinct files")
    for name, payload in replacements.items():
        tensor = control.tensors[name]
        if tensor.type_id != Q4_0 or len(payload) != tensor.size:
            raise ValueError(f"Invalid Q4_0 replacement for {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with control.path.open("rb") as source:
                shutil.copyfileobj(source, stream)
            for name, payload in replacements.items():
                stream.seek(control.tensors[name].offset)
                stream.write(payload)
            stream.close()
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("GGML file truncated")
    return data


def _ints(stream: BinaryIO, count: int) -> tuple[int, ...]:
    return struct.unpack(f"<{count}i", _read_exact(stream, 4 * count))


def _skip_mel_and_tokens(stream: BinaryIO) -> None:
    mel_bands, fft_bins = _ints(stream, 2)
    if min(mel_bands, fft_bins) <= 0:
        raise ValueError("Invalid GGML mel filter dimensions")
    stream.seek(4 * mel_bands * fft_bins, 1)
    (token_count,) = _ints(stream, 1)
    if token_count <= 0:
        raise ValueError("Invalid GGML vocabulary size")
    for _ in range(token_count):
        (token_length,) = _ints(stream, 1)
        if token_length < 0:
            raise ValueError("Invalid GGML token length")
        stream.seek(token_length, 1)
    prefix_end = stream.tell()
    if stream.seek(0, 2) < prefix_end:
        raise ValueError("GGML file truncated in vocabulary")
    stream.seek(prefix_end)


def _tensor_header(stream: BinaryIO) -> tuple[str, Tensor]:
    ndim, name_length, type_id = _ints(stream, 3)
    if not 1 <= ndim <= 4 or name_length <= 0:
        raise ValueError("Invalid GGML tensor header")
    dimensions = _ints(stream, ndim)
    name = _read_exact(stream, name_length).decode()
    if min(dimensions) <= 0:
        raise ValueError(f"Invalid GGML tensor shape for {name}")
    elements = math.prod(dimensions)
    if type_id == Q4_0 and dimensions[0] % BLOCK_WIDTH == 0:
        size = elements // BLOCK_WIDTH * BLOCK_BYTES
    elif type_id in (F16, F32):
        size = elements * (2 if type_id == F16 else 4)
    else:
        raise ValueError(f"Unsupported GGML type or shape for {name}")
    return name, Tensor(tuple(reversed(dimensions)), type_id, stream.tell(), size)
