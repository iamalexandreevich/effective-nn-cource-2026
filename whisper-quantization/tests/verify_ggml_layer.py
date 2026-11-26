import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ggml_model import Q4_0, Model
from q4 import BLOCK_WIDTH, Q4Blocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Check an exported Q4_0 tensor using native GGML")
    parser.add_argument("native_check", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tensor")
    args = parser.parse_args()
    model = Model.read(args.checkpoint)
    tensor = model.tensors[args.tensor]
    if tensor.type_id != Q4_0 or len(tensor.shape) != 2:
        parser.error("Expected a Q4_0 matrix")
    rows, width = tensor.shape
    payload = model.payload(args.tensor)
    expected_weights = Q4Blocks.read(payload, tensor.shape).dequantize()
    rng = np.random.default_rng(42)
    inputs = rng.integers(-127, 128, size=(3, width)).astype("<f4") / 128
    # Make each activation block exactly representable by GGML's Q8_0 dot-product operand.
    inputs[:, ::BLOCK_WIDTH] = 127 / 128
    with TemporaryDirectory() as directory:
        weights, features, decoded, result = (Path(directory) / name for name in
                                              ("weight.bin", "input.f32", "decoded.f32", "output.f32"))
        weights.write_bytes(payload)
        inputs.tofile(features)
        subprocess.run([str(args.native_check), str(width), str(rows), str(len(inputs)),
                        str(weights), str(features), str(decoded), str(result)], check=True)
        decoded_weights = np.fromfile(decoded, "<f4").reshape(rows, width)
        outputs = np.fromfile(result, "<f4").reshape(len(inputs), rows)
    np.testing.assert_array_equal(decoded_weights.view("<u4"), expected_weights.view("<u4"))
    expected_outputs = inputs @ expected_weights.T
    np.testing.assert_allclose(outputs, expected_outputs, rtol=1e-4, atol=2e-5)
    print(f"{args.tensor} {tensor.shape}: exact native dequantization; "
          f"max matmul abs error {np.abs(expected_outputs - outputs).max():.3g}")


if __name__ == "__main__":
    main()
