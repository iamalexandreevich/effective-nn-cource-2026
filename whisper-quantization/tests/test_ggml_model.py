import struct

import numpy as np
import pytest

from ggml_model import Model, replace_q4_payloads


def write_model(path, ftype, tensors):
    header = struct.pack("<12i", 0x67676D6C, 1, 8, 32, 1, 1, 8, 32, 1, 1, 80, ftype)
    mel_and_tokens = struct.pack("<2ifi", 1, 1, 0.5, 1) + struct.pack("<i", 1) + b"a"
    with path.open("wb") as stream:
        stream.write(header + mel_and_tokens)
        for name, dimensions, type_id, payload in tensors:
            encoded = name.encode()
            stream.write(struct.pack("<iii", len(dimensions), len(encoded), type_id))
            stream.write(struct.pack(f"<{len(dimensions)}i", *dimensions))
            stream.write(encoded)
            stream.write(payload)


def test_reads_fp16_rows_in_torch_dimension_order(tmp_path):
    path = tmp_path / "source.bin"
    weights = np.arange(64, dtype="<f2").reshape(2, 32)
    write_model(path, 1, [("encoder.weight", (32, 2), 1, weights.tobytes())])
    model = Model.read(path)
    np.testing.assert_array_equal(model.weights("encoder.weight"), weights)


def test_preserves_every_byte_outside_replaced_payload(tmp_path):
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    write_model(source, 2002, [("a", (32, 2), 2, b"A" * 36), ("b", (32, 1), 2, b"B" * 18)])
    model = Model.read(source)
    replace_q4_payloads(model, {"a": b"C" * 36}, target)
    offset = model.tensors["a"].offset
    assert target.read_bytes() == source.read_bytes()[:offset] + b"C" * 36 + source.read_bytes()[offset + 36:]


def test_rejects_truncated_tensor(tmp_path):
    path = tmp_path / "broken.bin"
    write_model(path, 2002, [("a", (32, 2), 2, b"A" * 35)])
    with pytest.raises(ValueError, match="truncated"):
        Model.read(path)


def test_rejects_output_that_aliases_control(tmp_path):
    path = tmp_path / "source.bin"
    alias = tmp_path / "same.bin"
    write_model(path, 2002, [("a", (32, 1), 2, b"A" * 18)])
    alias.hardlink_to(path)
    with pytest.raises(ValueError, match="distinct"):
        replace_q4_payloads(Model.read(path), {"a": b"B" * 18}, alias)
