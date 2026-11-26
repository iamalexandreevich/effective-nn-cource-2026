import numpy as np
import pytest

from calibrate import Moments, input_source, train_rows


def test_gram_matrix_is_averaged_over_valid_frames():
    moments = Moments(2)
    moments.update(np.array([[1, 2], [3, 4]], dtype=np.float32))
    moments.update(np.array([[5, 6]], dtype=np.float32))
    np.testing.assert_allclose(moments.average(), [[35 / 3, 44 / 3], [44 / 3, 56 / 3]])


def test_attention_key_and_query_use_same_input():
    assert input_source("blocks.3.attn.key") == input_source("blocks.3.attn.query")


def test_calibration_rejects_test_rows(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id":"1","audio":"example.wav","split":"test"}\n')
    with pytest.raises(ValueError, match="train"):
        train_rows(manifest)
