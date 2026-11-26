import numpy as np

from q4 import Q4Blocks, gptq, reference_q4


def test_packing_places_second_half_in_high_nibbles():
    codes = np.array([[[0] * 16 + [15] * 16]], dtype=np.uint8)
    blocks = Q4Blocks(np.array([[-0.5]], dtype=np.float16), codes)
    assert blocks.pack() == b"\x00\xb8" + b"\xf0" * 16


def test_decodes_signed_scales_and_preserves_row_order():
    payload = b"\x00\x38" + b"\xf0" * 16 + b"\x00\xb8" + b"\x1e" * 16
    decoded = Q4Blocks.read(payload, (2, 32)).dequantize()
    np.testing.assert_array_equal(decoded, [[-4.0] * 16 + [3.5] * 16, [-3.0] * 16 + [3.5] * 16])


def test_reference_grid_keeps_first_signed_peak_on_tie():
    weights = np.array([[1, -1] + [0] * 30], dtype=np.float32)
    assert reference_q4(weights).pack()[:4] == b"\x00\xb0\x80\x8f"


def test_gptq_lowers_matmul_error_on_correlated_held_out_inputs():
    rng = np.random.default_rng(41)
    weights = rng.normal(size=(32, 64)).astype(np.float32)
    signal = rng.normal(size=(512, 32)).astype(np.float32)
    inputs = np.concatenate((signal, signal + 0.02 * rng.normal(size=signal.shape)), axis=1)
    train, held_out = inputs[:256], inputs[256:]
    control = reference_q4(weights).dequantize()
    proposed = gptq(weights, train.T @ train / len(train), 0.01).dequantize()
    control_error = np.linalg.norm(held_out @ (weights - control).T)
    proposed_error = np.linalg.norm(held_out @ (weights - proposed).T)
    assert proposed_error < 0.85 * control_error


def test_zero_activations_leave_reference_quantization():
    weights = np.linspace(-0.4, 0.4, 32, dtype=np.float32).reshape(1, 32)
    assert gptq(weights, np.zeros((32, 32)), 0.01).pack() == reference_q4(weights).pack()
