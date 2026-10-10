import pytest
import torch
from transformers import LlamaConfig

from custom_models.drope import DroPELlamaConfig, DroPELlamaForCausalLM

from positional import frequency_multipliers, load_drope_model, set_attention_temperature

HEAD_DIM = 64
SCALE = 4.0
CONFIG = LlamaConfig(
    hidden_size=2 * HEAD_DIM,
    num_attention_heads=2,
    num_key_value_heads=2,
    rope_theta=10_000.0,
    max_position_embeddings=2048,
)


def test_base_keeps_frequencies():
    assert torch.allclose(frequency_multipliers(CONFIG, "base", SCALE), torch.ones(HEAD_DIM // 2))


def test_pi_divides_all_frequencies_by_scale():
    assert torch.allclose(frequency_multipliers(CONFIG, "pi", SCALE), torch.full((HEAD_DIM // 2,), 1 / SCALE))


def test_ntk_matches_paper_formula():
    m = torch.arange(HEAD_DIM // 2, dtype=torch.float32)
    expected = (1 / SCALE) ** (2 * m / (HEAD_DIM - 2))
    assert torch.allclose(frequency_multipliers(CONFIG, "ntk", SCALE), expected, rtol=1e-4)


@pytest.mark.parametrize(
    "index, expected",
    [(0, 1.0), (HEAD_DIM // 2 - 1, 1 / SCALE)],
    ids=["highest_frequency_kept", "lowest_frequency_interpolated"],
)
def test_yarn_endpoints(index, expected):
    assert frequency_multipliers(CONFIG, "yarn", SCALE)[index].item() == pytest.approx(expected, rel=1e-4)


def test_yarn_multipliers_do_not_increase_towards_low_frequencies():
    multipliers = frequency_multipliers(CONFIG, "yarn", SCALE)
    assert torch.all(multipliers[1:] <= multipliers[:-1] + 1e-6)


def greedy_continuation(model, use_cache):
    prompt = torch.tensor([[3, 9, 14, 27, 31, 42, 50, 5]])
    with torch.no_grad():
        output = model.generate(prompt, max_new_tokens=6, do_sample=False, use_cache=use_cache, pad_token_id=0)
    return output[0, prompt.shape[1]:].tolist()


def test_qk_norm_drope_generates_same_with_and_without_cache(tmp_path):
    torch.manual_seed(0)
    tiny = LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, attn_implementation="eager",
    )
    DroPELlamaForCausalLM(DroPELlamaConfig(attention_type="qk_norm_nope", **tiny.to_dict())).save_pretrained(tmp_path)
    model = load_drope_model(tmp_path, beta=1.0, device="cpu")
    assert greedy_continuation(model, use_cache=True) == greedy_continuation(model, use_cache=False)


def tiny_qk_norm_checkpoint(path):
    torch.manual_seed(0)
    tiny = LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, attn_implementation="eager",
    )
    DroPELlamaForCausalLM(DroPELlamaConfig(attention_type="qk_norm_nope", **tiny.to_dict())).save_pretrained(path)
    return path


def last_logits(model):
    with torch.no_grad():
        return model(torch.tensor([[3, 9, 14, 27, 31, 42, 50, 5]])).logits[0, -1]


def test_temperature_equals_scaling_logits_by_beta(tmp_path):
    checkpoint = tiny_qk_norm_checkpoint(tmp_path)
    model = load_drope_model(checkpoint, beta=1.5, device="cpu")
    reference = load_drope_model(checkpoint, beta=1.0, device="cpu")
    for layer in reference.model.layers:
        layer.self_attn.scaling = 1.5 * layer.self_attn.head_dim**-0.5
    assert torch.allclose(last_logits(model), last_logits(reference), atol=1e-5)


def test_temperature_can_be_reset_to_one(tmp_path):
    checkpoint = tiny_qk_norm_checkpoint(tmp_path)
    model = load_drope_model(checkpoint, beta=1.0, device="cpu")
    original = last_logits(model)
    set_attention_temperature(model, 1.7)
    set_attention_temperature(model, 1.0)
    assert torch.allclose(last_logits(model), original, atol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="нужен MPS")
def test_temperature_changes_logits_on_mps_with_sdpa(tmp_path):
    checkpoint = tiny_qk_norm_checkpoint(tmp_path)
    model = load_drope_model(checkpoint, beta=1.0, device="mps")
    model.config._attn_implementation = "sdpa"
    input_ids = torch.tensor([[3, 9, 14, 27, 31, 42, 50, 5]], device="mps")

    def logits_with(beta):
        set_attention_temperature(model, beta)
        with torch.no_grad():
            return model(input_ids).logits[0, -1].cpu()

    assert not torch.allclose(logits_with(1.0), logits_with(1.8), atol=1e-4)
