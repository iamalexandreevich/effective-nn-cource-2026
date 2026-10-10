import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from custom_models.drope import DroPELlamaConfig, DroPELlamaForCausalLM

TINY_CONFIG = dict(
    vocab_size=64,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    attn_implementation="eager",
)
PREFIX = torch.tensor([[3, 9, 14, 27, 31, 42, 50]])
LAST_TOKEN = torch.tensor([[5]])


def last_position_logits(model, prefix):
    input_ids = torch.cat([prefix, LAST_TOKEN], dim=1)
    with torch.no_grad():
        return model(input_ids).logits[0, -1]


def build_drope_model(attention_type):
    torch.manual_seed(0)
    return DroPELlamaForCausalLM(DroPELlamaConfig(attention_type=attention_type, **TINY_CONFIG)).eval()


@pytest.mark.parametrize("attention_type", ["nope", "qk_norm_nope"], ids=["nope", "qk_norm_nope"])
def test_drope_single_layer_ignores_prefix_order(attention_type):
    model = build_drope_model(attention_type)
    assert torch.allclose(
        last_position_logits(model, PREFIX), last_position_logits(model, PREFIX.flip(1)), atol=1e-5
    )


def test_rope_single_layer_depends_on_prefix_order():
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(**TINY_CONFIG)).eval()
    assert not torch.allclose(
        last_position_logits(model, PREFIX), last_position_logits(model, PREFIX.flip(1)), atol=1e-5
    )


def save_tiny_llama(path):
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(**TINY_CONFIG))
    model.save_pretrained(path)
    return model


def load_qk_norm_drope(path):
    config = DroPELlamaConfig.from_pretrained(path, attention_type="qk_norm_nope")
    return DroPELlamaForCausalLM.from_pretrained(path, config=config)


def test_qk_norm_weights_start_at_one_after_from_pretrained(tmp_path):
    save_tiny_llama(tmp_path)
    model = load_qk_norm_drope(tmp_path)
    attentions = [layer.self_attn for layer in model.model.layers]
    norm_weights = [a.q_norm.weight for a in attentions] + [a.k_norm.weight for a in attentions]
    assert all(torch.equal(weight, torch.ones_like(weight)) for weight in norm_weights)


def test_drope_from_pretrained_keeps_projection_weights(tmp_path):
    base = save_tiny_llama(tmp_path)
    model = load_qk_norm_drope(tmp_path)
    assert torch.equal(model.model.layers[0].self_attn.q_proj.weight, base.model.layers[0].self_attn.q_proj.weight)
