import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from analysis import bias_gradient_norms, diagonal_weights, positional_bias, previous_token_weights
from custom_models.drope import DroPELlamaConfig, DroPELlamaForCausalLM

LENGTH = 12
TINY_CONFIG = dict(
    vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, attn_implementation="eager",
)
REPEATED_TOKENS = torch.full((1, LENGTH), 7)
TEXT_TOKENS = torch.randint(0, 64, (1, LENGTH), generator=torch.Generator().manual_seed(0))
WEIGHT_BUILDERS = [diagonal_weights, previous_token_weights]


def nope_model():
    torch.manual_seed(0)
    return DroPELlamaForCausalLM(DroPELlamaConfig(attention_type="nope", **TINY_CONFIG))


def rope_model():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(**TINY_CONFIG))


@pytest.mark.parametrize("build_weights", WEIGHT_BUILDERS, ids=["diagonal", "previous_token"])
def test_weights_are_centered_per_row(build_weights):
    assert torch.allclose(build_weights(LENGTH).sum(dim=1), torch.zeros(LENGTH), atol=1e-6)


@pytest.mark.parametrize("build_weights", WEIGHT_BUILDERS, ids=["diagonal", "previous_token"])
def test_uniform_attention_has_zero_bias(build_weights):
    uniform = torch.tril(torch.ones(LENGTH, LENGTH)) / torch.arange(1, LENGTH + 1).unsqueeze(1)
    assert torch.allclose(positional_bias(uniform.view(1, 1, LENGTH, LENGTH), build_weights(LENGTH)), torch.zeros(1), atol=1e-6)


def test_diagonal_attention_maximizes_diagonal_bias():
    identity = torch.eye(LENGTH).view(1, 1, LENGTH, LENGTH)
    assert positional_bias(identity, diagonal_weights(LENGTH)).item() == pytest.approx((LENGTH - 1) / LENGTH)


def test_nope_attention_is_uniform_on_repeated_tokens():
    attentions = nope_model()(REPEATED_TOKENS, output_attentions=True).attentions
    uniform = torch.tril(torch.ones(LENGTH, LENGTH)) / torch.arange(1, LENGTH + 1).unsqueeze(1)
    assert all(torch.allclose(attention[0], uniform.expand_as(attention[0]), atol=1e-6) for attention in attentions)


def test_nope_query_key_gradients_vanish_on_repeated_tokens():
    model = nope_model()
    model(REPEATED_TOKENS, labels=REPEATED_TOKENS).loss.backward()
    gradients = [p.grad for name, p in model.named_parameters() if "q_proj" in name or "k_proj" in name]
    assert max(gradient.abs().max().item() for gradient in gradients) < 1e-7


def test_nope_outputs_are_constant_on_repeated_tokens():
    with torch.no_grad():
        logits = nope_model()(REPEATED_TOKENS).logits[0]
    assert torch.allclose(logits, logits[0].expand_as(logits), atol=1e-5)


@pytest.mark.parametrize("build_weights", WEIGHT_BUILDERS, ids=["diagonal", "previous_token"])
def test_nope_bias_gradient_vanishes_on_repeated_tokens(build_weights):
    assert max(bias_gradient_norms(nope_model(), REPEATED_TOKENS, build_weights(LENGTH))) < 1e-7


@pytest.mark.parametrize("build_weights", WEIGHT_BUILDERS, ids=["diagonal", "previous_token"])
def test_rope_bias_gradient_is_nonzero_on_repeated_tokens(build_weights):
    assert min(bias_gradient_norms(rope_model(), REPEATED_TOKENS, build_weights(LENGTH))) > 1e-4


def test_bias_gradient_norms_has_one_value_per_layer():
    assert len(bias_gradient_norms(rope_model(), TEXT_TOKENS, diagonal_weights(LENGTH))) == TINY_CONFIG["num_hidden_layers"]
