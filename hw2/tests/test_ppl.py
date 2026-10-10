import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ppl import perplexity

VOCAB_SIZE = 64
SEQUENCES = torch.randint(0, VOCAB_SIZE, (2, 48), generator=torch.Generator().manual_seed(0))


def tiny_model():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config).eval()


def test_uniform_predictions_give_vocabulary_size():
    model = tiny_model()
    torch.nn.init.zeros_(model.lm_head.weight)
    assert perplexity(model, SEQUENCES, device="cpu") == pytest.approx(VOCAB_SIZE, rel=1e-4)


def test_window_covering_sequence_equals_full_context():
    model = tiny_model()
    assert perplexity(model, SEQUENCES, device="cpu", window=64) == pytest.approx(
        perplexity(model, SEQUENCES, device="cpu"), rel=1e-5
    )


def test_short_window_changes_perplexity():
    model = tiny_model()
    assert perplexity(model, SEQUENCES, device="cpu", window=8) != pytest.approx(
        perplexity(model, SEQUENCES, device="cpu"), rel=1e-5
    )
