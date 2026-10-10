import copy
import math

import torch
from custom_models import drope  # noqa: F401  регистрирует llama_drope в AutoModel
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

BASE_MODEL = "HuggingFaceTB/SmolLM-360M"
TRAIN_CONTEXT = 2048
ROPE_METHODS = ("base", "pi", "ntk", "ntk_dynamic", "yarn")


def scaled_rope_config(config, method, scale):
    config = copy.deepcopy(config)
    if method == "pi":
        config.rope_scaling = {"rope_type": "linear", "factor": scale}
    elif method == "ntk":
        head_dim = config.hidden_size // config.num_attention_heads
        config.rope_theta = config.rope_theta * scale ** (head_dim / (head_dim - 2))
    elif method == "ntk_dynamic":
        config.rope_scaling = {"rope_type": "dynamic", "factor": scale}
    elif method == "yarn":
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": scale,
            "original_max_position_embeddings": TRAIN_CONTEXT,
        }
    elif method != "base":
        raise ValueError(f"неизвестный режим RoPE: {method}")
    return config


def frequency_multipliers(config, method, scale):
    return rope_frequencies(scaled_rope_config(config, method, scale)) / rope_frequencies(config)


def rope_frequencies(config):
    rope_type = (config.rope_scaling or {}).get("rope_type", "default")
    inv_freq, _ = ROPE_INIT_FUNCTIONS[rope_type](config, "cpu")
    return inv_freq


def load_rope_model(method, scale, device, dtype=torch.float32, model_name=BASE_MODEL):
    config = scaled_rope_config(AutoConfig.from_pretrained(model_name), method, scale)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, dtype=dtype)
    return model.to(device).eval()


def load_drope_model(checkpoint_dir, beta, device, dtype=torch.float32):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, dtype=dtype)
    route_cache_to_qk_norm_attention(model)
    set_attention_temperature(model, beta)
    return model.to(device).eval()


# QKNorm-внимание авторов ждёт past_key_value, а слой transformers 4.56 передаёт past_key_values:
# без переадресации KV-кэш при генерации не заполняется
def route_cache_to_qk_norm_attention(model):
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "q_norm"):
            layer.self_attn.forward = _forward_with_routed_cache(layer.self_attn.forward)


def _forward_with_routed_cache(forward):
    def forward_with_routed_cache(*args, past_key_values=None, **kwargs):
        return forward(*args, past_key_value=past_key_values, **kwargs)

    return forward_with_routed_cache


# SDPA на MPS игнорирует параметр scale, поэтому температуру задаём через веса q_norm:
# запросы умножаются на beta, и логиты внимания масштабируются так же, как при scaling = beta / sqrt(d)
def set_attention_temperature(model, beta):
    for layer in model.model.layers:
        attention = layer.self_attn
        if not hasattr(attention, "base_q_norm_weight"):
            attention.base_q_norm_weight = attention.q_norm.weight.detach().clone()
        attention.q_norm.weight.data.copy_(attention.base_q_norm_weight * beta)


def drope_temperature(scale, coefficient):
    return 1 + coefficient * math.log(scale)
