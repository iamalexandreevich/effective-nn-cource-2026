import torch


def diagonal_weights(length):
    weights = torch.zeros(length, length)
    for row in range(1, length):
        weights[row, :row] = -1 / row
        weights[row, row] = 1
    return weights


def previous_token_weights(length):
    weights = torch.zeros(length, length)
    for row in range(1, length):
        weights[row, : row + 1] = -1 / row
        weights[row, row - 1] = 1
    return weights


def positional_bias(attention, weights):
    length = attention.shape[-1]
    return ((attention * weights).sum(dim=(-2, -1)) / length).mean(dim=0)


def bias_gradient_norms(model, input_ids, weights):
    attentions = model(input_ids, output_attentions=True).attentions
    norms = []
    for layer, attention in zip(model.model.layers, attentions):
        bias = positional_bias(attention, weights).mean()
        parameters = [layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight]
        gradients = torch.autograd.grad(bias, parameters, retain_graph=True, allow_unused=True)
        squared = sum((gradient**2).sum() for gradient in gradients if gradient is not None)
        norms.append(float(torch.sqrt(torch.as_tensor(squared))))
    return norms
