import numpy as np

CONVS = [(3, 32, 7, 2), (32, 64, 5, 1), (64, 128, 3, 2), (128, 256, 1, 1), (256, 256, 3, 2), (256, 512, 1, 1)]
FCS = [(512, 256), (256, 100)]
FP32 = 4
INT64 = 8

def build_layers():
    layers = []
    live = []
    x = 3
    scale = 1.0
    for i, (cin, cout, k, stride) in enumerate(CONVS):
        inp = cin * scale**2
        scale /= stride
        out = cout * scale**2
        layers.append((f"conv{i + 1}", 2 * cin * k * k * out, 0, (inp + out) * FP32, 0, cin * cout * k * k * FP32))
        layers.append((f"relu{i + 1}", 0, 0, 2 * out * FP32, 0, 0))
        live.append((x + (inp if i > 0 else 0) + out) * FP32)
        if i == 0:
            scale /= 2
            pooled = cout * scale**2
            layers.append(("maxpool", 0, 0, (out + pooled) * FP32 + pooled * INT64, 0, 0))
            live.append((x + out + pooled) * FP32 + pooled * INT64)

    c = CONVS[-1][1]
    layers.append(("gap", 0, 0, c * scale**2 * FP32, c * FP32, 0))
    for i, (fin, fout) in enumerate(FCS):
        if i > 0:
            layers.append((f"relu_fc{i}", 0, 0, 0, 2 * fin * FP32, 0))
        layers.append((f"fc{i + 1}", 0, 2 * fin * fout, 0, (fin + fout) * FP32, (fin * fout + fout) * FP32))
    return layers, max(live)


LAYERS, PEAK_ACT_BYTES = build_layers()
WEIGHT_BYTES = sum(l[5] for l in LAYERS)
N_PARAMS = WEIGHT_BYTES // FP32


def layer_flops(layer, S, B):
    return B * (layer[1] * S**2 + layer[2])


def layer_bytes(layer, S, B):
    return B * (layer[3] * S**2 + layer[4]) + layer[5]


def flops(image_size, batch):
    return sum(layer_flops(l, image_size, batch) for l in LAYERS)


def bytes_moved(image_size, batch):
    return sum(layer_bytes(l, image_size, batch) for l in LAYERS)


def memory(image_size, batch):
    return WEIGHT_BYTES + PEAK_ACT_BYTES * batch * image_size**2


def layer_times(image_size, batch, theta):
    t_compute = [layer_flops(l, image_size, batch) / theta["peak_flops"] for l in LAYERS]
    t_memory = [layer_bytes(l, image_size, batch) / theta["bandwidth"] for l in LAYERS]
    return t_compute, t_memory


def latency(image_size, batch, theta):
    t_compute, t_memory = layer_times(image_size, batch, theta)
    gpu_time = sum(np.maximum(c, m) for c, m in zip(t_compute, t_memory))
    return np.maximum(theta["t_launch"], gpu_time)


def energy(image_size, batch, theta_energy):
    t = latency(image_size, batch, theta_energy["latency"])
    return (theta_energy["p_static"] * t
            + theta_energy["e_flop"] * flops(image_size, batch)
            + theta_energy["e_byte"] * bytes_moved(image_size, batch))


def regime(image_size, batch, theta):
    t_compute, t_memory = layer_times(image_size, batch, theta)
    gpu_time = sum(np.maximum(c, m) for c, m in zip(t_compute, t_memory))
    compute_part = sum(np.where(c > m, c, 0.0) for c, m in zip(t_compute, t_memory))
    gpu_regime = np.where(compute_part > gpu_time / 2, 2, 1)
    return np.where(theta["t_launch"] >= gpu_time, 0, gpu_regime)
