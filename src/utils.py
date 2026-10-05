import json
import math
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp(amp, device):
    """'auto' -> bf16 on CUDA when supported, else off. Returns 'bf16' | 'fp16' | 'none'."""
    if amp == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return "bf16"
        return "none"
    return amp


def mean_ci95(values):
    """Mean and 95% CI half-width (normal approx, 1.96 * std / sqrt(n))."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / math.sqrt(len(v)))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
