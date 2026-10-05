"""Embed every SD-198 image (train, val, test) with the frozen ImageNet ResNet-18 and the fine-tuned seed-42 model.

    python -u scripts/embed_all.py          # -> results/all_embeddings.npz  (~20 min on CPU)

Used by scripts/build_showcase.py. Eval transform, fp32, L2-normalised (exactly model.embed).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import SD198Split, build_transforms, load_splits  # noqa: E402
from model import ProtoNet, load_model  # noqa: E402

splits = load_splits(ROOT / "splits.json")
tf = build_transforms(False)
models = {"frozen": ProtoNet(pretrained=True).eval(), "finetuned": load_model(ROOT / "checkpoints/best.pt", "cpu")[0]}
out = {}
for split in ("train", "val", "test"):
    ds = SD198Split(ROOT / "data/SD-198-256", splits, split)
    out[f"{split}_paths"] = np.array([str(p.relative_to(ROOT / "data/SD-198-256")) for p in ds.paths])
    out[f"{split}_labels"] = np.array(ds.labels)
    out[f"{split}_classes"] = np.array(ds.classes)
    for name, m in models.items():
        zs = []
        with torch.no_grad():
            for i in range(0, len(ds), 64):
                zs.append(m.embed(torch.stack([tf(Image.open(p).convert("RGB")) for p in ds.paths[i:i + 64]])))
        out[f"{split}_{name}"] = torch.cat(zs).numpy().astype(np.float32)
        print(f"{split} {name}: {out[f'{split}_{name}'].shape}", flush=True)
np.savez_compressed(ROOT / "results/all_embeddings.npz", **out)
print("wrote results/all_embeddings.npz")
