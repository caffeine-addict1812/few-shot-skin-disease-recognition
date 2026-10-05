import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import make_splits, scan_images  # noqa: E402

# 9 base classes (>=20 imgs), 6 novel classes (10-15 imgs), 1 tiny novel class (7 imgs, dropped at test time)
COUNTS = {**{f"base_{i:02d}": 20 + i for i in range(9)},
          **{f"novel_{i:02d}": 10 + i for i in range(6)},
          "tiny_00": 7}


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """Imagefolder of class-coloured noise images + its splits dict."""
    root = tmp_path_factory.mktemp("imgs")
    rng = np.random.default_rng(0)
    for ci, (cls, n) in enumerate(COUNTS.items()):
        color = np.array([(ci * 67) % 256, (ci * 131) % 256, (ci * 29 + 90) % 256])
        (root / cls).mkdir()
        for i in range(n):
            arr = np.clip(color + rng.normal(0, 25, (80, 100, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(root / cls / f"{i:03d}.jpg", quality=90)
    counts = {c: len(v) for c, v in scan_images(root).items()}
    splits = {"meta": {}, "splits": make_splits(counts, seed=42), "counts": counts}
    return root, splits


def small_transform(size=64):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])
