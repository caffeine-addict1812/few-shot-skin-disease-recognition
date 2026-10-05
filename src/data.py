"""SD-198 data: class-disjoint splits, dataset, deterministic episode sampling, manifest.

    python src/data.py --make-splits --seed 42     # splits.json + data_manifest.json from data/SD-198-256
    python src/data.py --verify                    # check the local copy matches data_manifest.json
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils import load_json, save_json

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- scanning / splits
def scan_images(root):
    """{class_name: [filenames]} for a root laid out as root/<class>/<image>."""
    out = {}
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith(".")):
        files = sorted(f.name for f in d.iterdir()
                       if f.suffix.lower() in IMG_EXT and not f.name.startswith("."))
        if files:
            out[d.name] = files
    return out


def make_splits(counts, seed=42, novel_thresh=20, val_frac=0.22):
    """Class-disjoint split. novel (test) = classes with < novel_thresh images; the rest (base)
    is split train/val, with val stratified over image-count terciles."""
    rng = np.random.default_rng(seed)
    base = sorted((c for c in counts if counts[c] >= novel_thresh), key=lambda c: (counts[c], c))
    novel = sorted(c for c in counts if counts[c] < novel_thresh)

    val = []
    for group in np.array_split(np.arange(len(base)), 3):
        n_val = int(round(len(group) * val_frac))
        val += [base[group[i]] for i in rng.permutation(len(group))[:n_val]]
    val = set(val)

    split = {}
    for c in sorted(counts):
        split[c] = "test" if c in novel else ("val" if c in val else "train")

    groups = {s: {c for c, v in split.items() if v == s} for s in ("train", "val", "test")}
    assert not (groups["train"] & groups["val"]) and not (groups["train"] & groups["test"]) \
        and not (groups["val"] & groups["test"]), "splits are not class-disjoint"
    assert sum(len(g) for g in groups.values()) == len(counts)
    return split


def load_splits(path):
    return load_json(path)


# --------------------------------------------------------------------------- transforms
def build_transforms(train):
    if train:
        return T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(15),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


# --------------------------------------------------------------------------- dataset
class SD198Split(Dataset):
    """Images of the classes assigned to `split`. Classes with fewer than `min_images`
    are dropped (recorded in `dropped_classes`) so every kept class can fill an episode."""

    def __init__(self, root, splits, split, transform=None, min_images=0):
        self.root, self.split, self.transform = Path(root), split, transform
        images = scan_images(self.root)
        want = sorted(c for c, s in splits["splits"].items() if s == split)
        missing = [c for c in want if c not in images]
        if missing:
            raise ValueError(f"{len(missing)} classes in splits.json missing under {self.root}: {missing[:5]}")
        self.dropped_classes = [c for c in want if len(images[c]) < min_images]
        self.classes = [c for c in want if len(images[c]) >= min_images]

        self.paths, self.labels, self.class_indices = [], [], []
        for ci, c in enumerate(self.classes):
            start = len(self.paths)
            self.paths += [self.root / c / f for f in images[c]]
            self.labels += [ci] * len(images[c])
            self.class_indices.append(np.arange(start, len(self.paths)))

    def __len__(self):
        return len(self.paths)

    def load_raw(self, i):
        return Image.open(self.paths[i]).convert("RGB")

    def __getitem__(self, i):
        img = self.load_raw(i)
        return (self.transform(img) if self.transform else img), self.labels[i]


class CachedSplit:
    """A split decoded once into RAM (fp16). Only valid for deterministic (eval) transforms.

    Val/test hold ~1k images each, so caching makes the repeated 600-episode val sweeps
    (60k images each) nearly free instead of re-decoding JPEGs every time.
    """

    def __init__(self, ds):
        for k in ("root", "split", "classes", "paths", "labels", "class_indices", "dropped_classes"):
            setattr(self, k, getattr(ds, k))
        self.load_raw = ds.load_raw
        self.images = torch.stack([ds[i][0] for i in range(len(ds))]).half()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.images[i].float(), self.labels[i]


def load_eval_split(root, splits, split, min_images=0, cache=True):
    ds = SD198Split(root, splits, split, build_transforms(False), min_images)
    return CachedSplit(ds) if cache else ds


# --------------------------------------------------------------------------- episodes
def sample_episode(class_indices, n_way, k_shot, n_query, rng):
    """One N-way episode. Support and query of a class are disjoint slices of one permutation
    (no replacement), so a query image can never also be a support image."""
    classes = rng.choice(len(class_indices), n_way, replace=False)
    support = np.empty((n_way, k_shot), np.int64)
    query = np.empty((n_way, n_query), np.int64)
    for i, c in enumerate(classes):
        idx = rng.permutation(class_indices[c])[: k_shot + n_query]
        assert len(idx) == k_shot + n_query
        support[i], query[i] = idx[:k_shot], idx[k_shot:]
    return classes, support, query


class EpisodeDataset(Dataset):
    """Deterministic episode source: episode `i` depends only on (seed, i), so val/test episode
    lists are identical across checkpoints and runs, and training can resume at any offset.

    Item: (x [N*(K+Q), 3, H, W] support-first/class-major, classes [N] dataset-class ids).
    """

    def __init__(self, dataset, n_way, k_shot, n_query, n_episodes, seed, offset=0, fixed_idx=None):
        need = k_shot + n_query
        short = [dataset.classes[c] for c, ix in enumerate(dataset.class_indices) if len(ix) < need]
        if short:
            raise ValueError(f"{len(short)} classes have < {need} images (k_shot+n_query): {short[:5]}")
        if len(dataset.classes) < n_way:
            raise ValueError(f"{len(dataset.classes)} classes < n_way={n_way}")
        self.ds, self.n_way, self.k, self.q = dataset, n_way, k_shot, n_query
        self.n_episodes, self.seed, self.offset, self.fixed_idx = n_episodes, seed, offset, fixed_idx

    def __len__(self):
        return self.n_episodes

    def episode(self, idx):
        rng = np.random.default_rng([self.seed, int(idx)])
        return sample_episode(self.ds.class_indices, self.n_way, self.k, self.q, rng)

    def indices(self, i):
        """(classes [N], dataset indices [N*(K+Q)] support-first, class-major) of the i-th episode."""
        idx = self.fixed_idx if self.fixed_idx is not None else self.offset + i
        classes, support, query = self.episode(idx)
        return classes, np.concatenate([support.ravel(), query.ravel()])

    def __getitem__(self, i):
        classes, order = self.indices(i)
        x = torch.stack([self.ds[int(j)][0] for j in order])
        return x, torch.as_tensor(classes)


# --------------------------------------------------------------------------- manifest / CLI
def filelist_hash(images):
    h = hashlib.sha256()
    for c in sorted(images):
        for f in images[c]:
            h.update(f"{c}/{f}\n".encode())
    return h.hexdigest()


def build_manifest(images, download_info=None):
    # Hash file *names* only: thumbnail bytes can differ across PIL versions, names cannot.
    return {
        "n_classes": len(images),
        "n_images": sum(len(v) for v in images.values()),
        "filelist_sha256": filelist_hash(images),
        "counts": {c: len(v) for c, v in sorted(images.items())},
        "download": download_info,
    }


def summarize(counts, splits):
    print(f"classes: {len(counts)}   images: {sum(counts.values())}")
    for s in ("train", "val", "test"):
        cs = [c for c, v in splits.items() if v == s]
        n = [counts[c] for c in cs]
        print(f"  {s:5s} {len(cs):3d} classes  {sum(n):5d} images  "
              f"(per-class min/median/max = {min(n)}/{int(np.median(n))}/{max(n)})")
    test_n = np.array([counts[c] for c, v in splits.items() if v == "test"])
    print(f"test classes with <10 images (dropped from eval): {(test_n < 10).sum()} of {len(test_n)}")
    print("images/class histogram (all classes):")
    edges = [0, 5, 10, 15, 20, 30, 50, 100, 10**9]
    hist, _ = np.histogram(list(counts.values()), bins=edges)
    for lo, hi, h in zip(edges[:-1], edges[1:], hist):
        print(f"  [{lo:>3}, {'inf' if hi > 10**8 else hi:>3})  {'#' * int(h):s} {h}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-splits", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--root", default="data/SD-198-256")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--manifest", default="data_manifest.json")
    ap.add_argument("--download-info", default="data/download_info.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--novel-thresh", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.22)
    args = ap.parse_args()

    images = scan_images(args.root)
    counts = {c: len(v) for c, v in images.items()}

    if args.make_splits:
        splits = make_splits(counts, args.seed, args.novel_thresh, args.val_frac)
        save_json({"meta": dict(seed=args.seed, novel_thresh=args.novel_thresh, val_frac=args.val_frac),
                   "splits": splits, "counts": counts}, args.splits)
        info = load_json(args.download_info) if Path(args.download_info).exists() else None
        save_json(build_manifest(images, info), args.manifest)
        summarize(counts, splits)
        print(f"wrote {args.splits}, {args.manifest}")

    if args.verify:
        ref = load_json(args.manifest)
        got = build_manifest(images)
        ok = (got["n_classes"], got["n_images"], got["filelist_sha256"]) == \
             (ref["n_classes"], ref["n_images"], ref["filelist_sha256"])
        spl = load_splits(args.splits)
        ok &= set(spl["splits"]) == set(counts)
        print(f"{got['n_classes']} classes / {got['n_images']} images, filelist sha256 {got['filelist_sha256'][:16]}...")
        print("VERIFY OK" if ok else "VERIFY FAILED: local data does not match the committed manifest/splits")
        sys.exit(0 if ok else 1)

    if not (args.make_splits or args.verify):
        ap.print_help()


if __name__ == "__main__":
    main()
