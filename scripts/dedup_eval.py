"""Re-score every model on a de-duplicated test set.

    python scripts/dedup_eval.py            # -> results/dedup_test.json

SD-198 contains near-identical shots of the same patient inside a class. If one lands in the
support set and the other in the query set, the "match" is trivial and few-shot accuracy is
inflated. Here, inside each test class, images whose frozen-ImageNet embeddings have cosine
similarity > t are grouped (connected components) and only one image per group is kept.
Classes left with fewer than 10 images (k + 5 queries) are dropped and counted.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import EpisodeDataset, SD198Split, build_transforms, load_splits  # noqa: E402
from eval import MIN_TEST_IMAGES, N_QUERY_TEST, TEST_CONFIGS, TEST_SEED  # noqa: E402
from model import ProtoNet, load_model  # noqa: E402
from utils import mean_ci95, save_json  # noqa: E402

THRESHOLDS = [None, 0.95, 0.90]  # None = full test set (same scorer, for a like-for-like reference)


@torch.no_grad()
def embed(model, paths, bs=64):
    tf = build_transforms(False)
    return torch.cat([model.embed(torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]))
                      for i in range(0, len(paths), bs)]).numpy()


def components(sim, t):
    """Connected components of the graph sim > t (union-find)."""
    parent = list(range(len(sim)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in zip(*np.nonzero(np.triu(sim > t, 1))):
        parent[find(i)] = find(j)
    return [find(i) for i in range(len(sim))]


class Subset:
    """The attributes EpisodeDataset needs, restricted to kept images."""

    def __init__(self, ds, keep):
        self.classes, self.class_indices = [], []
        for c, ix in zip(ds.classes, ds.class_indices):
            k = np.array([i for i in ix if i in keep])
            if len(k) >= MIN_TEST_IMAGES:
                self.classes.append(c)
                self.class_indices.append(k)
        self.dropped = [c for c in ds.classes if c not in self.classes]
        self.n_images = int(sum(len(i) for i in self.class_indices))


def score(z, eds):
    acc = np.empty(len(eds))
    for e in range(len(eds)):
        _, sup, qry = eds.episode(e)
        d2 = ((z[qry.ravel()][:, None] - z[sup].mean(1)[None]) ** 2).sum(-1)
        acc[e] = 100 * (d2.argmin(1) == np.repeat(np.arange(eds.n_way), eds.q)).mean()
    return acc


test = SD198Split(ROOT / "data/SD-198-256", load_splits(ROOT / "splits.json"), "test", min_images=MIN_TEST_IMAGES)
models = {"frozen": ProtoNet(pretrained=True).eval()}
for s in (42, 43, 44):
    models[f"seed{s}"] = load_model(ROOT / f"runs/seed{s}/best.pt", "cpu")[0]
cache = ROOT / "results/test_embeddings.npz"
if cache.exists():
    Z = dict(np.load(cache))
else:
    Z = {m: embed(mod, test.paths) for m, mod in models.items()}
    np.savez(cache, **Z)
print("embedded", len(test.paths), "test images x", len(models), "models")

out = {}


def _score_variant(key, t, sub):
    out[key] = dict(threshold=t, images=sub.n_images, classes=len(sub.classes),
                    removed_images=len(test.paths) - sub.n_images, dropped_classes=sub.dropped, results={})
    print(f"\n[{key}] {sub.n_images} images, {len(sub.classes)} classes (dropped {len(sub.dropped)})")
    for n, k in TEST_CONFIGS:
        eds = EpisodeDataset(sub, n, k, N_QUERY_TEST, 600, TEST_SEED)
        accs = {m: score(Z[m], eds) for m in models}
        ft = np.mean([accs[f"seed{s}"] for s in (42, 43, 44)], 0)
        r = {m: round(float(a.mean()), 2) for m, a in accs.items()}
        r["finetuned_mean"] = round(float(ft.mean()), 2)
        r["finetuned_ci95"] = round(float(mean_ci95(ft)[1]), 2)
        r["gain"] = round(float((ft - accs["frozen"]).mean()), 2)
        out[key]["results"][f"{n}way_{k}shot"] = r
        print(f"  {n}-way {k}-shot  fine-tuned {r['finetuned_mean']:.2f} ±{r['finetuned_ci95']:.2f}  "
              f"frozen {r['frozen']:.2f}  gain {r['gain']:+.2f}")


for t in THRESHOLDS:
    if t is None:
        keep = set(range(len(test.paths)))
    else:
        keep = set()
        zf = Z["frozen"]
        for ix in test.class_indices:
            comp = components(zf[ix] @ zf[ix].T, t)
            seen = set()
            for i, c in zip(ix, comp):
                if c not in seen:
                    seen.add(c)
                    keep.add(int(i))
    sub = Subset(test, keep)
    key = "full" if t is None else f"cos>{t}"
    variants = [(key, sub)]
    if t == min(x for x in THRESHOLDS if x is not None):
        # control: the same surviving classes but with ALL their images, to separate the effect of removing
        # duplicates from the effect of the class mix changing (dedup drops the smallest classes)
        ctrl = Subset(test, {int(i) for c, ix in zip(test.classes, test.class_indices) if c in sub.classes for i in ix})
        variants.append((f"{key}_classes_all_images", ctrl))
    for key, sub in variants:
        _score_variant(key, t, sub)


save_json(out, ROOT / "results/dedup_test.json")
