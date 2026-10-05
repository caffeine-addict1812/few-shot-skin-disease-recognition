"""Build the data behind the project site's demo (site/demo.html).

    python scripts/build_showcase.py        # -> site/assets/data.js, site/assets/thumbs.jpg, results/site_cache.npz

Lab:   the 965 test images (70 never-seen diseases), their frozen-ImageNet and fine-tuned (seed 42) embeddings as
       float16, and the report's own 600 fixed test episodes for 5-way 1-shot and 5-way 5-shot.
Galaxy: (needs results/all_embeddings.npz from scripts/embed_all.py) a 3-D t-SNE of all 6,584 photos under the
       fine-tuned model, plus a frozen-model t-SNE initialised from it so the morph between them follows real
       neighbourhoods. results/site_cache.npz feeds the live-model server (scripts/serve_site.py).
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import EpisodeDataset, SD198Split, load_splits  # noqa: E402
from eval import MIN_TEST_IMAGES, N_QUERY_TEST, TEST_SEED  # noqa: E402
from utils import load_json  # noqa: E402

OUT = ROOT / "site" / "assets"
OUT.mkdir(exist_ok=True)
THUMB, COLS = 96, 32


def f16(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f2").tobytes()).decode()


def pretty(c):
    return c.replace("_", " ").replace("  ", " ").strip()


splits = load_splits(ROOT / "splits.json")
test = SD198Split(ROOT / "data/SD-198-256", splits, "test", min_images=MIN_TEST_IMAGES)
Z = np.load(ROOT / "results/test_embeddings.npz")
assert Z["seed42"].shape[0] == len(test.paths)

# ---- thumbnails sprite (centre square crop, same framing as the model's centre crop)
rows = (len(test.paths) + COLS - 1) // COLS
sheet = Image.new("RGB", (COLS * THUMB, rows * THUMB), (20, 22, 28))
for i, p in enumerate(test.paths):
    im = Image.open(p).convert("RGB")
    s = min(im.size)
    im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width - s) // 2 + s, (im.height - s) // 2 + s))
    sheet.paste(im.resize((THUMB, THUMB), Image.BICUBIC), ((i % COLS) * THUMB, (i // COLS) * THUMB))
sheet.save(OUT / "thumbs.jpg", quality=72, optimize=True, progressive=True)

# ---- the report's fixed test episodes
episodes = {}
for k in (1, 5):
    eds = EpisodeDataset(test, 5, k, N_QUERY_TEST, 600, TEST_SEED)
    eps = []
    for e in range(600):
        classes, sup, qry = eds.episode(e)
        eps.append([int(c) for c in classes] + [int(i) for i in sup.ravel()] + [int(i) for i in qry.ravel()])
    episodes[f"5way_{k}shot"] = eps

stored = {m: {f"{r['n_way']}way_{r['k_shot']}shot": r["acc"] for r in load_json(f)["results"]}
          for m, f in [("finetuned", ROOT / "results/seed42_test.json"), ("frozen", ROOT / "results/frozen_test.json")]}
ckpt_scale = 14.290221214294434  # learned temperature of checkpoints/best.pt (exp(log_scale))

# share of never-seen photos whose nearest other photo (512-d, squared Euclidean) shows the same disease
lab = np.array(test.labels)
nn_same = {}
for m, key in (("finetuned", "seed42"), ("frozen", "frozen")):
    z = Z[key].astype(np.float64)
    d = (z * z).sum(1)[:, None] + (z * z).sum(1)[None] - 2 * z @ z.T
    np.fill_diagonal(d, np.inf)
    nn_same[m] = float((lab[d.argmin(1)] == lab).mean() * 100)

data = dict(
    note="SD-198 research demo. Not a diagnostic tool.",
    classes=[pretty(c) for c in test.classes],
    labels=[int(x) for x in test.labels],
    files=[str(p.relative_to(ROOT / "data/SD-198-256")) for p in test.paths],
    thumb=dict(size=THUMB, cols=COLS),
    dim=int(Z["seed42"].shape[1]),
    emb=dict(finetuned=f16(Z["seed42"]), frozen=f16(Z["frozen"])),
    scale=dict(finetuned=ckpt_scale, frozen=10.0),
    episodes=episodes,
    stored_acc=stored,
    nn_same=nn_same,
    headline=dict(  # report headline numbers (mean of 3 training seeds), shown in the story's closing step
        trained={k: v["acc_mean"] for k, v in load_json(ROOT / "results/summary.json")["configs"].items()},
        frozen={k: v["acc"] for k, v in load_json(ROOT / "results/summary.json")["frozen_baseline"].items()}),
)

# ---- atlas (optional until embed_all.py has run)
allz = ROOT / "results/all_embeddings.npz"
if allz.exists():
    from sklearn.manifold import TSNE
    A = np.load(allz)
    order = ("train", "val", "test")
    ft = np.concatenate([A[f"{s}_finetuned"] for s in order])
    fz = np.concatenate([A[f"{s}_frozen"] for s in order])
    split_id = np.concatenate([np.full(len(A[f"{s}_labels"]), i) for i, s in enumerate(order)])
    names = {s: [pretty(c) for c in A[f"{s}_classes"]] for s in order}
    offsets = np.cumsum([0] + [len(names[s]) for s in order])[:-1]
    cls = np.concatenate([A[f"{s}_labels"] + off for s, off in zip(order, offsets)])
    assert list(A["test_classes"]) == list(test.classes)
    assert list(A["test_paths"]) == data["files"], "atlas test order must match the thumbnail sprite"  # hover shows the right photo
    # 3-D t-SNE for the galaxy. The frozen layout starts from the trained one, so the morph between them follows
    # real changes in neighbourhood rather than two unrelated random layouts. Cached: t-SNE takes a few minutes.
    cache = ROOT / "results/tsne3d.npz"
    if cache.exists() and np.load(cache)["n"] == len(ft):
        y_ft, y_fz = np.load(cache)["ft"], np.load(cache)["fz"]
    else:
        y_ft = TSNE(3, perplexity=30, init="pca", random_state=0).fit_transform(ft)
        y_fz = TSNE(3, perplexity=30, init=y_ft / np.std(y_ft[:, 0]) * 1e-4, random_state=0).fit_transform(fz)
        np.savez(cache, ft=y_ft, fz=y_fz, n=len(ft))

    def norm(y):
        y = y - y.mean(0)
        return np.round(y / np.abs(y).max() * 1000).astype(int)  # [-1000, 1000] per axis triple
    test_start = offsets[2]
    # per-disease relative spread in the real 512-d space: mean squared distance of its photos to their own mean,
    # divided by the same quantity for all 6,584 photos. Training spreads the whole space out, so raw spread grows
    # for almost every disease; what matters for nearest-prototype classification is spread relative to the space.
    spread = {}
    for name, Y in (("finetuned", ft), ("frozen", fz)):
        tot = ((Y - Y.mean(0)) ** 2).sum(1).mean()
        spread[name] = [round(float(((Y[cls == c] - Y[cls == c].mean(0)) ** 2).sum(1).mean() / tot), 4)
                        for c in range(int(cls.max()) + 1)]
    ch = [(spread["finetuned"][c] - spread["frozen"][c]) / spread["frozen"][c] for c in range(test_start, len(spread["frozen"]))]
    spread_summary = dict(tighter=int(sum(x < 0 for x in ch)), of=len(ch), median_change=float(np.median(ch) * 100))
    data["atlas"] = dict(
        names=names["train"] + names["val"] + names["test"],
        split=split_id.tolist(), cls=cls.tolist(), test_class_offset=int(test_start),
        test_index=[int(i) - len(A["train_labels"]) - len(A["val_labels"]) if s == 2 else -1
                    for i, s in enumerate(split_id)],
        dims=3, finetuned=norm(y_ft).ravel().tolist(), frozen=norm(y_fz).ravel().tolist(), spread=spread,
        spread_summary=spread_summary,
        paths=[str(x) for s in order for x in A[f"{s}_paths"]])
    # everything the live-model server needs to place a new photo in the galaxy and name its neighbours
    np.savez(ROOT / "results/site_cache.npz", emb_finetuned=ft, emb_frozen=fz,
             xyz_finetuned=norm(y_ft) / 1000.0, xyz_frozen=norm(y_fz) / 1000.0, cls=cls, split=split_id,
             names=np.array(data["atlas"]["names"]), paths=np.array(data["atlas"]["paths"]), test_start=test_start)

js = "window.SHOWCASE = " + json.dumps(data, separators=(",", ":")) + ";\n"
(OUT / "data.js").write_text(js)
print(f"wrote site/assets/data.js ({len(js) / 1e6:.1f} MB), site/assets/thumbs.jpg "
      f"({(OUT / 'thumbs.jpg').stat().st_size / 1e6:.1f} MB), atlas={'atlas' in data}")
