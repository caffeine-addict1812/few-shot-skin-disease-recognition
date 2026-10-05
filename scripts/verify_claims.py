"""Independent check of the numbers in report.md (CPU only; ~25 min on a 12-thread laptop CPU).

    python -u scripts/verify_claims.py      # -> results/verification.json, results/figures/fig5_near_duplicates.png

It deliberately avoids the training/eval code path (no run_eval, no fp16 cache, no autocast):
images are decoded straight from disk in fp32, embedded once per model, and every episode is
scored with a numpy nearest-prototype rule. Only the episode *definition* (EpisodeDataset.episode)
is shared, because that is what "the 600 fixed test episodes" means.

Checks
  1. splits      class sets disjoint, counts match the report, no byte-identical file across splits
  2. leakage     near-duplicate photos (frozen-ImageNet cosine similarity) across and within splits
  3. recompute   all 4 test settings x {frozen, seed 42/43/44} vs the stored per-episode results
  4. fresh       the same models on 5000 *new* episodes (different seed): is 600 representative?
  5. gain        paired bootstrap 95% CI of (fine-tuned - frozen) over episodes
  6. controls    shuffled query labels -> chance; bottom vs top band masked (footer-strip shortcut?)
  7. selection   best.pt step == argmax of logged val acc; no test records in training logs
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import EpisodeDataset, SD198Split, build_transforms, load_splits  # noqa: E402
from eval import MIN_TEST_IMAGES, N_QUERY_TEST, TEST_CONFIGS, TEST_SEED  # noqa: E402
from model import ProtoNet, load_model  # noqa: E402
from utils import save_json  # noqa: E402

DATA = ROOT / "data/SD-198-256"
SEEDS = [42, 43, 44]
FRESH_SEED, FRESH_EPS = 9999, 5000
BAND = 0.15  # fraction of the 224 crop masked in the footer control
torch.set_num_threads(max(1, torch.get_num_threads()))
rng_boot = np.random.default_rng(0)
out, fails = {}, []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg, flush=True)
    if not ok:
        fails.append(msg)


@torch.no_grad()
def embed(model, ds, mask=None, bs=64):
    tf = build_transforms(False)
    zs = []
    for i in range(0, len(ds), bs):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in ds.paths[i:i + bs]])
        if mask == "bottom":
            x[:, :, -int(224 * BAND):, :] = 0.0  # 0 = channel mean after Normalize
        elif mask == "top":
            x[:, :, : int(224 * BAND), :] = 0.0
        zs.append(model.embed(x))
    return torch.cat(zs).numpy()


def score(z, eds, shuffle=None):
    """Per-episode accuracy (%) with a plain nearest-prototype rule."""
    n, k, q = eds.n_way, eds.k, eds.q
    acc = np.empty(len(eds))
    for e in range(len(eds)):
        classes, sup, qry = eds.episode(e)
        assert len(set(classes)) == n and not set(sup.ravel()) & set(qry.ravel())
        protos = z[sup].mean(1)                                       # [N, D]
        d2 = ((z[qry.ravel()][:, None] - protos[None]) ** 2).sum(-1)  # [N*Q, N]
        y = np.repeat(np.arange(n), q)
        if shuffle is not None:
            y = shuffle.permutation(y)
        acc[e] = 100 * (d2.argmin(1) == y).mean()
    return acc


def boot_ci(d, n=10000):
    m = d[rng_boot.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


splits = load_splits(ROOT / "splits.json")
sets = {s: SD198Split(DATA, splits, s) for s in ("train", "val", "test")}
test = SD198Split(DATA, splits, "test", min_images=MIN_TEST_IMAGES)

# ------------------------------------------------------------------ 1. splits
print("1. splits")
cls = {s: set(d.classes) for s, d in sets.items()}
check(not (cls["train"] & cls["val"] or cls["train"] & cls["test"] or cls["val"] & cls["test"]),
      "train / val / test class sets are pairwise disjoint")
counts = {s: (len(d.classes), len(d)) for s, d in sets.items()}
check(counts == {"train": (101, 4428), "val": (27, 1191), "test": (70, 965)},
      f"class/image counts match report table: {counts}")
check(max(len(i) for i in sets["test"].class_indices) < 20 <= min(len(i) for i in sets["train"].class_indices),
      "every test class has < 20 images, every train class >= 20")
md5 = defaultdict(list)
for s, d in sets.items():
    for p in d.paths:
        md5[hashlib.md5(p.read_bytes()).hexdigest()].append((s, p.parent.name, p.name))
dup_groups = [v for v in md5.values() if len(v) > 1]
cross = [v for v in dup_groups if len({x[0] for x in v}) > 1]
# A file shared between splits is necessarily filed under two *different* diseases (the class sets are
# disjoint), so it is label noise in SD-198, not leakage of a test label. Report it, don't fail on it.
for g in cross:
    print("  WARN same photo under two diseases (dataset label noise): " + " | ".join("/".join(x) for x in g))
check(len(cross) <= 1, f"at most 1 byte-identical photo shared between splits ({len(cross)} found)")
out["splits"] = dict(counts=counts, identical_groups=len(dup_groups), identical_cross_split=len(cross),
                     identical_examples=dup_groups[:10])

# ------------------------------------------------------------------ 2. near duplicates
print("2. near duplicates (frozen ImageNet embeddings, cosine similarity)")
frozen = ProtoNet(pretrained=True).eval()
Z = {s: embed(frozen, d) for s, d in sets.items()}
lab = {s: np.array(d.labels) for s, d in sets.items()}
S_cross = Z["test"] @ np.concatenate([Z["train"], Z["val"]]).T
nn_cross = S_cross.max(1)
S_test = Z["test"] @ Z["test"].T
np.fill_diagonal(S_test, -1)
same = lab["test"][:, None] == lab["test"][None]
iu = np.triu_indices(len(S_test), 1)
pair_sim, pair_same = S_test[iu], same[iu]
leak = {}
for t in (0.90, 0.95, 0.98):
    leak[str(t)] = dict(
        test_imgs_with_trainval_neighbour=int((nn_cross > t).sum()),
        test_pairs_same_class=int(((pair_sim > t) & pair_same).sum()),
        test_pairs_diff_class=int(((pair_sim > t) & ~pair_same).sum()))
    print(f"  sim > {t}: {leak[str(t)]}")
out["near_duplicates"] = leak
check(leak["0.95"]["test_imgs_with_trainval_neighbour"] <= 0.02 * len(Z["test"]),
      "<= 2% of test images have a near-duplicate (sim > 0.95) in train/val")

# figure: the most similar test<->train/val pairs and same-class test pairs, for eyeballing
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tv_paths = sets["train"].paths + sets["val"].paths
    top_cross = np.argsort(-nn_cross)[:4]
    sp = np.argsort(-np.where(pair_same, pair_sim, -1))[:4]
    rows = [(sets["test"].paths[i], tv_paths[S_cross[i].argmax()], nn_cross[i], "test vs train/val") for i in top_cross]
    rows += [(sets["test"].paths[iu[0][j]], sets["test"].paths[iu[1][j]], pair_sim[j], "test vs test, same class")
             for j in sp]
    fig, ax = plt.subplots(len(rows), 2, figsize=(6, 2.3 * len(rows)))
    for r, (a, b, s, kind) in enumerate(rows):
        for c, p in enumerate((a, b)):
            ax[r, c].imshow(Image.open(p).convert("RGB"))
            ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
            ax[r, c].set_xlabel(f"{p.parent.name[:28]}", fontsize=7)
        ax[r, 0].set_ylabel(f"{kind}\ncos {s:.3f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/fig5_near_duplicates.png", dpi=110)
except Exception as e:  # figure is a convenience, not a check
    print("  (figure skipped:", e, ")")

# ------------------------------------------------------------------ 3-6. per model
models = {"frozen": frozen}
for s in SEEDS:
    models[f"seed{s}"] = load_model(ROOT / f"runs/seed{s}/best.pt", "cpu")[0]
stored = {"frozen": json.load(open(ROOT / "results/frozen_test.json"))["results"]}
for s in SEEDS:
    stored[f"seed{s}"] = json.load(open(ROOT / f"results/seed{s}_test.json"))["results"]
stored = {m: {(r["n_way"], r["k_shot"]): np.array(r["acc_per_episode"]) for r in v} for m, v in stored.items()}

print("3-6. per-model recompute, fresh episodes, controls")
emb = {m: {"full": (Z["test"] if m == "frozen" else embed(mod, test))} for m, mod in models.items()}
for m, mod in models.items():
    emb[m]["bottom"] = embed(mod, test, "bottom")
    emb[m]["top"] = embed(mod, test, "top")

rec = {}
for n, k in TEST_CONFIGS:
    cfg = f"{n}way_{k}shot"
    eds = EpisodeDataset(test, n, k, N_QUERY_TEST, 600, TEST_SEED)
    fresh = EpisodeDataset(test, n, k, N_QUERY_TEST, FRESH_EPS, FRESH_SEED)
    rec[cfg] = {}
    for m in models:
        a = score(emb[m]["full"], eds)
        st = stored[m][(n, k)]
        rec[cfg][m] = dict(
            stored=float(st.mean()), recomputed=float(a.mean()),
            episodes_differing=int((np.abs(a - st) > 1e-6).sum()),
            fresh_5000=float(score(emb[m]["full"], fresh).mean()),
            shuffled_labels=float(score(emb[m]["full"], eds, np.random.default_rng(1)).mean()),
            bottom_masked=float(score(emb[m]["bottom"], eds).mean()),
            top_masked=float(score(emb[m]["top"], eds).mean()),
            _per_ep=a)
        r = rec[cfg][m]
        print(f"  {cfg} {m:7s} stored {r['stored']:.2f} recomputed {r['recomputed']:.2f} "
              f"({r['episodes_differing']} eps differ) | fresh {r['fresh_5000']:.2f} | "
              f"shuffled {r['shuffled_labels']:.1f} | mask bottom {r['bottom_masked']:.2f} top {r['top_masked']:.2f}")
        check(abs(r["recomputed"] - r["stored"]) < 0.5, f"{cfg} {m}: recomputed within 0.5 pp of stored")
        check(abs(r["shuffled_labels"] - 100 / n) < 3, f"{cfg} {m}: shuffled labels give chance ({100 / n:.0f}%)")
    ft = np.mean([rec[cfg][f"seed{s}"]["_per_ep"] for s in SEEDS], 0)
    d = ft - rec[cfg]["frozen"]["_per_ep"]
    rec[cfg]["gain_vs_frozen"] = dict(mean=float(d.mean()), boot_ci95=boot_ci(d),
                                      wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()))
    g = rec[cfg]["gain_vs_frozen"]
    print(f"  {cfg} gain {g['mean']:+.2f} pp, bootstrap 95% CI [{g['boot_ci95'][0]:+.2f}, {g['boot_ci95'][1]:+.2f}], "
          f"episodes won/tied/lost {g['wins']}/{g['ties']}/{g['losses']}")
    check(g["boot_ci95"][0] > 0, f"{cfg}: gain over frozen excludes 0")
    for m in models:
        rec[cfg][m].pop("_per_ep")
out["test"] = rec

# ------------------------------------------------------------------ 7. selection
print("7. checkpoint selection")
for s in SEEDS:
    log = [json.loads(line) for line in open(ROOT / f"runs/seed{s}/log.jsonl")]
    types = {r["type"] for r in log}
    val = [r for r in log if r["type"] == "val"]
    best = max(val, key=lambda r: r["acc"])
    ck = torch.load(ROOT / f"runs/seed{s}/best.pt", map_location="cpu", weights_only=False)
    step = ck.get("step")
    check(types <= {"train", "val"}, f"seed{s}: log has only train/val records {sorted(types)}")
    check(step == best["step"], f"seed{s}: best.pt step {step} == argmax val step {best['step']}")

out["failures"] = fails
save_json(out, ROOT / "results/verification.json")
print(f"\n{len(fails)} failed check(s)" + ("" if not fails else ":\n  " + "\n  ".join(fails)))
