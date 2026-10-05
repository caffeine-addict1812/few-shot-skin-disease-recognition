"""Summarise the follow-up experiments (scripts/run_extra.sh) into results/extra_summary.json.

    python scripts/summarize_extra.py

  repro   the main seed-42 run retrained on a second VM: how much do GPU runs vary on their own?
  splits  the same recipe on three train/val partitions of the base classes (test classes identical)
  r50     ResNet-50 vs ResNet-18, paired over the same 600 test episodes per setting
Missing parts are skipped, so this can run while experiments are still going.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils import load_json, save_json  # noqa: E402

R = ROOT / "results"
SEEDS = (42, 43, 44)
rng = np.random.default_rng(0)


def per_ep(path):
    return {f"{r['n_way']}way_{r['k_shot']}shot": np.array(r["acc_per_episode"]) for r in load_json(path)["results"]}


def seed_mean(files):
    """Per-setting: mean accuracy over seeds, and the per-episode accuracy averaged over seeds."""
    eps = [per_ep(f) for f in files]
    return {k: dict(acc=float(np.mean([e[k].mean() for e in eps])), per_seed=[float(e[k].mean()) for e in eps],
                    _ep=np.mean([e[k] for e in eps], 0)) for k in eps[0]}


def boot(d, n=10000):
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def best_val(run):
    return load_json(run / "done.json")["best_val"]


def strip(d):
    return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in d.items()}


out = {}
frozen = per_ep(R / "frozen_test.json")

# ------------------------------------------------------------------ repro
if (R / "repro/seed42_test.json").exists():
    a, b = per_ep(R / "seed42_test.json"), per_ep(R / "repro/seed42_test.json")
    out["repro"] = dict(
        best_val=dict(original=best_val(ROOT / "runs/seed42"), repro=best_val(ROOT / "runs/repro/seed42")),
        test={k: dict(original=float(a[k].mean()), repro=float(b[k].mean()), diff=float(b[k].mean() - a[k].mean()),
                      episodes_differing=int((a[k] != b[k]).sum())) for k in a})

# ------------------------------------------------------------------ splits
splits = {}
for sp, rdir, runs in [(42, R, ROOT / "runs"), (7, R / "split7", ROOT / "runs/split7"), (123, R / "split123", ROOT / "runs/split123")]:
    files = [rdir / f"seed{s}_test.json" for s in SEEDS]
    if not all(f.exists() for f in files):
        continue
    fv = load_json((R / "frozen_val.json") if sp == 42 else (rdir / "frozen_val.json"))["5way_5shot"]["acc"]
    sm = seed_mean(files)
    splits[sp] = dict(test=sm, best_val=[best_val(runs / f"seed{s}") for s in SEEDS], frozen_val=fv,
                      gain_vs_frozen={k: float(sm[k]["acc"] - frozen[k].mean()) for k in sm})
if splits:
    keys = list(next(iter(splits.values()))["test"])
    across = {k: dict(mean=float(np.mean([splits[s]["test"][k]["acc"] for s in splits])),
                      std_over_splits=float(np.std([splits[s]["test"][k]["acc"] for s in splits], ddof=1)) if len(splits) > 1 else 0.0,
                      min=float(min(splits[s]["test"][k]["acc"] for s in splits)),
                      max=float(max(splits[s]["test"][k]["acc"] for s in splits)),
                      frozen=float(frozen[k].mean()))
              for k in keys}
    out["splits"] = dict(per_split={str(s): dict(v, test=strip(v["test"])) for s, v in splits.items()}, across=across)

# ------------------------------------------------------------------ ResNet-50
r50_files = [R / f"r50/seed{s}_test.json" for s in SEEDS]
if all(f.exists() for f in r50_files):
    chosen = (R / "r50/chosen_lr.txt").read_text().strip().split("=")[-1].strip()
    sel = {lr: best_val(ROOT / f"runs/r50/lr{lr}_seed42") for lr in ("1e-5", "3e-5")}
    r50, r18 = seed_mean(r50_files), seed_mean([R / f"seed{s}_test.json" for s in SEEDS])
    fz50 = per_ep(R / "r50/frozen_test.json")
    cmp = {}
    for k in r50:
        d = r50[k]["_ep"] - r18[k]["_ep"]
        cmp[k] = dict(r50=r50[k]["acc"], r50_per_seed=r50[k]["per_seed"], r18=r18[k]["acc"],
                      frozen_r50=float(fz50[k].mean()), frozen_r18=float(frozen[k].mean()),
                      diff=float(d.mean()), diff_boot_ci95=boot(d),
                      r50_gain_vs_own_frozen=float(r50[k]["acc"] - fz50[k].mean()))
    out["r50"] = dict(selection_val=sel, chosen_lr=chosen,
                      frozen_val=load_json(R / "r50/frozen_val.json")["5way_5shot"]["acc"],
                      best_val=[best_val(ROOT / f"runs/r50/lr{chosen}_seed{s}") for s in SEEDS], test=cmp,
                      device=load_json(ROOT / f"runs/r50/lr{chosen}_seed42/done.json")["device"],
                      torch=load_json(ROOT / f"runs/r50/lr{chosen}_seed42/config.json").get("torch"),
                      wall_min=[load_json(ROOT / f"runs/r50/lr{chosen}_seed{s}/done.json")["wall_clock_s"] / 60 for s in SEEDS])

save_json(out, R / "extra_summary.json")
print(json.dumps(out, indent=1)[:4000])
