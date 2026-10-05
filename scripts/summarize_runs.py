"""Summarise training runs from their log.jsonl: best/final validation accuracy and the settings that differ.

    python scripts/summarize_runs.py runs/sweep/*  [runs/_evidence/*]
"""
import json
import sys
from pathlib import Path

KEYS = ["lr_backbone", "lr_head", "episodes", "n_way", "n_query", "freeze_bn", "warmup_resolved"]


def summarize(d):
    d = Path(d)
    log, cfg_path = d / "log.jsonl", d / "config.json"
    if not log.exists():
        return None
    recs = [json.loads(line) for line in open(log) if line.strip()]
    val = [r for r in recs if r["type"] == "val"]
    train = [r for r in recs if r["type"] == "train"]
    if not val:
        return None
    best = max(val, key=lambda r: r["acc"])
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}
    return dict(name=d.name, best_val=best["acc"], ci=best["ci95"], best_step=best["step"],
                final_val=val[-1]["acc"], last_step=val[-1]["step"],
                train_acc=train[-1]["acc"] if train else float("nan"), cfg=cfg)


def main():
    rows = [r for r in (summarize(p) for p in sys.argv[1:]) if r]
    rows.sort(key=lambda r: -r["best_val"])
    print(f"{'run':34s} {'best val':>14s} {'@step':>6s} {'final val':>10s} {'train acc':>9s}  settings")
    for r in rows:
        c = r["cfg"]
        sett = " ".join(f"{k}={c[k]}" for k in KEYS if k in c)
        print(f"{r['name']:34s} {r['best_val']:7.2f} +-{r['ci']:4.2f} {r['best_step']:6d} "
              f"{r['final_val']:10.2f} {r['train_acc']:9.1f}  {sett}")


if __name__ == "__main__":
    main()
