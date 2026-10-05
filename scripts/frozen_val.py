"""Frozen ImageNet backbone + prototypes on the VALIDATION classes (the selection metric of train.py).

    python scripts/frozen_val.py                                   # -> results/frozen_val.json
    python scripts/frozen_val.py --backbone resnet50 --out results/r50/frozen_val.json
    python scripts/frozen_val.py --splits splits_s7.json --out results/split7/frozen_val.json

This is the reference line for the validation curves: fine-tuning has only helped if val accuracy
ends above this number.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import load_eval_split, load_splits  # noqa: E402
from eval import VAL_SEED, run_eval  # noqa: E402
from model import ProtoNet  # noqa: E402
from utils import get_device, resolve_amp, save_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--splits", default="splits.json")
ap.add_argument("--backbone", default="resnet18")
ap.add_argument("--out", default="results/frozen_val.json")
args = ap.parse_args()

device = get_device()
amp = resolve_amp("auto", device)
val = load_eval_split("data/SD-198-256", load_splits(args.splits), "val")
model = ProtoNet(pretrained=True, backbone=args.backbone).to(device).eval()

out = {}
for n_way, k in [(5, 5), (5, 1)]:
    r = run_eval(model, val, n_way, k, 15, 600, VAL_SEED, device, amp)
    r.pop("acc_per_episode")
    out[f"{n_way}way_{k}shot"] = r
    print(f"frozen {args.backbone} | val ({args.splits}) | {n_way}-way {k}-shot, 15 queries: acc {r['acc']:.2f} +- {r['acc_ci95']:.2f}")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
save_json(out, args.out)
