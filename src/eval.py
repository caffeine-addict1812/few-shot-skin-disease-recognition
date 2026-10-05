"""Episodic evaluation on fixed episode lists.

    python src/eval.py --ckpt runs/seed42/best.pt --out results/seed42_test.json
    python src/eval.py --frozen --out results/frozen_test.json         # ImageNet ResNet-18, no fine-tuning
    python src/eval.py --aggregate results/seed4{2,3,4}_test.json --frozen-file results/frozen_test.json \
        --out results/summary.json
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import EpisodeDataset, load_eval_split, load_splits
from model import ProtoNet, episode_labels, load_model
from utils import get_device, load_json, mean_ci95, resolve_amp, save_json

VAL_SEED, TEST_SEED = 1000, 2000            # fixed: every checkpoint / run sees identical episodes
TEST_CONFIGS = [(5, 1), (5, 5), (2, 1), (2, 5)]   # (n_way, k_shot)
N_QUERY_TEST = 5                            # novel classes have 10-19 images: no room for 15 queries
MIN_TEST_IMAGES = 10                        # 5-shot + 5 queries, drawn without replacement


def autocast_ctx(device, amp):
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=amp != "none")


def episode_metrics(logits, labels, n_way):
    """logits [E, M, N], labels [E, M] -> per-episode accuracy [E] and macro-F1 over the N ways [E]."""
    pred = logits.argmax(-1)
    acc = (pred == labels).float().mean(1)
    f1 = []
    for c in range(n_way):
        tp = ((pred == c) & (labels == c)).sum(1).float()
        fp = ((pred == c) & (labels != c)).sum(1).float()
        fn = ((pred != c) & (labels == c)).sum(1).float()
        f1.append(2 * tp / (2 * tp + fp + fn).clamp(min=1))
    return acc, torch.stack(f1, 1).mean(1)


def episode_batches(dataset, eds, eps_per_batch, device, workers=0):
    """Yield [E, N*(K+Q), 3, H, W] float batches on `device`.

    A CachedSplit lives in memory (and is moved to `device` once), so episodes are gathered by indexing
    it directly: no JPEG decoding, no per-episode CPU stacking. Anything else goes through a DataLoader.
    """
    if hasattr(dataset, "images"):
        dataset.images = dataset.images.to(device)
        for s in range(0, len(eds), eps_per_batch):
            order = np.stack([eds.indices(i)[1] for i in range(s, min(s + eps_per_batch, len(eds)))])
            yield dataset.images[torch.as_tensor(order, device=device)].float()
    else:
        loader = DataLoader(eds, batch_size=eps_per_batch, shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda")
        for x, _ in loader:
            yield x.to(device, non_blocking=True)


@torch.no_grad()
def run_eval(model, dataset, n_way, k_shot, n_query, n_episodes, seed, device,
             amp="none", workers=0, eps_per_batch=4):
    model.eval()
    eds = EpisodeDataset(dataset, n_way, k_shot, n_query, n_episodes, seed)
    labels = episode_labels(n_way, n_query, eps_per_batch, device)
    accs, f1s, losses = [], [], []
    for x in episode_batches(dataset, eds, eps_per_batch, device, workers):
        with autocast_ctx(device, amp):
            logits = model(x, n_way, k_shot, n_query)
        lab = labels[: len(x)]
        a, f = episode_metrics(logits, lab, n_way)
        loss = F.cross_entropy(logits.flatten(0, 1), lab.flatten(), reduction="none")
        accs.append(a.cpu()), f1s.append(f.cpu()), losses.append(loss.view(len(x), -1).mean(1).cpu())
    acc = torch.cat(accs).numpy() * 100
    f1 = torch.cat(f1s).numpy() * 100
    acc_m, acc_ci = mean_ci95(acc)
    f1_m, f1_ci = mean_ci95(f1)
    return dict(n_way=n_way, k_shot=k_shot, n_query=n_query, n_episodes=n_episodes, seed=seed,
                acc=acc_m, acc_ci95=acc_ci, f1=f1_m, f1_ci95=f1_ci,
                loss=float(torch.cat(losses).mean()), acc_per_episode=np.round(acc, 2).tolist())


def evaluate_test(model, root, splits, device, amp="none", n_episodes=600, configs=TEST_CONFIGS,
                  workers=0):
    ds = load_eval_split(root, splits, "test", min_images=MIN_TEST_IMAGES)
    results = []
    for n_way, k in configs:
        t0 = time.time()
        r = run_eval(model, ds, n_way, k, N_QUERY_TEST, n_episodes, TEST_SEED, device, amp, workers)
        print(f"  {n_way}-way {k}-shot: acc {r['acc']:.2f} +- {r['acc_ci95']:.2f}   "
              f"macro-F1 {r['f1']:.2f} +- {r['f1_ci95']:.2f}   ({time.time() - t0:.0f}s)")
        results.append(r)
    return dict(results=results, n_test_classes=len(ds.classes), n_test_images=len(ds),
                dropped_test_classes=ds.dropped_classes)


def aggregate(seed_files, frozen_file=None):
    runs = [load_json(f) for f in seed_files]
    out = {"n_seeds": len(runs), "seeds": [r.get("seed") for r in runs], "configs": {}}
    for i, cfg in enumerate(runs[0]["results"]):
        key = f"{cfg['n_way']}way_{cfg['k_shot']}shot"
        accs = [r["results"][i]["acc"] for r in runs]
        f1s = [r["results"][i]["f1"] for r in runs]
        sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        out["configs"][key] = dict(
            acc_mean=float(np.mean(accs)), acc_std_over_seeds=sd(accs),
            acc_ci95_episodes=float(np.mean([r["results"][i]["acc_ci95"] for r in runs])),
            f1_mean=float(np.mean(f1s)), f1_std_over_seeds=sd(f1s), per_seed_acc=accs)
    if frozen_file:
        fz = load_json(frozen_file)
        out["frozen_baseline"] = {f"{r['n_way']}way_{r['k_shot']}shot":
                                  dict(acc=r["acc"], acc_ci95=r["acc_ci95"], f1=r["f1"])
                                  for r in fz["results"]}
    out["n_test_classes"] = runs[0]["n_test_classes"]
    out["dropped_test_classes"] = runs[0]["dropped_test_classes"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--frozen", action="store_true", help="ImageNet-pretrained, not fine-tuned")
    ap.add_argument("--backbone", default="resnet18", help="backbone for --frozen")
    ap.add_argument("--aggregate", nargs="+", metavar="RESULT_JSON")
    ap.add_argument("--frozen-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="data/SD-198-256")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "none"])
    args = ap.parse_args()

    if args.aggregate:
        summary = aggregate(args.aggregate, args.frozen_file)
        save_json(summary, args.out)
        for k, v in summary["configs"].items():
            print(f"{k}: {v['acc_mean']:.2f} +- {v['acc_std_over_seeds']:.2f} (std over {summary['n_seeds']} seeds)")
        return

    device = get_device(args.device)
    amp = resolve_amp(args.amp, device)
    splits = load_splits(args.splits)
    meta = {}
    if args.frozen:
        model = ProtoNet(pretrained=True, backbone=args.backbone).to(device).eval()
        meta["model"] = "imagenet_frozen" if args.backbone == "resnet18" else f"imagenet_frozen_{args.backbone}"
    else:
        model, ckpt = load_model(args.ckpt, device)
        meta.update(ckpt=args.ckpt, step=ckpt.get("step"), val_acc=ckpt.get("best_val"),
                    seed=ckpt.get("args", {}).get("seed"))
    print(f"device={device} amp={amp} {meta}")
    res = evaluate_test(model, args.root, splits, device, amp, args.episodes, workers=args.workers)
    save_json({**meta, **res}, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
