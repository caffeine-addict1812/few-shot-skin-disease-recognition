"""Episodic training of the ResNet-18 ProtoNet.

    python src/train.py --out runs/seed42 --seed 42                       # full run (4k episodes, lr 1e-5)
    python src/train.py --out runs/smoke --episodes 500 --val-every 50    # smoke
    python src/train.py --out runs/overfit --overfit-episode              # bug catcher: loss -> ~0
"""
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import EpisodeDataset, SD198Split, build_transforms, load_eval_split, load_splits
from eval import VAL_SEED, autocast_ctx, run_eval
from model import ProtoNet, episode_labels, set_bn_eval
from utils import get_device, resolve_amp, save_json, seed_everything


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


def set_lr(opt, step, total, warmup, min_lr):
    f = lr_factor(step, total, warmup)
    for g in opt.param_groups:
        g["lr"] = min_lr + (g["base_lr"] - min_lr) * f


def save_ckpt(path, model, step, best_val, args, opt=None):
    ck = dict(model=model.state_dict(), model_cfg=model.cfg, step=step, best_val=best_val, args=vars(args))
    if opt is not None:
        ck["optimizer"] = opt.state_dict()
    torch.save(ck, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="data/SD-198-256")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--eps-per-step", type=int, default=4)
    ap.add_argument("--n-way", type=int, default=5)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--n-query", type=int, default=15)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=50, help="steps; 0 disables validation")
    ap.add_argument("--val-episodes", type=int, default=600)
    ap.add_argument("--val-way", type=int, default=5, help="validation task is fixed, independent of the training task")
    ap.add_argument("--val-shot", type=int, default=5)
    ap.add_argument("--val-query", type=int, default=15)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "none"])
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-l2-norm", action="store_true")
    ap.add_argument("--freeze-bn", action="store_true", help="keep BN running stats frozen")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overfit-episode", action="store_true",
                    help="train on one fixed episode with eval transforms (loss should go to ~0)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_num_threads(4)  # dataloader workers do the CPU work; avoid oversubscribing the cgroup quota
    device = get_device(args.device)
    amp = resolve_amp(args.amp, device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    splits = load_splits(args.splits)
    n_way, k, q, E = args.n_way, args.k_shot, args.n_query, args.eps_per_step
    train_ds = SD198Split(args.root, splits, "train", build_transforms(train=not args.overfit_episode))
    val_ds = None
    if args.val_every > 0 and not args.overfit_episode:
        val_ds = load_eval_split(args.root, splits, "val")

    model = ProtoNet(pretrained=not args.no_pretrained, l2_norm=not args.no_l2_norm, backbone=args.backbone).to(device)
    groups = [dict(params=list(model.encoder.parameters()), base_lr=args.lr_backbone, weight_decay=args.wd)]
    if model.log_scale.requires_grad:
        groups.append(dict(params=[model.log_scale], base_lr=args.lr_head, weight_decay=0.0))
    opt = torch.optim.AdamW(groups, lr=args.lr_backbone)
    scaler = torch.amp.GradScaler(device.type, enabled=amp == "fp16")

    total_steps = args.episodes // E
    warmup = min(args.warmup, max(1, total_steps // 10))
    step, best_val = 0, -1.0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, best_val = ck["step"], ck["best_val"]
        print(f"resumed from step {step} (best val {best_val:.2f})")

    eds = EpisodeDataset(train_ds, n_way, k, q, total_steps * E, seed=args.seed, offset=step * E,
                         fixed_idx=0 if args.overfit_episode else None)
    eds.n_episodes = (total_steps - step) * E
    loader = DataLoader(eds, batch_size=E, shuffle=False, num_workers=args.workers, drop_last=True,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers > 0 else None,
                        generator=torch.Generator().manual_seed(args.seed))
    labels = episode_labels(n_way, q, E, device)

    cfg = {**vars(args), "amp_resolved": amp, "total_steps": total_steps, "warmup_resolved": warmup,
           "train_classes": len(train_ds.classes), "train_images": len(train_ds),
           "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "torch": torch.__version__}
    save_json(cfg, out / "config.json")
    print(f"device={device} amp={amp} steps={total_steps} warmup={warmup} "
          f"train: {len(train_ds.classes)} classes / {len(train_ds)} images"
          + (f"  val: {len(val_ds.classes)} classes / {len(val_ds)} images" if val_ds else ""))

    log = open(out / "log.jsonl", "a")
    t0, t_win = time.time(), time.time()
    run_loss = run_acc = torch.zeros((), device=device)
    n_win = 0
    last_loss = float("nan")
    for x, _ in loader:
        if step >= total_steps:
            break
        model.train()
        if args.freeze_bn:
            set_bn_eval(model)
        set_lr(opt, step, total_steps, warmup, args.min_lr)

        x = x.to(device, non_blocking=True)
        with autocast_ctx(device, amp):
            logits = model(x, n_way, k, q)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(opt)
        scaler.update()
        step += 1

        run_loss = run_loss + loss.detach()
        run_acc = run_acc + (logits.argmax(-1) == labels).float().mean()
        n_win += 1
        if step % args.log_every == 0 or step == total_steps:
            last_loss = (run_loss / n_win).item()
            rec = dict(type="train", step=step, episodes=step * E, loss=last_loss,
                       acc=(run_acc / n_win).item() * 100, scale=model.scale.item(),
                       lr=opt.param_groups[0]["lr"],
                       img_per_s=n_win * E * n_way * (k + q) / (time.time() - t_win),
                       elapsed=time.time() - t0)
            log.write(__import__("json").dumps(rec) + "\n"), log.flush()
            print(f"step {step}/{total_steps}  loss {rec['loss']:.4f}  acc {rec['acc']:.1f}  "
                  f"scale {rec['scale']:.2f}  {rec['img_per_s']:.0f} img/s")
            run_loss = run_acc = torch.zeros((), device=device)
            n_win, t_win = 0, time.time()

        if val_ds is not None and (step % args.val_every == 0 or step == total_steps):
            r = run_eval(model, val_ds, args.val_way, args.val_shot, args.val_query, args.val_episodes,
                         VAL_SEED, device, amp, 0, E)
            rec = dict(type="val", step=step, episodes=step * E, acc=r["acc"], ci95=r["acc_ci95"],
                       f1=r["f1"], loss=r["loss"], elapsed=time.time() - t0)
            log.write(__import__("json").dumps(rec) + "\n"), log.flush()
            improved = r["acc"] > best_val
            print(f"  [val] step {step}: acc {r['acc']:.2f} +- {r['acc_ci95']:.2f}"
                  + ("  * best" if improved else ""))
            if improved:
                best_val = r["acc"]
                save_ckpt(out / "best.pt", model, step, best_val, args)
            save_ckpt(out / "last.pt", model, step, best_val, args, opt)

    if val_ds is None:  # no validation (smoke / overfit): keep the final weights
        save_ckpt(out / "best.pt", model, step, best_val, args)
    save_json(dict(steps=step, best_val=best_val, final_train_loss=last_loss,
                   wall_clock_s=time.time() - t0, device=cfg["device_name"]), out / "done.json")
    print(f"done: {step} steps in {(time.time() - t0) / 60:.1f} min, best val {best_val:.2f}")


if __name__ == "__main__":
    main()
