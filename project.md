# Skin Lesion Few-Shot Classification — Project Plan (SD-198 + Prototypical Networks)

> Goal: a working, demonstrable class project. Not SOTA, not novel. Prove the pipeline works end-to-end:
> notebook + short report + live demo. Efficient first: **1× GPU, <2h total**, 2–3h is the hard max.

## 0. Revisions after planning (2026-10-03) — these override the sections they name

| Where | Was | Now | Why |
|---|---|---|---|
| §1, §3 backbone | ResNet-12, ImageNet-pretrained | **torchvision ResNet-18, ImageNet-pretrained** (512-d). ResNet-12 stays in the report as cited lineage only; no DropBlock | no pretrained ResNet-12 (64/160/320/640) exists |
| §1, §4.3 compute | 1× H100 | **Remote NVIDIA VM** for training. This machine (RX 5600 6 GB, no CUDA) is CPU dev/smoke only | local ROCm gfx1010 is unsupported |
| §2.2 data | Kaggle first, HF fallback | **HF `resyhgerwshshgdfghsdfgh/SD-198` @ `68c02939…`** (ungated, one zip, sha256 `71bf8d78…` verified against HF LFS). Kaggle slugs remain unverified | verified 2026-10-03 |
| §2.3 split | ~100 / ~28 / ~70 | **101 train / 27 val / 70 test** classes (4,428 / 1,191 / 965 images), seed 42. No novel class has < 10 images (min 10, median 14) | actual counts from the real copy |
| §4.2 optimizer | AdamW 1e-3 for all params | **backbone 1e-5 (planned 1e-4, see §4.1 row), learnable temperature 1e-3**; L2-normalised embeddings × learnable scale (init 10); warm-up min(500, steps/10) = 100 steps | 1e-3 would wreck pretrained weights; 1e-4 overfit |
| §4.1 schedule | 20,000 episodes, lr 1e-4 | **4,000 episodes (1,000 steps), backbone lr 1e-5, val every 50 steps** | the planned schedule overfit: train acc 99.4% while val on unseen classes fell 78.2% → 73.8% (frozen ImageNet: 70.9%). A 5-config sweep judged on val only gave 78.5–79.2% for every gentle setting; simplest kept. Evidence logs: `runs/_evidence/`, `runs/sweep/` |
| §4.4 seeds | resume-from-checkpoint shortcut | **3 full independent runs (42/43/44)** | ResNet-18 + 256 px thumbnails makes this cheap |
| §5.1 test queries | 15 per class | **5 per class, no replacement** (test classes need ≥ k+5 images) | novel classes have 10–19 images; replacement would leak support into query |
| §5.1 baseline | none | **Frozen ImageNet ResNet-18 + prototypes** row on identical test episodes. CPU smoke (20 episodes, wide CI ±4–5): 5-way 1-shot ≈ 64%, 5-shot ≈ 81% | the §5.1 "expected band" is reached with no training, so "it learned something" means beating this row |
| §8 data hygiene | – | The mirror zip contains a stray 40 MB Windows `.exe` (`Steroid_Use_abusemisuse_Dermatitis/2015qiangpzb.exe`). Never executed; the pipeline only reads image extensions | flagged for the report; don't redistribute the raw zip |

## 1. Decisions (locked from your answers)

| Decision | Choice | Why |
|---|---|---|
| Dataset | **SD-198** (198 classes, ~6,584 clinical images) | Only public skin dataset with enough classes for disjoint train/val/test few-shot splits. HAM10000/ISIC (7–8 classes) cannot do class-disjoint eval. |
| Backbone init | **ImageNet-pretrained, fine-tuned** (allowed) | +5–10% accuracy for free, standard in SCAN / Meta-Transfer papers. |
| GPU | **1× H100 80GB preferred**, efficient config. A100/H200/B200 all fine, no code change | Few-shot tensors are small; 2 GPUs add launch/coordination overhead for ~zero gain at this scale. |
| Resolution | **224×224** only. No 336/384 sweep | SD-198 natives are small variable-size clinical photos (tens–hundreds of KB). 224 matches pretraining + all published SD-198 work. Higher res = 3–4× cost, ~0 gain. Verified against SCAN + Meta-Transfer (both 224). |
| Deliverables | **notebook + short report + simple demo** | No paper-grade ablations. One main run + 3-seed variance + t-SNE is enough to prove it works. |

## 2. Dataset

### 2.1 What it is
* **SD-198**: 198 fine-grained skin-disease classes, 6,584 clinical (phone-style) photos, long-tailed. ~70 classes have <20 images → natural few-shot / rare-disease story.
* Modality: clinical photography (varied lighting, background, framing). NOT standardized dermoscopy. Augmentation matters more than resolution.

### 2.2 Source (you don't have it yet — get it in this order)
1. **Kaggle mirror (easiest)**: search `SD-198` — `longngzzz/sd-198` and `tiulongchau/sd-198` are the commonly used copies. Download zip, unzip to `data/SD-198/`.
2. **Fallback — HuggingFace**: `resyhgerwshshgdfghsdfgh/SD-198` (mirror, sometimes gated — request access if 401).
3. **Original (flaky)**: `http://xiaopingwu.cn/assets/projects/sd-198/` — referenced by `xmed-lab/SCAN` repo README. Try only if mirrors fail.

Pin what you used in the report: e.g. "Kaggle `longngzzz/sd-198`, downloaded <date>, md5 of zip: ___". Any mirror is fine for a class project as long as you record it.

### 2.3 Split (class-disjoint — the whole point)
Follow Meta-Transfer / SCAN convention, simplified for a class project:

* **Novel/test**: classes with **<20 images** → ~70 classes. These are never seen in training.
* **Base**: remaining ~128 classes → random split into **~100 train / ~28 val** (class-disjoint from test AND from each other).
* Implementation: one `splits.json` mapping `class_name → {train|val|test}` generated once with fixed seed (e.g. seed 42), committed to repo. Stratify so val covers a mix of common/medium-frequency classes.
* SCAN's tighter subset (20 base × 60 imgs + 70 novel × 10–19 imgs = ~2,200 imgs) is a valid **debug/fast option** if your download is partial or you want a 15-min smoke run. Note which split you used.

### 2.4 Preprocessing (keep it standard)
* Train: `Resize(256) → RandomResizedCrop(224, scale 0.8–1.0) → RandomHorizontalFlip(0.5) → RandomRotation(15°) → ColorJitter(0.2,0.2,0.2,0.1) → ToTensor → Normalize(ImageNet mean/std)`.
* Val/test: `Resize(256) → CenterCrop(224) → Normalize`.
* No elastic deformation, no MixUp/CutMix inside episodes. Those help *transfer-pretraining* (Meta-Transfer DTL branch), not episodic ProtoNet support sets, and they complicate the notebook story. Skip.
* Dataloader: `num_workers=8`, `pin_memory=True`, `persistent_workers=True`. Pre-resize to 256px once offline if I/O bound (optional `scripts/prep_thumbs.py`).

## 3. Architecture

**ResNet-12 + Prototypical head.** Citable (TADAM / MetaOptNet / CloserLook lineage), bigger than toy Conv-4, small enough for 1 GPU.

```
Input 224×224×3
→ ResNet-12: 4 residual blocks, channels [64, 160, 320, 640]
|  each block: 3×3 conv ×3 + BN + LeakyReLU(0.1) + skip, DropBlock at last 2 blocks
|  downsample ×2 per block (maxpool or stride-2)
→ feature map 14×14×640
→ GlobalAvgPool → L2-normalize (optional, helps Euclidean stability)
→ 640-d embedding
→ Prototypical head: class prototype = mean(support embeddings); logits = -euclidean_dist(query, prototypes); softmax + CE loss
```

* Params: ~8–12M depending on DropBlock/width variant. Fits anywhere.
* No classifier FC during meta-training. For the demo you may fit a temporary linear/nearest-centroid on a support set at inference time — same embeddings, no retraining.
* Why not WRN-28-10 / DenseNet-121: they ARE the SOTA backbones on SD-198 (SCAN uses WRN-28-10), but 2–3× slower and overkill for "prove it works." Mention as future work in report. If training finishes early and you want one flex line, swap to ResNet-18 (drop-in, still cheap) — optional, not planned.

## 4. Training (efficient config — the default)

### 4.1 Episodic setup
* **5-way, 5-shot, 15 query/class** → 25 support + 75 query = 100 images/episode. This is the standard reportable setting.
* **Train: 20,000 episodes total, 4 episodes batched per optimizer step → 5,000 steps.** (Not 50k. 20k is enough to converge from ImageNet init on SD-198 and keeps you ~60–75 min on 1× H100.)
* **Val: 600 random episodes** every 1,000 episodes (i.e. every 250 steps); keep best checkpoint on val 5-way 5-shot accuracy.
* **Test (once, at end)**: 600 episodes each on 5-way {1-shot, 5-shot} + 2-way {1-shot, 5-shot} for a slightly wider table. Report mean ± 95% CI over episodes AND mean ± std over 3 seeds (see §4.4).

### 4.2 Optimization
* Optimizer: **AdamW, lr 1e-3, weight_decay 1e-4** (simpler than SGD schedules, stable from pretrained init; SGD+momentum 0.1/cosine is the literature alt — either accepted, pick AdamW for notebook simplicity).
* Scheduler: cosine decay to 1e-5 over the 5,000 steps, 500-step warmup.
* Precision: **bf16 mixed precision** (`torch.autocast('cuda', dtype=torch.bf16)` + GradScaler or AMP). Native on H100/H200/B200; on A100 bf16 is also supported (Ampere+). Falls back to fp16 if needed — one flag.
* Gradient clip: 1.0 norm. DropBlock keep_prob 0.9 (blocks 3–4 only).
* Seed everything: `seed_everything(42)` + deterministic dataloader seed for the main run.

### 4.3 Time budget (1× GPU — estimates, not guarantees)
Measure on your box; dataloader workers and storage change these by ±30%.
| Phase | Time on 1× H100 | Notes |
|---|---|---|
| Env + data unzip + `splits.json` | ~10 min | one-time |
| Smoke run (500 episodes, 224px) | ~5 min | sanity check before full run |
| Full train (20k eps, batch 4, bf16) | **~60–75 min** | |
| Val sweeps (during train) | included | 600 eps × ~6 checkpoints ≈ 10 min inside the above |
| Final test (4 configs × 600 eps) + t-SNE | ~10 min | |
| **Total** | **~90–110 min** | comfortably under your 2–3h max |

* On A100: ×1.3–1.5 time (~2h, still fits). On H200/B200: same or faster than H100. No config change needed.
* Do NOT use 2 GPUs. DDP for 400-img steps adds more sync overhead than it saves. If you ever scale, scale episodes-per-batch on 1 GPU first (4→8) before touching a second card.

### 4.4 Seeds / variance (cheap credibility)
* Main run seed 42. Then **2 repeat runs (seeds 43, 44) at same config** — each ~60 min if rerun fully, OR ~15 min if you resume from epoch-80% checkpoint with different episode sampler seed (acceptable for class report; disclose it). Report `mean ± std across 3 seeds`.
* If even that feels long: run seeds 43/44 for 5k episodes each from the best checkpoint with fresh episode sampling — enough for a variance line, ~10 min each. Disclose as "fine-tune seeds."

## 5. Evaluation & demo

### 5.1 Metrics (report table)
* Accuracy (%) + F1 (macro) on novel classes, mean ± 95% CI over 600 episodes per config.
* Configs: 5-way 1-shot, 5-way 5-shot (headline), 2-way 1-shot, 2-way 5-shot (bonus row, cheap since same embeddings).
* Expected range (ballpark from SCAN / Meta-Transfer with stronger backbones — your ResNet-12 may land a few points lower, that's fine): 5-way 1-shot ~55–65%, 5-way 5-shot ~75–85%. If you're within this band, it works. Below 50% on 5-shot = bug (usually leak, wrong norm, or prototype computed on query set).

### 5.2 Required figures (3, all notebook-generated)
1. Train loss + val accuracy curve.
2. t-SNE (or UMAP) of ~10 novel classes' query embeddings, colored by true label — the "it learned something" plot.
3. 3–5 qualitative episodes: support thumbnails + query + predicted/true + distance margin.

### 5.3 Demo (keep simple — notebook IS the demo)
* Notebook section 6: pick 1 novel class episode, show 5 support images, upload/hold-out 1 query image, display distances to the 5 prototypes + predicted class. That satisfies "demo" with zero deployment.
* Optional (only if time): 30-line Gradio `demo.py` wrapping the same checkpoint (`gr.Image` → top-3 prototypes). Don't build Streamlit + hosting; not needed for grading.

## 6. Repo / notebook layout

```
project/
  notebook.ipynb        # THE deliverable: data → splits → train → eval → t-SNE → demo (runnable top-to-bottom)
  src/
    model.py            # ResNet12 + ProtoHead (or timm resnet12 if allowed)
    data.py             # SD198Dataset + NWayEpisodeSampler + splits.json loader
    train.py            # training loop (called from notebook, also runnable standalone)
    eval.py             # 600-episode eval + CI computation
    viz.py              # curves, t-SNE, episode grids
  splits.json           # generated once, committed
  checkpoints/best.pt   # best val checkpoint (embed in notebook release or Drive link)
  report.md / report.pdf # short report (see §7)
  requirements.txt      # torch>=2.2, torchvision, numpy, sklearn, matplotlib, tqdm, jupyter
```

Notebook sections: 0 Setup → 1 Data + splits → 2 Model → 3 Train (~1 cell, long run) → 4 Eval table → 5 Figures → 6 Demo episode → 7 Conclusion + future work.

## 7. Report outline (1–2 pages is enough)
1. Problem + why few-shot (1 para) — rare diseases, disjoint classes.
2. Dataset + split table (base/novel counts, imgs/class histogram).
3. Method: ResNet-12 + ProtoNet diagram + 5-way 5-shot definition (half page).
4. Results table (4 configs) + training curve + t-SNE.
5. What didn't work / limits: clinical-photo variability, low native res → why 224, small-class noise.
6. Future work (1 line each): WRN-28-10, Derm7pt cross-modal, transfer-pretraining (DTL) comparison.

## 8. Risks & mitigations
* **Mirror download differs slightly from paper's 6,584 count** → record exact counts + hash; splits generated from YOUR copy, so results are self-consistent regardless.
* **Some classes have <10 images** → episode sampler must sample with replacement for support/query when class is tiny; assert `n_support + n_query ≤ N_class` else shrink query to 5 for that episode and note it.
* **Pretrained weight mismatch (224 → fine)** → none, we stay at 224. Load with `strict=False` if DropBlock keys differ.
* **Overfitting to base classes** → early-stop on val novel-proxy (val classes are also unseen), DropBlock + jitter handle the rest.

## 9. What we are explicitly NOT doing
* No 384/448 resolution runs. No Derm7pt cross-domain. No CutMix/MixUp-in-episode. No 2-GPU DDP. No WRN/DenseNet unless ahead of schedule. Each would add 30–60 min for no grading benefit.

## 10. References to cite
* Snell et al. Prototypical Networks (NeurIPS 2017) — method.
* Oreshkin et al. TADAM / Lee et al. MetaOptNet — ResNet-12 justification.
* Sun et al. SD-198 (ECCV 2016) — dataset.
* Li et al. SCAN / xmed-lab/SCAN repo — SD-198-20 split + WRN baseline.
* Özdemir et al. Meta-Transfer Derm-Diagnosis (2024) — 128-base/70-novel split, 224px, augmentation findings.
* Kawahara et al. Derm7pt (JBHI 2019) — listed as related/future only.

## 11. Build checklist
- [x] `pip install -r requirements.txt` on the GPU box (RTX 3090, Vast.ai)
- [x] Download mirror → `data/SD-198-256/` (HF, sha256-checked) → `python src/data.py --make-splits --seed 42` → `splits.json`; `--verify` hash matched on both machines
- [x] Smoke: 500 episodes on GPU; overfit 1 episode to loss 0.0012 (also a pytest)
- [x] Full: ~~20k episodes~~ 4k episodes batch-4 bf16 (see §0) × seeds 42/43/44 → `checkpoints/best.pt` (seed 42)
- [x] Eval 4 configs × 600 eps → `results/results_table.md`, plus frozen-ImageNet baseline row
- [x] 3 figures → embed in notebook + report (+ split histogram)
- [x] Demo episode cell runs top-to-bottom on fresh kernel (`jupyter nbconvert --execute`)
- [x] Report: fill §7, record mirror name/date/hash + seed + GPU type + wall-clock time (`report.md`, `report.pdf`)
- [x] Independent re-check of every reported number + leakage checks: `python scripts/verify_claims.py`; same-patient re-shoots: `python scripts/dedup_eval.py` (report §6)
- [x] Follow-ups (report §7): re-run on a second GPU (<0.2 pp), two more train/val splits (`splits_s7.json`, `splits_s123.json`; within 0.3 pp in 5-way), ResNet-50 (+0.1 to +1.3 pp, significant only in 1-shot) — `scripts/run_extra.sh`, `scripts/summarize_extra.py`
- [x] Project website `site/` (Home + 3-D Demo with Story / Explore / Few-shot lab, live-model toggle). Build: `scripts/embed_all.py` → `scripts/build_showcase.py` → `scripts/build_site.py`; check: `node scripts/check_showcase.js`; serve: `python scripts/serve_site.py` → http://127.0.0.1:8765
- [ ] (Out of scope, by decision 2026-10-05) Slide deck `~/Downloads/Few-Shot_Rare_Skin_Diagnostics.pptx` still shows pre-training projections; not updated
