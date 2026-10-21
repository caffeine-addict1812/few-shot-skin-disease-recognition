# Few-shot skin-lesion classification on SD-198

Can a model recognise a skin disease it has **never seen in training** from only 1 or 5 example photos?
This project fine-tunes an ImageNet-pretrained **ResNet-18** as a **Prototypical Network** on
[SD-198](https://link.springer.com/chapter/10.1007/978-3-319-46466-4_13) (198 skin diseases, 6,584 clinical photos)
and tests it on the 70 rarest diseases, which are held out completely (class-disjoint evaluation).

> **Research demo and class project. Not a diagnostic tool.**

## Results (70 unseen test diseases, 600 fixed test episodes, mean of 3 seeds)

| setting | fine-tuned | frozen ImageNet | gain |
|---|---|---|---|
| 5-way 1-shot | **70.7%** | 62.6% | +8.1 pp |
| 5-way 5-shot | **85.9%** | 80.2% | +5.6 pp |
| 2-way 1-shot | **86.4%** | 81.4% | +5.0 pp |
| 2-way 5-shot | **94.5%** | 91.7% | +2.8 pp |

Chance is 20% (5-way) and 50% (2-way). The 95% CI over episodes is about ±1 pp. The result is stable across seeds
and across two other train/val class splits (within 0.3 pp in the 5-way settings). Near-duplicate re-shoots of the
same patient in SD-198 cost about 1.2 pp (5-way 1-shot) and 0.8 pp (5-way 5-shot); the gain over frozen ImageNet is
unchanged. Full details, the failed first attempt (overfitting) and every check are in [`report.md`](report.md).

## Method in one paragraph

A torchvision ResNet-18 (ImageNet weights) maps a 224x224 crop to a 512-d, L2-normalised embedding. A class prototype
is the mean embedding of its support photos, and a query is scored by `-scale * ||f(query) - prototype||^2` with a
learned temperature. There is no classifier head, so a new set of diseases needs no retraining. Training is episodic
(5-way 5-shot, 15 queries per class), 4,000 episodes (1,000 steps), AdamW with backbone lr 1e-5, bf16, about 7 minutes
per run on an RTX 3090. The checkpoint is chosen on 27 validation diseases that are also unseen in training.

## Run the website (no GPU needed)

The `site/` folder is a static website with a 3-D demo (Story, Explore, Few-shot lab). It needs only Python:

```bash
python -m http.server 8765 --directory site      # Windows
python3 -m http.server 8765 --directory site     # Mac / Linux
```

Open <http://localhost:8765>. See [`SETUP.md`](SETUP.md) for a click-by-click version.

For the **Live model** (upload your own photo) and hover previews of training photos, run the Python server instead:

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install torch torchvision numpy pillow
python scripts/serve_site.py                      # http://127.0.0.1:8765
```

This needs `checkpoints/best.pt`, `results/site_cache.npz` and `data/SD-198-256/` (see "Files that are not in this
repo" below). The server uses only the Python standard library for HTTP and binds to `127.0.0.1`; an uploaded photo
stays on your computer.

## Repository layout

```
src/            model.py (ResNet-18 + ProtoNet), data.py (splits, episode sampler), train.py, eval.py, viz.py, utils.py
scripts/        download_data.py, prep_thumbs.py, run_all.sh, run_seed.sh, sweep.sh, run_extra.sh,
                verify_claims.py, dedup_eval.py, transductive_eval.py, summarize_*.py, frozen_val.py,
                embed_all.py, build_showcase.py, build_site.py, check_showcase.js, serve_site.py,
                build_report.py, build_pdf.sh, build_notebook.py
site/           static website: index.html, demo.html, assets/ (JS, CSS)
tests/          pytest tests
runs/, results/ training logs and result JSON files (small); results/*.npz and results/figures/ are rebuilt
splits.json     class -> train/val/test, seed 42 (splits_s7.json, splits_s123.json: robustness splits)
data_manifest.json   pinned dataset revision and hashes
report.md       the report (every number is generated from result files)
project.md      the original plan and the revisions made after planning
requirements.txt
```

## Reproduce the experiments

Training needs an NVIDIA GPU. The scripts are bash; on Windows use WSL or Git Bash.

```bash
pip install -r requirements.txt

# 1. data: pinned HuggingFace revision, sha256-checked, then 256 px thumbnails
python scripts/download_data.py
python scripts/prep_thumbs.py --src data/SD-198-raw --dst data/SD-198-256
python src/data.py --verify                       # must match data_manifest.json and splits.json

# 2. train and test (seeds 42, 43, 44) plus the frozen ImageNet baseline
bash scripts/run_all.sh 12                        # or one seed: bash scripts/run_seed.sh 42

# 3. independent checks and the report
python scripts/verify_claims.py                   # CPU, about 25 min
python scripts/dedup_eval.py
python scripts/build_report.py

# 4. follow-ups: other class splits, ResNet-50
bash scripts/run_extra.sh all && python scripts/summarize_extra.py

# 5. website data, then serve
python scripts/embed_all.py                       # about 20 min on CPU
python scripts/build_showcase.py
python scripts/build_site.py
node scripts/check_showcase.js                    # the page's own maths must match the report within 0.5 pp
python scripts/serve_site.py

# tests
pytest
```

## Files that are not in this repo (see `.gitignore`)

| file or folder | why | how to get it |
|---|---|---|
| `data/` | dataset, license listed as "unknown" | `python scripts/download_data.py`, then `prep_thumbs.py` |
| `*.pt` (checkpoints, run weights) | size (about 45 MB each) | retrain with `scripts/run_seed.sh`, or ask the authors for the backup |
| `results/*.npz`, `results/figures/` | large, generated | `embed_all.py`, `build_showcase.py`, the notebook / `src/viz.py` |
| `site/assets/data.js`, `thumbs.jpg` | generated, contain dataset photos | `python scripts/build_showcase.py` |
| `site/assets/vendor/` | third-party three.js r149 (MIT) | download `https://cdn.jsdelivr.net/npm/three@0.149.0/build/three.min.js` to `site/assets/vendor/three.min.js` |
| `report.pdf`, `notebook.ipynb`, and their copies in `site/assets/` | generated | `scripts/build_pdf.sh`, `scripts/build_notebook.py`, `scripts/build_site.py` |

## Data note

The dataset is a HuggingFace mirror (`resyhgerwshshgdfghsdfgh/SD-198`, revision `68c02939c06d6850f40f2911e1d1d49bd760139d`,
zip sha256 `71bf8d78...`). The zip contains an unrelated 40 MB Windows `.exe` in one class folder
(`Steroid_Use_abusemisuse_Dermatitis/2015qiangpzb.exe`). It is never executed, the pipeline only reads image files,
and the raw zip should not be redistributed.

## Limits

* ResNet-18 and a plain ProtoNet: the numbers are **not comparable** with published SD-198 results that use larger
  backbones and more elaborate methods.
* SD-198 contains re-shoots of the same patient, and the standard protocol can put two of them in support and query.
* The recipe was chosen after seeing validation results (never test results).
* The standard deviation over seeds understates total uncertainty: seeds share the class split and the test episodes.

## References

Snell et al., *Prototypical Networks* (NeurIPS 2017) · Sun et al., *SD-198* (ECCV 2016) · Oreshkin et al., *TADAM* ·
Lee et al., *MetaOptNet* · Li et al., *SCAN* (2022) · Özdemir et al., *Meta-Transfer Derm-Diagnosis* (2024) ·
Kawahara et al., *Derm7pt* (JBHI 2019) · Ren et al., *Semi-supervised few-shot learning* (2018).
