"""Build notebook.ipynb (then execute it with `jupyter nbconvert --to notebook --execute --inplace notebook.ipynb`).

The notebook is the deliverable: it runs top-to-bottom on a fresh kernel by LOADING the finished runs
(runs/, results/, checkpoints/best.pt). Set RUN_TRAIN = True to retrain from inside the notebook (GPU, ~1.5 h).
"""
import nbformat as nbf

cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip("\n")))

md("""
# Few-shot skin-lesion classification on SD-198 (ResNet-18 + Prototypical Network)

**Question.** Can a model recognise a *skin disease it has never been trained on* from 1 or 5 example photos?
That is the rare-disease situation: some diseases have a handful of labelled images.

**Setup.** SD-198 (198 diseases, 6,584 clinical photos). Classes are split into *disjoint* train / val / test sets;
the test classes are the 70 rarest ones (10-19 images each) and the model never sees them in training.
An ImageNet-pretrained ResNet-18 is fine-tuned episodically (5-way 5-shot) and scored on unseen classes.

**Sections:** 0 Setup · 1 Data and splits · 2 Model · 3 Training runs · 4 Results table · 5 Figures · 6 Demo · 7 Conclusion
""")

md("## 0. Setup")
code("""
import json, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from IPython.display import Image as IPImage, Markdown, display
from PIL import Image

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import viz
from data import EpisodeDataset, load_eval_split, load_splits
from eval import MIN_TEST_IMAGES, N_QUERY_TEST, TEST_SEED, aggregate, evaluate_test
from model import ProtoNet, load_model
from utils import get_device, load_json, seed_everything

# False: load the finished runs (fast, runs anywhere).  True: retrain 3 seeds here (needs an NVIDIA GPU, ~1.5 h).
RUN_TRAIN = False

DATA_ROOT, SPLITS = "data/SD-198-256", "splits.json"
SEEDS = [42, 43, 44]
MAIN_SEED = 42
CKPT = ROOT / "checkpoints" / "best.pt"          # best-val checkpoint of the main run (seed 42)
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
device = get_device()
seed_everything(0)
print("device:", device, "| torch", torch.__version__)
""")

md("""
## 1. Data and class-disjoint splits

* **Source:** HuggingFace mirror of SD-198 (Sun et al., ECCV 2016), pinned to one revision and verified by sha256.
* **Novel / test classes:** every class with **< 20 images** (70 classes). Never used for training or model selection.
* **Base classes:** the other 128, split into 101 train / 27 val (val stratified over class-size terciles), fixed seed 42.
* Images are pre-resized to 256 px (shorter side); the model sees 224x224 crops.
""")
code("""
splits = load_splits(SPLITS)
manifest = load_json("data_manifest.json")
dl = manifest["download"]
display(pd.Series({
    "repo": dl["repo"], "revision": dl["revision"][:12] + "...", "zip sha256": dl["zip_sha256"][:16] + "...",
    "zip md5": dl["zip_md5"], "downloaded (UTC)": dl["downloaded_utc"], "license": dl["license"],
    "classes / images": f"{manifest['n_classes']} / {manifest['n_images']}",
}, name="provenance").to_frame())

rows = []
for s in ("train", "val", "test"):
    n = [splits["counts"][c] for c, v in splits["splits"].items() if v == s]
    rows.append(dict(split=s, classes=len(n), images=sum(n), min=min(n), median=int(np.median(n)), max=max(n)))
display(pd.DataFrame(rows).set_index("split"))
viz.plot_split_histogram(splits["counts"], splits["splits"], FIG / "fig0_split_histogram.png")
display(IPImage(str(FIG / "fig0_split_histogram.png")))
""")

md("""
## 2. Model

`ResNet-18` (ImageNet weights) -> global average pool -> 512-d embedding -> L2-normalise.
A class **prototype** is the mean embedding of its support images; a query is scored by
`logit_c = -scale * ||embedding(query) - prototype_c||^2` (softmax + cross-entropy in training, `argmax` at test time).
`scale` is a learned temperature. There is **no classifier head**, so the same network handles any set of new classes.
""")
code("""
model = ProtoNet(pretrained=False)
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params / 1e6:.1f} M   embedding dim: {model.encoder.out_dim}   init scale: {model.scale.item():.1f}")
""")

md("""
## 3. Training runs

Episodic training: 4,000 episodes (5-way 5-shot, 15 queries per class), 4 episodes per optimizer step (1,000 steps),
AdamW (backbone lr 1e-5, temperature lr 1e-3), 100-step warm-up + cosine decay, bf16.
Model selection = best accuracy on 600 *fixed* validation episodes drawn from the 27 validation classes, checked every 50 steps.
Three independent runs (seeds 42, 43, 44).

**Why so short?** The first attempt followed the original plan (20,000 episodes, lr 1e-4). It memorised the 101 training
classes (train accuracy 99.4%) while accuracy on unseen validation classes *fell* from 78.2% to 73.8%. A short sweep, judged on
validation classes only, showed that every gentle setting reaches the same ~79% plateau; we kept the simplest one.
""")
code("""
if RUN_TRAIN:
    subprocess.run(["bash", "scripts/run_all.sh"], check=True)

runs = []
for s in SEEDS:
    d = ROOT / "runs" / f"seed{s}"
    if (d / "done.json").exists():
        cfg, done = load_json(d / "config.json"), load_json(d / "done.json")
        runs.append(dict(seed=s, gpu=done["device"], steps=done["steps"], best_val_acc=round(done["best_val"], 2),
                         wall_clock_min=round(done["wall_clock_s"] / 60, 1), torch=cfg["torch"]))
SEEDS = [r["seed"] for r in runs]
display(pd.DataFrame(runs).set_index("seed"))
""")

md("""
## 4. Results on unseen (novel) classes

Mean accuracy over the **same 600 fixed test episodes** for every model. Test queries: 5 per class, drawn without
replacement (novel classes have only 10-19 images). `+-` after the mean is the standard deviation over seeds;
the 95% CI is over the 600 episodes. **Frozen ImageNet** = the same ResNet-18 with prototypes and *no* fine-tuning:
the bar a trained model has to beat.
""")
code("""
files = [ROOT / "results" / f"seed{s}_test.json" for s in SEEDS]
summary = aggregate(files, ROOT / "results" / "frozen_test.json")
frozen = summary["frozen_baseline"]
rows = []
for key, v in summary["configs"].items():
    f = frozen[key]
    rows.append({
        "config": key.replace("way_", "-way ").replace("shot", "-shot"),
        "fine-tuned acc (%)": f"{v['acc_mean']:.1f} +- {v['acc_std_over_seeds']:.1f}",
        "95% CI (episodes)": f"+-{v['acc_ci95_episodes']:.1f}",
        "fine-tuned macro-F1": f"{v['f1_mean']:.1f}",
        "frozen ImageNet acc (%)": f"{f['acc']:.1f} +-{f['acc_ci95']:.1f}",
        "gain (pp)": f"{v['acc_mean'] - f['acc']:+.1f}",
    })
table = pd.DataFrame(rows).set_index("config")
display(table)
print(f"{summary['n_seeds']} seeds {summary['seeds']}; test classes: {summary['n_test_classes']} "
      f"(dropped for < {MIN_TEST_IMAGES} images: {len(summary['dropped_test_classes'])}); "
      f"per-seed 5-way 5-shot acc: {[round(a, 1) for a in summary['configs']['5way_5shot']['per_seed_acc']]}")
(ROOT / "results" / "results_table.md").write_text(table.to_string())
""")

md("""
## 5. Figures
""")
code("""
# Figure 1: training loss and validation accuracy, one line per seed
run_dirs = [ROOT / "runs" / f"seed{s}" for s in SEEDS]
viz.plot_training_curves(run_dirs, FIG / "fig1_training_curves.png")
display(IPImage(str(FIG / "fig1_training_curves.png")))
""")
code("""
# Figure 2: t-SNE of query-able images from unseen classes (main run). 8 best-populated test classes.
model, ckpt = load_model(CKPT, device)
test_ds = load_eval_split(DATA_ROOT, splits, "test", min_images=MIN_TEST_IMAGES)
ids = viz.pick_classes(test_ds, 8)
emb, lab = viz.extract_embeddings(model, test_ds, ids, device)
viz.plot_tsne(emb, lab, test_ds.classes, FIG / "fig2_tsne.png",
              title="t-SNE of embeddings of 8 unseen disease classes (fine-tuned model)")
display(IPImage(str(FIG / "fig2_tsne.png")))
""")
code("""
# Figure 3: three qualitative 5-way 5-shot episodes = the FIRST three of the fixed test episode list (not cherry-picked)
eds = EpisodeDataset(test_ds, 5, 5, N_QUERY_TEST, 600, seed=TEST_SEED)
for i in range(3):
    path = FIG / f"fig3_episode{i}.png"
    viz.plot_episode(model, test_ds, eds.episode(i), device, path)
    display(IPImage(str(path)))
""")

md("""
## 6. Demo: classify a photo of a disease the model has never seen

Pick a 5-way episode from the test classes: the model gets **5 example photos per disease** and one held-out query photo,
and answers by nearest prototype. No weights are updated.
""")
code("""
DEMO_SEED, QUERY_CLASS = 7, 2                # change to explore: which episode, which of the 5 diseases is the query
ep = EpisodeDataset(test_ds, 5, 5, 1, 1, seed=DEMO_SEED).episode(0)
classes, support, query = ep
d2 = viz.episode_distances(model, test_ds, support, query, device)[QUERY_CLASS]
query_img = test_ds.load_raw(int(query[QUERY_CLASS, 0]))

# Optional: classify YOUR OWN photo against the same 5 prototypes (any skin photo; JPG/PNG path)
QUERY_IMAGE = None                            # e.g. "my_photo.jpg"
true_idx = QUERY_CLASS
if QUERY_IMAGE:
    from data import build_transforms
    with torch.no_grad():
        sup = torch.stack([test_ds[int(j)][0] for j in support.ravel()]).to(device)
        protos = model.embed(sup).view(5, 5, -1).mean(1)
        q = model.embed(build_transforms(False)(Image.open(QUERY_IMAGE).convert("RGB")).unsqueeze(0).to(device))
        d2 = ((q - protos) ** 2).sum(-1)[0].cpu().numpy()
    query_img, true_idx = Image.open(QUERY_IMAGE).convert("RGB"), None

viz.plot_demo(test_ds, ep, query_img, d2, true_idx, FIG / "fig4_demo.png")
display(IPImage(str(FIG / "fig4_demo.png")))
names = [test_ds.classes[c].replace("_", " ") for c in classes]
print(f"prediction: {names[int(np.argmin(d2))]}" + (f"   |   truth: {names[true_idx]}" if true_idx is not None else ""))
""")

md("## 7. Conclusion")
code("""
cfg = summary["configs"]
g5, g1 = (cfg["5way_5shot"]["acc_mean"] - frozen["5way_5shot"]["acc"],
          cfg["5way_1shot"]["acc_mean"] - frozen["5way_1shot"]["acc"])
ci5, ci1 = cfg["5way_5shot"]["acc_ci95_episodes"], cfg["5way_1shot"]["acc_ci95_episodes"]
verdict = lambda g, ci: ("a clear gain" if g > 2 * ci else "a small gain, comparable to the noise" if g > 0
                          else "no gain")
dd_path = Path("results/dedup_test.json")
if dd_path.exists():
    dd = json.loads(dd_path.read_text())
    f1 = lambda k, c: dd[k]["results"][c]["finetuned_mean"]
    dup_note = (f"SD-198 holds many re-shoots of the same patient, and the standard protocol can put two of them in support and query; "
                f"removing near-duplicates (`scripts/dedup_eval.py`) costs about "
                f"{f1('cos>0.9_classes_all_images', '5way_1shot') - f1('cos>0.9', '5way_1shot'):.1f} pp on 5-way 1-shot for the same classes "
                f"(at most {f1('full', '5way_1shot') - f1('cos>0.9', '5way_1shot'):.1f} pp if the classes that fall below 10 images without them are dropped too), "
                f"with the same gain over frozen ImageNet")
else:
    dup_note = "SD-198 may contain near-duplicate photos, which this notebook does not check"
display(Markdown(f'''
**What worked.** On 70 disease classes never seen in training, the model reaches
**{cfg["5way_5shot"]["acc_mean"]:.1f}%** (5-way 5-shot) and **{cfg["5way_1shot"]["acc_mean"]:.1f}%** (5-way 1-shot)
vs. 20% / 20% for chance, averaged over {summary["n_seeds"]} seeds and 600 fixed episodes each.

**Does fine-tuning matter?** The frozen ImageNet ResNet-18 with the same prototype rule already scores
{frozen["5way_5shot"]["acc"]:.1f}% / {frozen["5way_1shot"]["acc"]:.1f}%. Episodic fine-tuning adds
**{g5:+.1f} pp** (5-shot: {verdict(g5, ci5)}) and **{g1:+.1f} pp** (1-shot: {verdict(g1, ci1)}),
with an episode-level 95% CI of about +-{ci5:.1f} / +-{ci1:.1f} pp.

**Limits.** Clinical photos vary in lighting, framing and background; novel classes have only 10-19 images, so test
episodes use 5 queries per class and the confidence intervals are wider than for larger query sets;
{dup_note}; class-disjoint splits depend on one
seed-42 partition of the base classes.

**Future work.** WRN-28-10 / DenseNet backbones (the SD-198 SOTA), cross-dataset transfer to Derm7pt,
transfer-pretraining (DTL) comparison.
'''))
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
nbf.write(nb, "notebook.ipynb")
print("wrote notebook.ipynb with", len(cells), "cells")
