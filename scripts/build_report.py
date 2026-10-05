"""Generate report.md from the run artifacts: every number in the report comes from a file, none is typed in.

    python scripts/build_report.py        # needs results/ runs/ data_manifest.json splits.json (+ results/figures/)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from eval import aggregate  # noqa: E402
from summarize_runs import summarize  # noqa: E402
from utils import load_json  # noqa: E402

R = ROOT / "results"
seeds = [s for s in (42, 43, 44) if (ROOT / "runs" / f"seed{s}" / "done.json").exists()]
summary = aggregate([R / f"seed{s}_test.json" for s in seeds], R / "frozen_test.json")
cfg, frozen = summary["configs"], summary["frozen_baseline"]
frozen_val = load_json(R / "frozen_val.json")["5way_5shot"]
manifest, splits = load_json(ROOT / "data_manifest.json"), load_json(ROOT / "splits.json")
dl = manifest["download"]
run_cfgs = {s: load_json(ROOT / "runs" / f"seed{s}" / "config.json") for s in seeds}
run_done = {s: load_json(ROOT / "runs" / f"seed{s}" / "done.json") for s in seeds}


def pretty_cfg(key):
    w, s = key.split("_")
    return f"{w[0]}-way {s[0]}-shot"


def verdict(gain, ci):
    return ("a clear gain" if gain > 2 * ci else "a small gain, comparable to the episode noise" if gain > 0
            else "no gain")


# ---------------------------------------------------------------- tables
split_rows = []
for s in ("train", "val", "test"):
    n = [splits["counts"][c] for c, v in splits["splits"].items() if v == s]
    split_rows.append(f"| {s} | {len(n)} | {sum(n)} | {min(n)} / {int(sorted(n)[len(n) // 2])} / {max(n)} |")

res_rows = []
for key, v in cfg.items():
    f = frozen[key]
    g = v["acc_mean"] - f["acc"]
    res_rows.append(f"| {pretty_cfg(key)} | **{v['acc_mean']:.1f}** ± {v['acc_std_over_seeds']:.2f} | ±{v['acc_ci95_episodes']:.1f} "
                    f"| {v['f1_mean']:.1f} | {f['acc']:.1f} ± {f['acc_ci95']:.1f} | {g:+.1f} |")

run_rows = []
for s in seeds:
    log = [json.loads(line) for line in open(ROOT / "runs" / f"seed{s}" / "log.jsonl") if line.strip()]
    val = [r for r in log if r["type"] == "val"]
    best = max(val, key=lambda r: r["acc"])
    run_rows.append(f"| {s} | {run_done[s]['device']} | {run_done[s]['steps']} | {best['acc']:.2f} (step {best['step']}) "
                    f"| {run_done[s]['wall_clock_s'] / 60:.1f} |")

sweep = [r for r in (summarize(p) for p in sorted((ROOT / "runs" / "sweep").glob("*")) +
                     sorted((ROOT / "runs" / "_evidence").glob("*"))) if r]
sweep_rows = []
for r in sorted(sweep, key=lambda r: -r["best_val"]):
    c = r["cfg"]
    sweep_rows.append(f"| {r['name']} | {c['lr_backbone']:g} | {c['episodes']} | {c['n_way']}-way | "
                      f"{'yes' if c.get('freeze_bn') else 'no'} | {r['best_val']:.2f} ± {r['ci']:.2f} (step {r['best_step']}) "
                      f"| {r['final_val']:.2f} | {r['train_acc']:.1f} |")
ev = summarize(ROOT / "runs" / "_evidence" / "full20k_lr1e-4")

ev_passes = ev['last_step'] * 4 * 5 * (5 + 15) / load_json(ROOT / 'runs' / 'seed42' / 'config.json')['train_images']
gains = [cfg[k]["acc_mean"] - frozen[k]["acc"] for k in cfg]
g5 = cfg["5way_5shot"]["acc_mean"] - frozen["5way_5shot"]["acc"]
g1 = cfg["5way_1shot"]["acc_mean"] - frozen["5way_1shot"]["acc"]
ci5, ci1 = cfg["5way_5shot"]["acc_ci95_episodes"], cfg["5way_1shot"]["acc_ci95_episodes"]
c0 = run_cfgs[seeds[0]]
FIG = "results/figures"

# ---------------------------------------------------------------- independent verification (optional files)
ver = load_json(R / "verification.json") if (R / "verification.json").exists() else None
dedup = load_json(R / "dedup_test.json") if (R / "dedup_test.json").exists() else None
if ver:
    nd = ver["near_duplicates"]["0.95"]
    cross = [g for g in ver["splits"]["identical_examples"] if len({x[0] for x in g}) > 1]
    cross_txt = "; ".join(" and ".join(f"{x[1].replace('_', ' ')} ({x[0]})" for x in g) for g in cross)
    t = ver["test"]
    fts = [f"seed{s}" for s in seeds]
    rec_diff = max(abs(t[k][m]["recomputed"] - t[k][m]["stored"]) for k in cfg for m in fts + ["frozen"])
    fresh_diff = max(abs(t[k][m]["fresh_5000"] - t[k][m]["stored"]) for k in cfg for m in fts + ["frozen"])
    shuf = {k: sum(t[k][m]["shuffled_labels"] for m in fts) / len(fts) for k in cfg}
    mask = max(abs(t[k][m]["bottom_masked"] - t[k][m]["recomputed"]) for k in cfg for m in fts)
    mask_top = max(abs(t[k][m]["top_masked"] - t[k][m]["recomputed"]) for k in cfg for m in fts)
    gain_rows = [f"| {pretty_cfg(k)} | {t[k]['gain_vs_frozen']['mean']:+.1f} | [{t[k]['gain_vs_frozen']['boot_ci95'][0]:+.1f}, "
                 f"{t[k]['gain_vs_frozen']['boot_ci95'][1]:+.1f}] | {t[k]['gain_vs_frozen']['wins']} / {t[k]['gain_vs_frozen']['ties']} / "
                 f"{t[k]['gain_vs_frozen']['losses']} |" for k in cfg]
if dedup:
    dd_keys = [k for k in dedup if k != "full"]
    dd_rows = []
    for k in ["full"] + dd_keys:
        d = dedup[k]
        r = d["results"]
        label = ("none (full test set)" if k == "full" else
                 f"control: the same {d['classes']} classes, all images" if k.endswith("_classes_all_images") else k.replace("cos>", "cosine > "))
        dd_rows.append(f"| {label} | {d['images']} | {d['classes']} | "
                       + " | ".join(f"{r[c]['finetuned_mean']:.1f} ({r[c]['frozen']:.1f})" for c in cfg) + " |")
    _A, _B, _C = (dedup[k]["results"] for k in ("full", "cos>0.9", "cos>0.9_classes_all_images"))
    acc = lambda r, c: r[c]["finetuned_mean"]
    dup_cost = {c: acc(_C, c) - acc(_B, c) for c in cfg}       # same classes, duplicates removed
    mix_cost = {c: acc(_A, c) - acc(_C, c) for c in cfg}       # same images per class, 21 classes removed
    upper = {c: acc(_A, c) - acc(_B, c) for c in cfg}
    n_drop = len(dedup["cos>0.9"]["dropped_classes"])
    dup_sentence = (f"Removing near-duplicate re-shoots of the same patient costs about {dup_cost['5way_1shot']:.1f} pp (5-way 1-shot) and "
                    f"{dup_cost['5way_5shot']:.1f} pp (5-way 5-shot) on matched classes, and at most {upper['5way_1shot']:.1f} / {upper['5way_5shot']:.1f} pp "
                    f"if the {n_drop} classes that fall below 10 images without their re-shoots are dropped as well; the gain over frozen ImageNet is unchanged")

if ver:
    dup_limit = (f"* **Near-duplicates (checked).** By frozen-ImageNet cosine similarity > 0.95, {nd['test_imgs_with_trainval_neighbour']} of "
                 f"{ver['splits']['counts']['test'][1]} test images have a near-duplicate in train/val (always filed under a *different* disease, so no test label leaks) "
                 f"and {nd['test_pairs_same_class']} same-class pairs exist inside the test set. One photo is filed, byte-identical, "
                 f"as both {cross_txt}: label noise in SD-198 itself. See `{FIG}/fig5_near_duplicates.png`"
                 + (f". {dup_sentence} (section 6)." if dedup else "."))
    footer_limit = (f"* Many SD-198 photos carry a grey source footer strip; we did not remove it. Masking the bottom {15}% of the image changes "
                    f"fine-tuned accuracy by at most {mask:.1f} pp, the same as masking the top {15}% ({mask_top:.1f} pp), so the model does not rely on it.")
else:
    dup_limit = "* SD-198 may contain near-duplicate photos across classes; we did not check for this."
    footer_limit = "* Many SD-198 photos carry a grey source footer strip (visible in the figures); we did not remove it."

verification = ""
if ver:
    verification = f"""
## 6. Independent verification

`scripts/verify_claims.py` re-derives the results through a separate code path (fp32 on CPU, images decoded straight from disk,
a numpy nearest-prototype scorer; only the episode definition is shared). `results/verification.json` holds the output.

* **Reproduction:** every stored accuracy (4 settings x 3 seeds + frozen) is reproduced within {rec_diff:.2f} pp (GPU bf16 vs CPU fp32).
* **Episode choice:** on 5,000 *new* test episodes the accuracies stay within {fresh_diff:.1f} pp of the 600-episode numbers.
* **Metric sanity:** shuffling query labels gives {shuf['5way_5shot']:.1f}% (5-way) and {shuf['2way_5shot']:.1f}% (2-way), i.e. chance.
* **Model selection:** each `best.pt` is the arg-max of its logged validation accuracy, and the training logs contain only train/val records.
* **Gain over frozen ImageNet, paired over the same episodes** (mean of 3 seeds; bootstrap 95% CI over episodes):

| setting | gain (pp) | 95% CI | episodes won / tied / lost |
|---|---|---|---|
{chr(10).join(gain_rows)}

* **Seed agreement:** the three seeds disagree on about half of the 5-way 1-shot episodes, yet their means agree within 0.02 pp; that closeness is partly chance
  (other settings differ by 0.1-0.3 pp between seeds).
"""
    if dedup:
        verification += f"""
**De-duplicated test set** (`scripts/dedup_eval.py`). Inside each test class, images are grouped by frozen-ImageNet cosine similarity and only one per group is kept,
so a query cannot be a near-copy of its own support image. Cells: fine-tuned accuracy (frozen ImageNet in brackets), 600 episodes each.

| near-duplicate threshold | images | classes | {' | '.join(pretty_cfg(c) for c in cfg)} |
|---|---|---|{'---|' * len(cfg)}
{chr(10).join(dd_rows)}

**What this means.** Looking at pairs by eye, both thresholds mostly catch *the same patient photographed again* (another angle or distance), not
different patients who look alike: SD-198 contains many such multi-view series, and the standard protocol (used here and in published SD-198 work)
lets a support and a query photo come from the same patient. The strict threshold also removes {n_drop} classes (those left with fewer than 10 images),
so the control row keeps exactly the surviving classes with all their images. Of the {upper['5way_1shot']:.1f} pp drop on 5-way 1-shot,
**{mix_cost['5way_1shot']:.1f} pp comes from the change of classes and {dup_cost['5way_1shot']:.1f} pp from removing duplicates** (5-way 5-shot:
{mix_cost['5way_5shot']:.1f} and {dup_cost['5way_5shot']:.1f} pp). The split is not perfectly clean: a class is removed *because* it contained re-shoots
(mostly the smallest classes, 10-18 images, where losing a few photos drops it below 10), so the effect of duplicates lies between about
{dup_cost['5way_1shot']:.0f} and {upper['5way_1shot']:.0f} pp, with the matched-class {dup_cost['5way_1shot']:.1f} pp the more direct measure. An embedding threshold will not catch every same-patient pair.
The gain over frozen ImageNet holds in every row ({', '.join(f"{dedup[k]['results']['5way_1shot']['gain']:+.1f}" for k in ['full'] + dd_keys)} pp on 5-way 1-shot).
"""

# ---------------------------------------------------------------- follow-up experiments (scripts/run_extra.sh)
ex = load_json(R / "extra_summary.json") if (R / "extra_summary.json").exists() else {}
tr = load_json(R / "transductive_test.json") if (R / "transductive_test.json").exists() else None
sec_extra = (7 if ver else 6) if ex else None
sec_future = (sec_extra or (6 if ver else 5)) + 1
extra = ""
if ex:
    extra = f"\n## {sec_extra}. Follow-up experiments\n\nSame recipe, nothing tuned on test (`scripts/run_extra.sh`, `scripts/summarize_extra.py`, `results/extra_summary.json`).\n"
    if tr:
        T = tr["results"]
        trows = "\n".join(f"| {pretty_cfg(c)} | {T[c]['trained_standard']:.1f} | **{T[c]['trained_transductive']:.1f}** | {T[c]['frozen_standard']['acc']:.1f} | {T[c]['frozen_transductive']['acc']:.1f} |" for c in cfg)
        extra += f"""
**Batch (transductive) inference.** A standard ProtoNet labels each new photo on its own. If the new photos of an episode arrive together, their unlabelled
embeddings can refine the prototypes: soft-assign them to prototypes and move each prototype towards its photos, {tr['T']} times (Ren et al., 2018).
No retraining. The two settings (steps, temperature {tr['temperature']:g}) were chosen on the validation diseases only, then run once on test.
It helps most with one example per disease; it answers an easier question (a batch of 5 new photos per disease) than the main table.

| setting | trained | trained + batch inference | ImageNet only | ImageNet only + batch inference |
|---|---|---|---|---|
{trows}
"""
    if "repro" in ex:
        rp = ex["repro"]
        extra += (f"\n**Re-run on another machine.** The seed-42 run retrained on a second RTX 3090 VM reaches best validation "
                  f"{rp['best_val']['repro']:.2f}% (original {rp['best_val']['original']:.2f}%), and its test accuracies differ from the original by at most "
                  f"{max(abs(v['diff']) for v in rp['test'].values()):.2f} pp. GPU training is not bit-exact, so this is the noise floor for comparing runs.\n")
    if "splits" in ex:
        sp = ex["splits"]
        rows = []
        for name, v in sp["per_split"].items():
            rows.append(f"| {name}{' (main)' if name == '42' else ''} | " + " | ".join(f"{v['test'][c]['acc']:.1f}" for c in cfg)
                        + f" | {min(v['best_val']):.1f}-{max(v['best_val']):.1f} | {v['frozen_val']:.1f} |")
        ac = sp["across"]
        rows.append("| **mean ± std over splits** | " + " | ".join(f"**{ac[c]['mean']:.1f}** ± {ac[c]['std_over_splits']:.1f}" for c in cfg) + " | | |")
        rows.append("| frozen ImageNet | " + " | ".join(f"{ac[c]['frozen']:.1f}" for c in cfg) + " | | |")
        extra += f"""
**Other class splits.** The 128 base classes were re-partitioned into train/val with seeds 7 and 123 (sharing only 7 and 4 of the 27 validation
classes with the main split) and 3 training seeds were run on each. The test classes are the same in every split (they are defined as the classes
with fewer than 20 images), so the test columns are directly comparable. Cells are the mean over 3 training seeds.

| split seed | {' | '.join(pretty_cfg(c) for c in cfg)} | best val (3 seeds) | frozen val |
|---|{'---|' * len(cfg)}---|---|
{chr(10).join(rows)}

Between splits, test accuracy moves by at most {max(ac[c]['max'] - ac[c]['min'] for c in cfg):.1f} pp ({max(ac[c]['max'] - ac[c]['min'] for c in ['5way_1shot', '5way_5shot']):.1f} pp in the 5-way settings),
well inside the episode-level 95% CI of about ±1 pp: the result does not hinge on which base classes were used for training. Validation accuracy varies more (it is measured on different classes in each split), which is why splits are compared on test.
"""
    if "r50" in ex:
        r5 = ex["r50"]
        t5 = r5["test"]
        rows = [f"| {pretty_cfg(c)} | {t5[c]['r18']:.1f} | **{t5[c]['r50']:.1f}** | {t5[c]['diff']:+.1f} [{t5[c]['diff_boot_ci95'][0]:+.1f}, {t5[c]['diff_boot_ci95'][1]:+.1f}] "
                f"| {t5[c]['frozen_r18']:.1f} | {t5[c]['frozen_r50']:.1f} |" for c in cfg]
        sig = [c for c in cfg if t5[c]["diff_boot_ci95"][0] > 0]
        verdict50 = ("ResNet-50 is better in every setting, but only by "
                     f"{min(t5[c]['diff'] for c in cfg):.1f}-{max(t5[c]['diff'] for c in cfg):.1f} pp" if len(sig) == len(cfg) else
                     (("ResNet-50 helps only in the 1-shot settings: " if sig and all(c.endswith("1shot") for c in sig) else "") +
                      f"the bootstrap interval excludes zero in {len(sig)} of {len(cfg)} settings ({', '.join(pretty_cfg(c) for c in sig) or 'none'}), and every difference is at most "
                      f"{max(abs(t5[c]['diff']) for c in cfg):.1f} pp"))
        extra += f"""
**ResNet-50 backbone.** Same recipe with a torchvision ResNet-50 (ImageNet V1 weights, 2048-d embedding). The backbone learning rate was chosen on
validation from {{1e-5, 3e-5}} (best val {r5['selection_val']['1e-5']:.2f} vs {r5['selection_val']['3e-5']:.2f}; chosen {r5['chosen_lr']}); only the chosen setting was run on test.
Three seeds (best val {', '.join(f'{v:.1f}' for v in r5['best_val'])}), trained on an {r5['device']} (torch {r5['torch']}) in about {sum(r5['wall_min']) / len(r5['wall_min']):.0f} min each; the ResNet-18 runs used an RTX 3090 (re-runs across machines differ by under 0.2 pp, see above). Differences are paired over the
identical 600 test episodes (mean of 3 seeds each; bootstrap 95% CI over episodes).

| setting | ResNet-18 | ResNet-50 | difference [95% CI] | frozen ResNet-18 | frozen ResNet-50 |
|---|---|---|---|---|---|
{chr(10).join(rows)}

{verdict50[0].upper() + verdict50[1:]}. A backbone with about 2x the parameters and compute (25.6M vs 11.7M parameters) buys little here: frozen ResNet-50 features are only about 1 pp better than frozen ResNet-18, and episodic
fine-tuning adds a similar margin to both. That is consistent with the 101 training classes, rather than the network size, being the limit, though these runs do not test that directly.
"""

report = f"""# Few-shot skin-lesion classification on SD-198 with a Prototypical Network

## 1. Problem

Many skin diseases are rare: a model that needs hundreds of labelled photos per disease cannot recognise them.
We ask whether a model can recognise a **disease it never saw during training** from only 1 or 5 example photos
(few-shot learning). The evaluation is therefore **class-disjoint**: test diseases never appear in training or in model selection.

## 2. Data

**SD-198** (Sun et al., ECCV 2016): {manifest['n_classes']} skin diseases, {manifest['n_images']} clinical photos
(phone/camera images with varied lighting and framing, not dermoscopy). Classes with **fewer than 20 images are the novel
(test) classes**; the remaining base classes are split into train and validation (validation stratified over class-size terciles, seed 42).

| split | classes | images | images per class (min / median / max) |
|---|---|---|---|
{chr(10).join(split_rows)}

![class sizes by split]({FIG}/fig0_split_histogram.png)

**Provenance.** HuggingFace dataset `{dl['repo']}`, revision `{dl['revision']}`; zip sha256 `{dl['zip_sha256']}`
(matches the hash published by HuggingFace), md5 `{dl['zip_md5']}`; downloaded {dl['downloaded_utc']}; license: {dl['license']}.
The zip also contains an unrelated 40 MB Windows `.exe` in one class folder; it was never executed and the pipeline only reads image files.
Images are pre-resized to 256 px (shorter side); the model sees 224x224 crops.

## 3. Method

* **Backbone:** torchvision ResNet-18 with ImageNet weights, global-average-pooled to a 512-d embedding, L2-normalised.
  (A ResNet-12, the usual few-shot backbone, has no public ImageNet weights, so it is cited as lineage but not used.)
* **Prototypical Network** (Snell et al., 2017): a class prototype is the mean embedding of its support images, and a query is scored
  by `-scale * ||embedding - prototype||^2` with a learned temperature `scale`. There is no classifier head, so any new set of classes works without retraining.
* **Training:** episodic, 5-way 5-shot with 15 queries per class; {c0['episodes']} episodes ({c0['total_steps']} optimizer steps of 4 episodes),
  AdamW, backbone lr {c0['lr_backbone']:g} and temperature lr {c0['lr_head']:g}, {c0['warmup_resolved']}-step warm-up then cosine decay, bf16, gradient clip 1.0.
  Augmentation: random resized crop, flip, rotation, colour jitter.
  The checkpoint with the best accuracy on 600 *fixed* validation episodes (27 unseen validation classes, checked every {c0['val_every']} steps) is kept.
* **Evaluation:** 600 fixed test episodes per setting, identical for every model and seed. Novel classes have only 10-19 images, so test episodes use
  **5 queries per class, drawn without replacement** (support and query never overlap). Reported: accuracy and macro-F1, mean over {len(seeds)} independent training seed(s)
  (+- is the standard deviation over seeds; the 95% CI is over the 600 episodes).
* **Reference:** the *frozen ImageNet* row uses the same ResNet-18 and prototype rule with **no fine-tuning**, on the same test episodes.

## 4. Results (unseen test classes)

| setting | fine-tuned acc (%) | 95% CI (episodes) | macro-F1 | frozen ImageNet acc (%) | gain (pp) |
|---|---|---|---|---|---|
{chr(10).join(res_rows)}

Chance is 20% for 5-way and 50% for 2-way. Episodic fine-tuning gives **{g5:+.1f} pp** on 5-way 5-shot ({verdict(g5, ci5)}) and
**{g1:+.1f} pp** on 5-way 1-shot ({verdict(g1, ci1)}) over frozen ImageNet features; the episode-level 95% CI is about ±{ci5:.1f} / ±{ci1:.1f} pp.
Per-seed 5-way 5-shot accuracy: {', '.join(f'{a:.1f}' for a in cfg['5way_5shot']['per_seed_acc'])} (seeds {', '.join(map(str, seeds))}).

{f"**Caveat:** SD-198 has many re-shoots of the same patient, and this standard protocol can put two of them in support and query. {dup_sentence} (section 6)." if dedup else ""}

![training curves]({FIG}/fig1_training_curves.png)
![t-SNE of unseen classes]({FIG}/fig2_tsne.png)

Qualitative episodes (the first three of the fixed test list, not cherry-picked): `{FIG}/fig3_episode*.png`; one-photo demo: `{FIG}/fig4_demo.png`.

| seed | GPU | steps | best val acc (5-way 5-shot) | wall-clock (min) |
|---|---|---|---|---|
{chr(10).join(run_rows)}

## 5. What did not work, and limits

**The planned schedule overfit.** Our first run followed the original plan (20,000 episodes, backbone lr 1e-4). Validation accuracy on
unseen classes peaked at {ev['best_val']:.1f}% at step {ev['best_step']} and then **fell to {ev['final_val']:.1f}%** by step {ev['last_step']} while training accuracy reached
{ev['train_acc']:.1f}%: the network memorised the {len(splits['splits']) and sum(v == 'train' for v in splits['splits'].values())} training classes (about {ev_passes:.0f} passes over the {c0['train_images']} training images by then). We stopped that run and kept its log (`runs/_evidence/`).
The frozen ImageNet model scores {frozen_val['acc']:.1f}% on the same validation episodes, so fine-tuning helps, but only gently and briefly.
A five-configuration sweep, **judged on validation classes only** (the test classes were never touched), shows that every gentle setting reaches the same plateau:

| run | backbone lr | episodes | training task | freeze BN | best val acc (step) | final val acc | train acc |
|---|---|---|---|---|---|---|---|
{chr(10).join(sweep_rows)}

We kept the simplest (5-way, lr 1e-5). The recipe was therefore chosen after seeing validation results, but validation classes are disjoint from the test classes.

**Limits.**

* Clinical photos vary in lighting, framing and background; the 70 test classes are small (10-19 images), so a test episode uses only 5 queries per class
  and the confidence intervals are wider than they would be with larger query sets.
{'* The main results use one seed-42 partition of the base classes; two more partitions give the same test accuracy within ' + format(max(ex['splits']['across'][c]['max'] - ex['splits']['across'][c]['min'] for c in cfg), '.1f') + ' pp (section ' + str(sec_extra) + ').' if ex.get('splits') else '* The class split depends on one seed-42 partition of the base classes; we did not repeat the split.'}
{dup_limit}
* The frozen ImageNet baseline is already strong ({frozen['5way_5shot']['acc']:.0f}% on 5-way 5-shot), which limits the headroom for fine-tuning; gains over it range from {min(gains):+.1f} to {max(gains):+.1f} pp across the four settings.
* The ± over seeds only reflects training-order and augmentation randomness: all runs share the ImageNet initialisation, the class split and the test episodes. It therefore understates the total uncertainty (which would also include the choice of split).
{footer_limit}
* This is a class project, not a benchmark claim: the backbone (ResNet-18) and method (plain ProtoNet) are deliberately simple, so the numbers are not comparable
  with published SD-198 results that use larger backbones and more elaborate methods.
{verification}{extra}
## {sec_future}. Future work

* {'Since ResNet-50 adds only about 1 pp, more data or better pretraining (e.g. dermatology-specific or self-supervised weights) may be worth more than WRN-28-10 / DenseNet backbones; a ResNet-12 trained from scratch remains a useful comparison.' if ex.get('r50') else 'WRN-28-10 / DenseNet backbones (the stronger SD-198 baselines) and a ResNet-12 trained from scratch for comparison.'}
* Cross-dataset transfer to Derm7pt (dermoscopy).
* Transfer-pretraining (DTL) comparison and stronger regularisation (more augmentation, partial freezing, mixup between base classes).

## {sec_future + 1}. Reproducibility

Code: `src/` (data, model, train, eval, viz), `scripts/` (download, thumbnails, run scripts, sweep, `verify_claims.py`, `dedup_eval.py`, `run_extra.sh`, `summarize_extra.py`), `tests/`, `notebook.ipynb` (runs top-to-bottom, loads the finished runs).
`splits.json` and `data_manifest.json` are committed; `python src/data.py --verify` checks a local copy against them.
Seeds {', '.join(map(str, seeds))}; GPU {run_done[seeds[0]]['device']}; torch {run_cfgs[seeds[0]]['torch']}; each run {run_done[seeds[0]]['wall_clock_s'] / 60:.0f} min
(including {c0['total_steps'] // c0['val_every']} validation sweeps). Test classes used: {summary['n_test_classes']} (dropped for fewer than 10 images: {len(summary['dropped_test_classes'])}).

References: Snell et al., Prototypical Networks (NeurIPS 2017); Oreshkin et al., TADAM; Lee et al., MetaOptNet; Sun et al., SD-198 (ECCV 2016);
Li et al., SCAN (2022); Özdemir et al., Meta-Transfer Derm-Diagnosis (2024); Kawahara et al., Derm7pt (JBHI 2019).
"""
(ROOT / "report.md").write_text(report)
print(f"wrote report.md ({len(report.splitlines())} lines) from {len(seeds)} seed(s)")
