# Few-shot skin-lesion classification on SD-198 with a Prototypical Network

## 1. Problem

Many skin diseases are rare: a model that needs hundreds of labelled photos per disease cannot recognise them.
We ask whether a model can recognise a **disease it never saw during training** from only 1 or 5 example photos
(few-shot learning). The evaluation is therefore **class-disjoint**: test diseases never appear in training or in model selection.

## 2. Data

**SD-198** (Sun et al., ECCV 2016): 198 skin diseases, 6584 clinical photos
(phone/camera images with varied lighting and framing, not dermoscopy). Classes with **fewer than 20 images are the novel
(test) classes**; the remaining base classes are split into train and validation (validation stratified over class-size terciles, seed 42).

| split | classes | images | images per class (min / median / max) |
|---|---|---|---|
| train | 101 | 4428 | 20 / 45 / 60 |
| val | 27 | 1191 | 20 / 49 / 60 |
| test | 70 | 965 | 10 / 14 / 19 |

![class sizes by split](results/figures/fig0_split_histogram.png)

**Provenance.** HuggingFace dataset `resyhgerwshshgdfghsdfgh/SD-198`, revision `68c02939c06d6850f40f2911e1d1d49bd760139d`; zip sha256 `71bf8d783c72c68add6cafae17e09b7f54e966c7f86efce9c86c903e1dd96858`
(matches the hash published by HuggingFace), md5 `1facac8850572a39798db63ca0f57a4b`; downloaded 2026-10-02T18:52:39+00:00; license: other (HF card page lists 'unknown'); original: Sun et al., ECCV 2016.
The zip also contains an unrelated 40 MB Windows `.exe` in one class folder; it was never executed and the pipeline only reads image files.
Images are pre-resized to 256 px (shorter side); the model sees 224x224 crops.

## 3. Method

* **Backbone:** torchvision ResNet-18 with ImageNet weights, global-average-pooled to a 512-d embedding, L2-normalised.
  (A ResNet-12, the usual few-shot backbone, has no public ImageNet weights, so it is cited as lineage but not used.)
* **Prototypical Network** (Snell et al., 2017): a class prototype is the mean embedding of its support images, and a query is scored
  by `-scale * ||embedding - prototype||^2` with a learned temperature `scale`. There is no classifier head, so any new set of classes works without retraining.
* **Training:** episodic, 5-way 5-shot with 15 queries per class; 4000 episodes (1000 optimizer steps of 4 episodes),
  AdamW, backbone lr 1e-05 and temperature lr 0.001, 100-step warm-up then cosine decay, bf16, gradient clip 1.0.
  Augmentation: random resized crop, flip, rotation, colour jitter.
  The checkpoint with the best accuracy on 600 *fixed* validation episodes (27 unseen validation classes, checked every 50 steps) is kept.
* **Evaluation:** 600 fixed test episodes per setting, identical for every model and seed. Novel classes have only 10-19 images, so test episodes use
  **5 queries per class, drawn without replacement** (support and query never overlap). Reported: accuracy and macro-F1, mean over 3 independent training seed(s)
  (+- is the standard deviation over seeds; the 95% CI is over the 600 episodes).
* **Reference:** the *frozen ImageNet* row uses the same ResNet-18 and prototype rule with **no fine-tuning**, on the same test episodes.

## 4. Results (unseen test classes)

| setting | fine-tuned acc (%) | 95% CI (episodes) | macro-F1 | frozen ImageNet acc (%) | gain (pp) |
|---|---|---|---|---|---|
| 5-way 1-shot | **70.7** ± 0.01 | ±1.0 | 68.6 | 62.6 ± 1.1 | +8.1 |
| 5-way 5-shot | **85.9** ± 0.05 | ±0.8 | 85.3 | 80.2 ± 0.8 | +5.6 |
| 2-way 1-shot | **86.4** ± 0.14 | ±1.2 | 85.6 | 81.4 ± 1.4 | +5.0 |
| 2-way 5-shot | **94.5** ± 0.11 | ±0.7 | 94.4 | 91.7 ± 0.9 | +2.8 |

Chance is 20% for 5-way and 50% for 2-way. Episodic fine-tuning gives **+5.6 pp** on 5-way 5-shot (a clear gain) and
**+8.1 pp** on 5-way 1-shot (a clear gain) over frozen ImageNet features; the episode-level 95% CI is about ±0.8 / ±1.0 pp.
Per-seed 5-way 5-shot accuracy: 85.8, 85.9, 85.8 (seeds 42, 43, 44).

**Caveat:** SD-198 has many re-shoots of the same patient, and this standard protocol can put two of them in support and query. Removing near-duplicate re-shoots of the same patient costs about 1.2 pp (5-way 1-shot) and 0.8 pp (5-way 5-shot) on matched classes, and at most 7.2 / 4.9 pp if the 21 classes that fall below 10 images without their re-shoots are dropped as well; the gain over frozen ImageNet is unchanged (section 6).

![training curves](results/figures/fig1_training_curves.png)
![t-SNE of unseen classes](results/figures/fig2_tsne.png)

Qualitative episodes (the first three of the fixed test list, not cherry-picked): `results/figures/fig3_episode*.png`; one-photo demo: `results/figures/fig4_demo.png`.

| seed | GPU | steps | best val acc (5-way 5-shot) | wall-clock (min) |
|---|---|---|---|---|
| 42 | NVIDIA GeForce RTX 3090 | 1000 | 78.85 (step 1000) | 6.7 |
| 43 | NVIDIA GeForce RTX 3090 | 1000 | 78.87 (step 950) | 6.6 |
| 44 | NVIDIA GeForce RTX 3090 | 1000 | 78.54 (step 800) | 6.6 |

## 5. What did not work, and limits

**The planned schedule overfit.** Our first run followed the original plan (20,000 episodes, backbone lr 1e-4). Validation accuracy on
unseen classes peaked at 78.2% at step 250 and then **fell to 73.8%** by step 1500 while training accuracy reached
99.4%: the network memorised the 101 training classes (about 136 passes over the 4428 training images by then). We stopped that run and kept its log (`runs/_evidence/`).
The frozen ImageNet model scores 70.9% on the same validation episodes, so fine-tuning helps, but only gently and briefly.
A five-configuration sweep, **judged on validation classes only** (the test classes were never touched), shows that every gentle setting reaches the same plateau:

| run | backbone lr | episodes | training task | freeze BN | best val acc (step) | final val acc | train acc |
|---|---|---|---|---|---|---|---|
| w15_lr1e-5 | 1e-05 | 4000 | 15-way | no | 79.20 ± 0.66 (step 1000) | 79.20 | 80.1 |
| lr3e-5 | 3e-05 | 4000 | 5-way | no | 78.99 ± 0.66 (step 500) | 78.72 | 95.2 |
| w15_lr3e-5 | 3e-05 | 4000 | 15-way | no | 78.98 ± 0.65 (step 400) | 77.97 | 93.7 |
| lr1e-5 | 1e-05 | 4000 | 5-way | no | 78.85 ± 0.67 (step 1000) | 78.85 | 87.6 |
| lr1e-5_fbn | 1e-05 | 4000 | 5-way | yes | 78.46 ± 0.66 (step 400) | 78.42 | 88.9 |
| full20k_lr1e-4 | 0.0001 | 20000 | 5-way | no | 78.23 ± 0.68 (step 250) | 73.82 | 99.4 |

We kept the simplest (5-way, lr 1e-5). The recipe was therefore chosen after seeing validation results, but validation classes are disjoint from the test classes.

**Limits.**

* Clinical photos vary in lighting, framing and background; the 70 test classes are small (10-19 images), so a test episode uses only 5 queries per class
  and the confidence intervals are wider than they would be with larger query sets.
* The main results use one seed-42 partition of the base classes; two more partitions give the same test accuracy within 0.6 pp (section 7).
* **Near-duplicates (checked).** By frozen-ImageNet cosine similarity > 0.95, 5 of 965 test images have a near-duplicate in train/val (always filed under a *different* disease, so no test label leaks) and 23 same-class pairs exist inside the test set. One photo is filed, byte-identical, as both Pityriasis Rosea (val) and Pityriasis Alba (test): label noise in SD-198 itself. See `results/figures/fig5_near_duplicates.png`. Removing near-duplicate re-shoots of the same patient costs about 1.2 pp (5-way 1-shot) and 0.8 pp (5-way 5-shot) on matched classes, and at most 7.2 / 4.9 pp if the 21 classes that fall below 10 images without their re-shoots are dropped as well; the gain over frozen ImageNet is unchanged (section 6).
* The frozen ImageNet baseline is already strong (80% on 5-way 5-shot), which limits the headroom for fine-tuning; gains over it range from +2.8 to +8.1 pp across the four settings.
* The ± over seeds only reflects training-order and augmentation randomness: all runs share the ImageNet initialisation, the class split and the test episodes. It therefore understates the total uncertainty (which would also include the choice of split).
* Many SD-198 photos carry a grey source footer strip; we did not remove it. Masking the bottom 15% of the image changes fine-tuned accuracy by at most 0.9 pp, the same as masking the top 15% (0.8 pp), so the model does not rely on it.
* This is a class project, not a benchmark claim: the backbone (ResNet-18) and method (plain ProtoNet) are deliberately simple, so the numbers are not comparable
  with published SD-198 results that use larger backbones and more elaborate methods.

## 6. Independent verification

`scripts/verify_claims.py` re-derives the results through a separate code path (fp32 on CPU, images decoded straight from disk,
a numpy nearest-prototype scorer; only the episode definition is shared). `results/verification.json` holds the output.

* **Reproduction:** every stored accuracy (4 settings x 3 seeds + frozen) is reproduced within 0.08 pp (GPU bf16 vs CPU fp32).
* **Episode choice:** on 5,000 *new* test episodes the accuracies stay within 1.1 pp of the 600-episode numbers.
* **Metric sanity:** shuffling query labels gives 19.7% (5-way) and 49.4% (2-way), i.e. chance.
* **Model selection:** each `best.pt` is the arg-max of its logged validation accuracy, and the training logs contain only train/val records.
* **Gain over frozen ImageNet, paired over the same episodes** (mean of 3 seeds; bootstrap 95% CI over episodes):

| setting | gain (pp) | 95% CI | episodes won / tied / lost |
|---|---|---|---|
| 5-way 1-shot | +8.0 | [+7.1, +8.8] | 449 / 22 / 129 |
| 5-way 5-shot | +5.6 | [+5.0, +6.3] | 422 / 62 / 116 |
| 2-way 1-shot | +5.1 | [+3.9, +6.3] | 276 / 185 / 139 |
| 2-way 5-shot | +2.8 | [+2.1, +3.6] | 187 / 327 / 86 |

* **Seed agreement:** the three seeds disagree on about half of the 5-way 1-shot episodes, yet their means agree within 0.02 pp; that closeness is partly chance
  (other settings differ by 0.1-0.3 pp between seeds).

**De-duplicated test set** (`scripts/dedup_eval.py`). Inside each test class, images are grouped by frozen-ImageNet cosine similarity and only one per group is kept,
so a query cannot be a near-copy of its own support image. Cells: fine-tuned accuracy (frozen ImageNet in brackets), 600 episodes each.

| near-duplicate threshold | images | classes | 5-way 1-shot | 5-way 5-shot | 2-way 1-shot | 2-way 5-shot |
|---|---|---|---|---|---|---|
| none (full test set) | 965 | 70 | 70.7 (62.7) | 85.8 (80.2) | 86.4 (81.3) | 94.5 (91.7) |
| cosine > 0.95 | 933 | 69 | 69.3 (61.1) | 85.4 (79.6) | 87.1 (81.9) | 95.0 (92.2) |
| cosine > 0.9 | 616 | 49 | 63.5 (55.4) | 81.0 (74.7) | 83.4 (77.1) | 93.2 (89.6) |
| control: the same 49 classes, all images | 703 | 49 | 64.7 (57.2) | 81.7 (75.7) | 84.2 (78.2) | 94.0 (90.6) |

**What this means.** Looking at pairs by eye, both thresholds mostly catch *the same patient photographed again* (another angle or distance), not
different patients who look alike: SD-198 contains many such multi-view series, and the standard protocol (used here and in published SD-198 work)
lets a support and a query photo come from the same patient. The strict threshold also removes 21 classes (those left with fewer than 10 images),
so the control row keeps exactly the surviving classes with all their images. Of the 7.2 pp drop on 5-way 1-shot,
**6.0 pp comes from the change of classes and 1.2 pp from removing duplicates** (5-way 5-shot:
4.1 and 0.8 pp). The split is not perfectly clean: a class is removed *because* it contained re-shoots
(mostly the smallest classes, 10-18 images, where losing a few photos drops it below 10), so the effect of duplicates lies between about
1 and 7 pp, with the matched-class 1.2 pp the more direct measure. An embedding threshold will not catch every same-patient pair.
The gain over frozen ImageNet holds in every row (+8.0, +8.2, +8.1, +7.5 pp on 5-way 1-shot).

## 7. Follow-up experiments

Same recipe, nothing tuned on test (`scripts/run_extra.sh`, `scripts/summarize_extra.py`, `results/extra_summary.json`).

**Batch (transductive) inference.** A standard ProtoNet labels each new photo on its own. If the new photos of an episode arrive together, their unlabelled
embeddings can refine the prototypes: soft-assign them to prototypes and move each prototype towards its photos, 8 times (Ren et al., 2018).
No retraining. The two settings (steps, temperature 10) were chosen on the validation diseases only, then run once on test.
It helps most with one example per disease; it answers an easier question (a batch of 5 new photos per disease) than the main table.

| setting | trained | trained + batch inference | ImageNet only | ImageNet only + batch inference |
|---|---|---|---|---|
| 5-way 1-shot | 70.7 | **77.3** | 62.7 | 68.5 |
| 5-way 5-shot | 85.8 | **86.8** | 80.2 | 81.0 |
| 2-way 1-shot | 86.4 | **91.3** | 81.3 | 86.9 |
| 2-way 5-shot | 94.5 | **95.1** | 91.7 | 92.4 |

**Re-run on another machine.** The seed-42 run retrained on a second RTX 3090 VM reaches best validation 78.83% (original 78.85%), and its test accuracies differ from the original by at most 0.18 pp. GPU training is not bit-exact, so this is the noise floor for comparing runs.

**Other class splits.** The 128 base classes were re-partitioned into train/val with seeds 7 and 123 (sharing only 7 and 4 of the 27 validation
classes with the main split) and 3 training seeds were run on each. The test classes are the same in every split (they are defined as the classes
with fewer than 20 images), so the test columns are directly comparable. Cells are the mean over 3 training seeds.

| split seed | 5-way 1-shot | 5-way 5-shot | 2-way 1-shot | 2-way 5-shot | best val (3 seeds) | frozen val |
|---|---|---|---|---|---|---|
| 42 (main) | 70.7 | 85.9 | 86.4 | 94.5 | 78.5-78.9 | 70.9 |
| 7 | 70.7 | 86.1 | 86.5 | 94.6 | 79.7-80.1 | 72.6 |
| 123 | 71.0 | 86.0 | 87.0 | 94.9 | 77.0-77.3 | 68.3 |
| **mean ± std over splits** | **70.8** ± 0.2 | **86.0** ± 0.1 | **86.6** ± 0.3 | **94.7** ± 0.2 | | |
| frozen ImageNet | 62.6 | 80.2 | 81.4 | 91.7 | | |

Between splits, test accuracy moves by at most 0.6 pp (0.3 pp in the 5-way settings),
well inside the episode-level 95% CI of about ±1 pp: the result does not hinge on which base classes were used for training. Validation accuracy varies more (it is measured on different classes in each split), which is why splits are compared on test.

**ResNet-50 backbone.** Same recipe with a torchvision ResNet-50 (ImageNet V1 weights, 2048-d embedding). The backbone learning rate was chosen on
validation from {1e-5, 3e-5} (best val 79.48 vs 78.79; chosen 1e-5); only the chosen setting was run on test.
Three seeds (best val 79.5, 79.3, 79.1), trained on an NVIDIA GeForce RTX 4090 (torch 2.14.1+cu126) in about 11 min each; the ResNet-18 runs used an RTX 3090 (re-runs across machines differ by under 0.2 pp, see above). Differences are paired over the
identical 600 test episodes (mean of 3 seeds each; bootstrap 95% CI over episodes).

| setting | ResNet-18 | ResNet-50 | difference [95% CI] | frozen ResNet-18 | frozen ResNet-50 |
|---|---|---|---|---|---|
| 5-way 1-shot | 70.7 | **71.5** | +0.8 [+0.1, +1.5] | 62.6 | 63.3 |
| 5-way 5-shot | 85.9 | **86.2** | +0.4 [-0.1, +0.9] | 80.2 | 81.8 |
| 2-way 1-shot | 86.4 | **87.7** | +1.3 [+0.4, +2.2] | 81.4 | 81.3 |
| 2-way 5-shot | 94.5 | **94.6** | +0.1 [-0.4, +0.6] | 91.7 | 91.9 |

ResNet-50 helps only in the 1-shot settings: the bootstrap interval excludes zero in 2 of 4 settings (5-way 1-shot, 2-way 1-shot), and every difference is at most 1.3 pp. A backbone with about 2x the parameters and compute (25.6M vs 11.7M parameters) buys little here: frozen ResNet-50 features are only about 1 pp better than frozen ResNet-18, and episodic
fine-tuning adds a similar margin to both. That is consistent with the 101 training classes, rather than the network size, being the limit, though these runs do not test that directly.

## 8. Future work

* Since ResNet-50 adds only about 1 pp, more data or better pretraining (e.g. dermatology-specific or self-supervised weights) may be worth more than WRN-28-10 / DenseNet backbones; a ResNet-12 trained from scratch remains a useful comparison.
* Cross-dataset transfer to Derm7pt (dermoscopy).
* Transfer-pretraining (DTL) comparison and stronger regularisation (more augmentation, partial freezing, mixup between base classes).

## 9. Reproducibility

Code: `src/` (data, model, train, eval, viz), `scripts/` (download, thumbnails, run scripts, sweep, `verify_claims.py`, `dedup_eval.py`, `run_extra.sh`, `summarize_extra.py`), `tests/`, `notebook.ipynb` (runs top-to-bottom, loads the finished runs).
`splits.json` and `data_manifest.json` are committed; `python src/data.py --verify` checks a local copy against them.
Seeds 42, 43, 44; GPU NVIDIA GeForce RTX 3090; torch 2.14.1+cu126; each run 7 min
(including 20 validation sweeps). Test classes used: 70 (dropped for fewer than 10 images: 0).

References: Snell et al., Prototypical Networks (NeurIPS 2017); Oreshkin et al., TADAM; Lee et al., MetaOptNet; Sun et al., SD-198 (ECCV 2016);
Li et al., SCAN (2022); Özdemir et al., Meta-Transfer Derm-Diagnosis (2024); Kawahara et al., Derm7pt (JBHI 2019).
