"""Transductive prototypes (Ren et al., 2018, "soft k-means" refinement), as an inference-time option.

    python scripts/transductive_eval.py        # -> results/transductive_test.json

Standard ProtoNet labels each query photo on its own. The transductive variant also uses the episode's other
*unlabelled* query photos: it softly assigns them to prototypes and moves each prototype towards its assigned photos,
T times, then classifies. No retraining; it reuses the saved embeddings. It assumes the new photos of an episode
arrive together as a batch (here 5 per disease), so it answers a different, easier question than the main results.

Settings were chosen on the 27 VALIDATION diseases only (5-way, 5 queries, 600 episodes, val seed):
  T in {1,2,4,8} x temperature in {5,10,20}; best mean over 1- and 5-shot: T=8, temperature 10
  (val 1-shot 64.0 -> 68.8, 5-shot 78.7 -> 79.7). Centring on the training mean (SimpleShot) did not help on
  validation and is not used. This script then runs once on the test diseases.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import EpisodeDataset, SD198Split, load_splits  # noqa: E402
from eval import MIN_TEST_IMAGES, N_QUERY_TEST, TEST_CONFIGS, TEST_SEED  # noqa: E402
from utils import mean_ci95, save_json  # noqa: E402

T_STEPS, TEMP = 8, 10.0          # fixed on validation (see docstring)


def episode_acc(Z, sup, qry, n, k, T):
    S, Q = Z[sup], Z[qry.ravel()]
    P = S.mean(1)
    for _ in range(T):
        d2 = ((Q[:, None] - P[None]) ** 2).sum(-1)
        w = np.exp(-TEMP * (d2 - d2.min(1, keepdims=True)))
        w /= w.sum(1, keepdims=True)
        P = (S.sum(1) + w.T @ Q) / (k + w.sum(0)[:, None])
    d2 = ((Q[:, None] - P[None]) ** 2).sum(-1)
    return 100 * (d2.argmin(1) == np.repeat(np.arange(n), qry.shape[1])).mean()


test = SD198Split(ROOT / "data/SD-198-256", load_splits(ROOT / "splits.json"), "test", min_images=MIN_TEST_IMAGES)
E = np.load(ROOT / "results/test_embeddings.npz")
out = dict(method="transductive soft k-means prototypes", T=T_STEPS, temperature=TEMP, selected_on="validation", results={})
for n, k in TEST_CONFIGS:
    eds = EpisodeDataset(test, n, k, N_QUERY_TEST, 600, TEST_SEED)
    eps = [eds.episode(e) for e in range(600)]
    row = {}
    for m in ("frozen", "seed42", "seed43", "seed44"):
        for T, tag in ((0, "standard"), (T_STEPS, "transductive")):
            a = np.array([episode_acc(E[m], sup, qry, n, k, T) for _, sup, qry in eps])
            row[f"{m}_{tag}"] = dict(acc=float(a.mean()), ci95=float(mean_ci95(a)[1]))
    for tag in ("standard", "transductive"):
        row[f"trained_{tag}"] = float(np.mean([row[f"seed{s}_{tag}"]["acc"] for s in (42, 43, 44)]))
    out["results"][f"{n}way_{k}shot"] = row
    print(f"{n}-way {k}-shot: trained {row['trained_standard']:.2f} -> {row['trained_transductive']:.2f} | "
          f"ImageNet only {row['frozen_standard']['acc']:.2f} -> {row['frozen_transductive']['acc']:.2f}", flush=True)
save_json(out, ROOT / "results/transductive_test.json")
