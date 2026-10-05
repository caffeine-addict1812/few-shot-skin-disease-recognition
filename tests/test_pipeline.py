"""End-to-end on the synthetic imagefolder: train CLI -> checkpoint -> test eval -> aggregate -> figures."""
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from conftest import ROOT
from data import SD198Split, load_eval_split
from eval import aggregate, evaluate_test
from model import ProtoNet, load_model
from utils import save_json
import viz


@pytest.fixture(scope="module")
def trained(synthetic, tmp_path_factory):
    root, splits = synthetic
    work = tmp_path_factory.mktemp("run")
    save_json(splits, work / "splits.json")
    out = work / "seed42"
    cmd = [sys.executable, str(ROOT / "src/train.py"), "--out", str(out), "--root", str(root),
           "--splits", str(work / "splits.json"), "--seed", "42", "--episodes", "8", "--eps-per-step", "2",
           "--n-way", "3", "--k-shot", "2", "--n-query", "2", "--val-every", "2", "--val-episodes", "4", "--val-way", "3", "--val-shot", "2", "--val-query", "2",
           "--log-every", "1", "--warmup", "2", "--workers", "0", "--device", "cpu", "--amp", "none",
           "--no-pretrained"]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT)
    return root, splits, out


def test_train_writes_artifacts_and_logs(trained):
    _, _, out = trained
    for f in ("best.pt", "last.pt", "log.jsonl", "config.json", "done.json"):
        assert (out / f).exists(), f
    recs = [json.loads(line) for line in open(out / "log.jsonl")]
    assert sum(r["type"] == "train" for r in recs) == 4 and sum(r["type"] == "val" for r in recs) == 2
    assert all(np.isfinite(r["loss"]) for r in recs if r["type"] == "train")
    assert json.load(open(out / "done.json"))["steps"] == 4


def test_checkpoint_roundtrip_and_resume(trained):
    root, splits, out = trained
    model, ck = load_model(out / "best.pt", torch.device("cpu"))
    assert isinstance(model, ProtoNet) and ck["step"] in (2, 4)
    work = out.parent
    cmd = [sys.executable, str(ROOT / "src/train.py"), "--out", str(out), "--root", str(root),
           "--splits", str(work / "splits.json"), "--episodes", "12", "--eps-per-step", "2", "--n-way", "3",
           "--k-shot", "2", "--n-query", "2", "--val-every", "2", "--val-episodes", "4", "--val-way", "3", "--val-shot", "2", "--val-query", "2", "--log-every", "1",
           "--warmup", "2", "--workers", "0", "--device", "cpu", "--amp", "none", "--no-pretrained",
           "--resume"]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT)
    assert "resumed from step 4" in res.stdout
    assert json.load(open(out / "done.json"))["steps"] == 6


def test_test_eval_aggregate_and_figures(trained, tmp_path):
    root, splits, out = trained
    model, _ = load_model(out / "best.pt", torch.device("cpu"))
    res = evaluate_test(model, root, splits, torch.device("cpu"), n_episodes=2)
    assert [(r["n_way"], r["k_shot"]) for r in res["results"]] == [(5, 1), (5, 5), (2, 1), (2, 5)]
    assert res["n_test_classes"] == 6 and res["dropped_test_classes"] == ["tiny_00"]
    assert all(r["n_query"] == 5 for r in res["results"])

    files = []
    for s in (42, 43):
        f = tmp_path / f"seed{s}.json"
        save_json({**res, "seed": s}, f)
        files.append(f)
    summ = aggregate(files, frozen_file=files[0])
    assert summ["n_seeds"] == 2 and "5way_5shot" in summ["configs"] and "frozen_baseline" in summ

    ds = load_eval_split(root, splits, "test", min_images=10)
    ids = viz.pick_classes(ds, 6)
    emb, lab = viz.extract_embeddings(model, ds, ids, torch.device("cpu"))
    assert emb.shape[1] == 512 and len(emb) == len(lab)
    viz.plot_tsne(emb, lab, ds.classes, tmp_path / "tsne.png")
    viz.plot_training_curves([out], tmp_path / "curves.png")
    from data import EpisodeDataset
    ep = EpisodeDataset(ds, 5, 1, 5, 1, seed=3).episode(0)
    viz.plot_episode(model, ds, ep, torch.device("cpu"), tmp_path / "episode.png")
    d2 = viz.episode_distances(model, ds, ep[1], ep[2], torch.device("cpu"))
    viz.plot_distance_bars(d2[0], [ds.classes[c] for c in ep[0]], true_idx=0, out_path=tmp_path / "bars.png")
    for f in ("tsne", "curves", "episode", "bars"):
        assert (tmp_path / f"{f}.png").stat().st_size > 2000
