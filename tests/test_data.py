import numpy as np
import pytest
import torch

from conftest import COUNTS
from data import (EpisodeDataset, SD198Split, build_manifest, filelist_hash, load_eval_split, make_splits,
                  sample_episode, scan_images)


def test_make_splits_disjoint_and_thresholded():
    sp = make_splits(COUNTS, seed=42, novel_thresh=20)
    assert set(sp) == set(COUNTS)
    assert {c for c, v in sp.items() if v == "test"} == {c for c, n in COUNTS.items() if n < 20}
    base = [c for c, v in sp.items() if v != "test"]
    assert all(COUNTS[c] >= 20 for c in base)
    assert {v for v in sp.values()} == {"train", "val", "test"}
    assert make_splits(COUNTS, seed=42) == sp                      # deterministic
    assert any(make_splits(COUNTS, seed=s) != sp for s in range(1, 8))  # seed actually matters


def test_val_is_stratified_over_count_terciles():
    counts = {f"c{i:03d}": 20 + i for i in range(60)}
    sp = make_splits(counts, seed=0)
    val = sorted(counts[c] for c, v in sp.items() if v == "val")
    assert len(val) == 12  # round(20 * 0.22) = 4 per tercile
    assert sum(v < 40 for v in val) == 4 and sum(v >= 60 for v in val) == 4  # every tercile represented


def test_episode_support_query_disjoint_and_class_distinct():
    class_indices = [np.arange(i * 100, i * 100 + n) for i, n in enumerate([20, 25, 30, 22, 21, 40])]
    for s in range(50):
        classes, sup, qry = sample_episode(class_indices, 5, 5, 15, np.random.default_rng(s))
        assert len(set(classes.tolist())) == 5
        for r, c in enumerate(classes):
            assert not set(sup[r]) & set(qry[r])
            assert len(set(sup[r]) | set(qry[r])) == 20
            assert set(sup[r]) | set(qry[r]) <= set(class_indices[c])


def test_episode_deterministic_in_seed_and_index(synthetic):
    root, splits = synthetic
    ds = SD198Split(root, splits, "train")
    a = EpisodeDataset(ds, 3, 2, 2, 10, seed=7)
    b = EpisodeDataset(ds, 3, 2, 2, 10, seed=7)
    for i in (0, 3, 9):
        for x, y in zip(a.episode(i), b.episode(i)):
            np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a.episode(0)[1], a.episode(1)[1])
    # offset (resume) shifts the episode stream without changing episode content
    c = EpisodeDataset(ds, 3, 2, 2, 5, seed=7, offset=4)
    np.testing.assert_array_equal(c.episode(4)[1], a.episode(4)[1])


def test_episode_rejects_classes_too_small(synthetic):
    root, splits = synthetic
    ds = SD198Split(root, splits, "test")  # includes the 7-image class
    with pytest.raises(ValueError, match="images"):
        EpisodeDataset(ds, 2, 5, 5, 10, seed=0)


def test_test_split_drops_small_classes(synthetic):
    root, splits = synthetic
    ds = SD198Split(root, splits, "test", min_images=10)
    assert ds.dropped_classes == ["tiny_00"]
    assert all(len(ix) >= 10 for ix in ds.class_indices)
    assert len(ds.classes) == 6


def test_splits_class_disjoint_in_loaded_datasets(synthetic):
    root, splits = synthetic
    names = {s: set(SD198Split(root, splits, s).classes) for s in ("train", "val", "test")}
    assert not names["train"] & names["val"] and not names["train"] & names["test"] and not names["val"] & names["test"]


def test_episode_tensor_layout_support_first_class_major():
    class Fake:  # image value == its dataset index, so layout is directly checkable
        classes = ["a", "b", "c"]
        class_indices = [np.arange(0, 10), np.arange(10, 20), np.arange(20, 30)]

        def __getitem__(self, j):
            return torch.full((3, 4, 4), float(j)), 0

    ed = EpisodeDataset(Fake(), 2, 3, 2, 4, seed=1)
    x, classes = ed[2]
    _, sup, qry = ed.episode(2)
    assert x.shape == (2 * (3 + 2), 3, 4, 4) and classes.shape == (2,)
    np.testing.assert_array_equal(x[:, 0, 0, 0].numpy(), np.concatenate([sup.ravel(), qry.ravel()]))


def test_cached_split_matches_uncached(synthetic):
    root, splits = synthetic
    cached = load_eval_split(root, splits, "val")
    plain = load_eval_split(root, splits, "val", cache=False)
    assert len(cached) == len(plain) and cached.classes == plain.classes
    assert torch.allclose(cached[3][0], plain[3][0], atol=2e-3)  # fp16 storage


def test_manifest_hash_depends_on_files_not_bytes(synthetic):
    root, _ = synthetic
    images = scan_images(root)
    m = build_manifest(images)
    assert m["n_classes"] == len(COUNTS) and m["n_images"] == sum(COUNTS.values())
    images2 = {c: list(v) for c, v in images.items()}
    images2["base_00"].pop()
    assert filelist_hash(images2) != filelist_hash(images)


def test_cached_gather_matches_episode_getitem(synthetic):
    """eval.episode_batches indexes the in-memory cache directly; it must equal the per-episode path."""
    from eval import episode_batches

    root, splits = synthetic
    ds = load_eval_split(root, splits, "val")
    eds = EpisodeDataset(ds, 2, 2, 3, 6, seed=11)
    batches = list(episode_batches(ds, eds, 4, torch.device("cpu")))
    assert [b.shape[0] for b in batches] == [4, 2]  # last batch is the remainder
    for i in range(6):
        x, _ = eds[i]
        torch.testing.assert_close(batches[i // 4][i % 4], x)
