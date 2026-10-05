import numpy as np
import torch
import torch.nn.functional as F

from conftest import small_transform
from data import EpisodeDataset, SD198Split
from eval import episode_metrics, run_eval
from model import ProtoNet, episode_labels, proto_logits, set_bn_eval
from utils import mean_ci95


def test_proto_logits_is_negative_scaled_squared_distance():
    support = torch.tensor([[[[0.0, 0.0], [2.0, 0.0]], [[10.0, 0.0], [10.0, 2.0]]]])  # [1, 2, 2, 2]
    query = torch.tensor([[[1.0, 0.0], [10.0, 1.0]]])                                   # [1, 2, 2]
    out = proto_logits(support, query, scale=3.0)
    protos = torch.tensor([[1.0, 0.0], [10.0, 1.0]])
    expected = -3.0 * ((query[0, :, None, :] - protos[None]) ** 2).sum(-1)
    torch.testing.assert_close(out[0], expected)
    assert out[0].argmax(-1).tolist() == [0, 1]


def test_forward_shapes_and_unit_norm_embeddings():
    m = ProtoNet(pretrained=False)
    x = torch.randn(2, 3 * (2 + 1), 3, 64, 64)
    logits = m(x, n_way=3, k_shot=2, n_query=1)
    assert logits.shape == (2, 3, 3) and logits.dtype == torch.float32
    z = m.embed(x[0])
    torch.testing.assert_close(z.norm(dim=-1), torch.ones(len(z)), atol=1e-5, rtol=0)


def test_no_l2_norm_has_unit_scale_and_scale_is_learnable():
    assert ProtoNet(pretrained=False, l2_norm=False).scale.item() == 1.0
    m = ProtoNet(pretrained=False)
    assert abs(m.scale.item() - 10.0) < 1e-5 and m.log_scale.requires_grad


def test_set_bn_eval_freezes_running_stats():
    m = ProtoNet(pretrained=False)
    m.train()
    set_bn_eval(m)
    bn = m.encoder.net.bn1
    before = bn.running_mean.clone()
    m(torch.randn(1, 6, 3, 64, 64), 2, 2, 1)
    torch.testing.assert_close(bn.running_mean, before)
    assert m.encoder.net.conv1.training  # non-BN modules still in train mode


def test_episode_metrics_known_values():
    labels = torch.tensor([[0, 0, 1, 1]])
    perfect = torch.tensor([[[5.0, 0], [5.0, 0], [0, 5.0], [0, 5.0]]])
    acc, f1 = episode_metrics(perfect, labels, 2)
    assert acc.item() == 1.0 and f1.item() == 1.0
    all_zero = torch.tensor([[[5.0, 0]] * 4])
    acc, f1 = episode_metrics(all_zero, labels, 2)
    assert acc.item() == 0.5
    assert abs(f1.item() - (2 * 2 / (2 * 2 + 2) + 0.0) / 2) < 1e-6  # class0 F1=2/3, class1 F1=0


def test_mean_ci95():
    m, ci = mean_ci95([1.0] * 10)
    assert m == 1.0 and ci == 0.0
    m, ci = mean_ci95(np.arange(100.0))
    assert abs(m - 49.5) < 1e-9 and abs(ci - 1.96 * np.std(np.arange(100.0), ddof=1) / 10) < 1e-9


def test_overfit_one_episode_reaches_near_zero_loss(synthetic):
    """The bug catcher from the plan: a fixed episode must be memorised (loss -> ~0)."""
    root, splits = synthetic
    torch.manual_seed(0)
    ds = SD198Split(root, splits, "train", small_transform(64))
    n_way, k, q = 3, 2, 2
    x, _ = EpisodeDataset(ds, n_way, k, q, 1, seed=0)[0]
    x = x.unsqueeze(0)
    labels = episode_labels(n_way, q, 1)
    model = ProtoNet(pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss = None
    for _ in range(120):
        model.train()
        loss = F.cross_entropy(model(x, n_way, k, q).flatten(0, 1), labels.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, loss.item()


def test_run_eval_reproducible_and_paired(synthetic):
    root, splits = synthetic
    ds = SD198Split(root, splits, "val", small_transform(64))
    model = ProtoNet(pretrained=False).eval()
    a = run_eval(model, ds, 2, 2, 3, 6, seed=1000, device=torch.device("cpu"))
    b = run_eval(model, ds, 2, 2, 3, 6, seed=1000, device=torch.device("cpu"))
    assert a["acc_per_episode"] == b["acc_per_episode"]
    assert 0 <= a["acc"] <= 100 and a["n_episodes"] == 6


def test_resnet50_backbone_and_old_checkpoint_compat(tmp_path):
    from model import load_model
    m = ProtoNet(pretrained=False, backbone="resnet50").eval()
    assert m.cfg["backbone"] == "resnet50" and m.encoder.out_dim == 2048
    with torch.no_grad():
        z = m.embed(torch.randn(2, 3, 64, 64))
    assert z.shape == (2, 2048)
    # checkpoints written before the backbone option existed have no "backbone" key -> ResNet-18
    old = ProtoNet(pretrained=False)
    cfg = {k: v for k, v in old.cfg.items() if k != "backbone"}
    torch.save(dict(model=old.state_dict(), model_cfg=cfg), tmp_path / "old.pt")
    loaded, _ = load_model(tmp_path / "old.pt", "cpu")
    assert loaded.encoder.out_dim == 512
