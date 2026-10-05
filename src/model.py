import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights, ResNet50_Weights

# name -> (constructor, ImageNet weights, embedding dim). Both use the V1 ImageNet recipe, so a
# ResNet-18 vs ResNet-50 comparison changes only the network size, not the pretraining recipe.
BACKBONES = {
    "resnet18": (torchvision.models.resnet18, ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet50": (torchvision.models.resnet50, ResNet50_Weights.IMAGENET1K_V1, 2048),
}


class ResNetEncoder(nn.Module):
    """torchvision ResNet without the classifier: image -> global-avg-pooled embedding."""

    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        ctor, weights, self.out_dim = BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x):
        return self.net(x)


ResNet18Encoder = ResNetEncoder  # backwards-compatible name


class ProtoNet(nn.Module):
    """Prototypical network: logits = -scale * ||query - prototype||^2.

    Episode tensors are laid out [E, N*(K+Q), C, H, W]: the N*K support images first
    (class-major), then the N*Q query images (class-major). Labels are therefore implicit:
    support/query labels are arange(N).repeat_interleave(K / Q).
    """

    def __init__(self, pretrained=True, l2_norm=True, learnable_scale=True, init_scale=None, backbone="resnet18"):
        super().__init__()
        self.cfg = dict(pretrained=pretrained, l2_norm=l2_norm,
                        learnable_scale=learnable_scale, init_scale=init_scale, backbone=backbone)
        self.encoder = ResNetEncoder(backbone, pretrained=pretrained)
        self.l2_norm = l2_norm
        if init_scale is None:
            # unit-norm embeddings give squared distances in [0, 4], so they need a temperature
            init_scale = 10.0 if l2_norm else 1.0
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)),
                                      requires_grad=learnable_scale)

    @property
    def scale(self):
        return self.log_scale.exp()

    def embed(self, x):
        """[B, C, H, W] -> [B, D] in fp32 (normalised if l2_norm)."""
        z = self.encoder(x).float()
        return F.normalize(z, dim=-1) if self.l2_norm else z

    def forward(self, x, n_way, k_shot, n_query):
        """x: [E, N*(K+Q), C, H, W] -> logits [E, N*Q, N] (fp32)."""
        E, M = x.shape[:2]
        assert M == n_way * (k_shot + n_query), (M, n_way, k_shot, n_query)
        z = self.embed(x.flatten(0, 1)).view(E, M, -1)
        support = z[:, : n_way * k_shot].view(E, n_way, k_shot, -1)
        query = z[:, n_way * k_shot:]
        return proto_logits(support, query, self.scale)


def proto_logits(support, query, scale=1.0):
    """support [E, N, K, D], query [E, M, D] -> logits [E, M, N] = -scale * squared Euclidean."""
    protos = support.mean(dim=2)                                  # [E, N, D]
    d2 = ((query.unsqueeze(2) - protos.unsqueeze(1)) ** 2).sum(-1)  # [E, M, N]
    return -scale * d2


def episode_labels(n_way, n_query, n_episodes, device=None):
    """Query labels [E, N*Q] matching the class-major query layout."""
    lab = torch.arange(n_way, device=device).repeat_interleave(n_query)
    return lab.unsqueeze(0).expand(n_episodes, -1)


def set_bn_eval(model):
    """Keep BatchNorm running stats frozen while the rest of the model trains."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt["model_cfg"])
    cfg["pretrained"] = False  # weights come from the checkpoint
    model = ProtoNet(**cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt
