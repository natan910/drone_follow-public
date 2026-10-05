"""
Our own networks (PyTorch). Architecture and weights are ours: no pretrained
download, no third-party model code.

    Backbone        MobileNetV2-style inverted residual blocks, plain ops only
                    (conv, batch-norm, ReLU, add): every op cv2.dnn has run for
                    years, so the ONNX export runs on the Pi without surprises.
    CenterNet       backbone + small FPN to stride 4 + three 1x1 heads:
                    person score, centre offset, log box size. One class.
    ReIDNet         backbone (last stage stride 1: finer detail) + average pool +
                    256-d embedding with a "BN neck"; identity classifier for training.

Export wrappers produce exactly what perception/ expects:
    CenterNetExport  RGB 0..255 in -> (1, 5, H/4, W/4) [sigmoid score, dx, dy, log w/4, log h/4]
    ReIDExport       ImageNet-normalised RGB (N, 3, 256, 128) in -> (N, 256) embedding
"""

import copy
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_div(c: float, d: int = 8) -> int:
    return max(d, int(c + d / 2) // d * d)


def conv_bn(cin: int, cout: int, k: int = 3, s: int = 1, groups: int = 1, act: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
                               nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, expand: int):
        super().__init__()
        hid = cin * expand
        self.use_res = stride == 1 and cin == cout
        layers: List[nn.Module] = []
        if expand != 1:
            layers.append(conv_bn(cin, hid, 1))
        layers += [conv_bn(hid, hid, 3, stride, groups=hid), conv_bn(hid, cout, 1, act=False)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


class Backbone(nn.Module):
    """Returns feature maps at strides 4, 8, 16, 32 (or 16 for the last one
    when last_stride=1)."""
    # (expand, channels, repeats, stride); output taken after stages 1, 2, 4, 5
    STAGES = ((1, 16, 1, 1), (4, 24, 2, 2), (4, 40, 3, 2), (4, 80, 3, 2), (4, 112, 2, 1), (4, 160, 2, 2))
    TAPS = (1, 2, 4, 5)

    def __init__(self, width: float = 1.0, last_stride: int = 2):
        super().__init__()
        c = make_div(16 * width)
        self.stem = conv_bn(3, c, 3, 2)
        self.stages = nn.ModuleList()
        self.channels: List[int] = []
        for i, (e, ch, n, s) in enumerate(self.STAGES):
            if i == len(self.STAGES) - 1:
                s = last_stride
            out = make_div(ch * width)
            blocks = []
            for j in range(n):
                blocks.append(InvertedResidual(c, out, s if j == 0 else 1, e))
                c = out
            self.stages.append(nn.Sequential(*blocks))
            if i in self.TAPS:
                self.channels.append(out)

    def forward(self, x) -> List[torch.Tensor]:
        x = self.stem(x)
        feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.TAPS:
                feats.append(x)
        return feats


class InputNorm(nn.BatchNorm2d):
    """(x / 255 - mean) / std as a frozen batch-norm layer: RGB 0..255 in.
    A BatchNormalization node is the most boring op there is for cv2.dnn, and
    it never learns or updates (always eval mode, no gradients)."""

    def __init__(self):
        super().__init__(3, eps=0.0)
        with torch.no_grad():
            self.running_mean.copy_(torch.tensor(IMAGENET_MEAN) * 255.0)
            self.running_var.copy_((torch.tensor(IMAGENET_STD) * 255.0) ** 2)
            self.weight.fill_(1.0)
            self.bias.zero_()
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)

    def train(self, mode: bool = True):
        return super().train(False)


class CenterNet(nn.Module):
    STRIDE = 4

    def __init__(self, width: float = 1.0, neck: int = 64):
        super().__init__()
        self.norm = InputNorm()
        self.backbone = Backbone(width)
        c2, c3, c4, c5 = self.backbone.channels
        self.lat = nn.ModuleList([conv_bn(c, neck, 1) for c in (c2, c3, c4, c5)])
        self.smooth = nn.ModuleList([nn.Sequential(conv_bn(neck, neck, 3, groups=neck), conv_bn(neck, neck, 1))
                                     for _ in range(3)])
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.head = nn.Sequential(conv_bn(neck, neck, 3, groups=neck), conv_bn(neck, neck, 1))
        self.heat = nn.Conv2d(neck, 1, 1)
        self.reg = nn.Conv2d(neck, 2, 1)
        self.wh = nn.Conv2d(neck, 2, 1)
        nn.init.constant_(self.heat.bias, -2.19)      # start at p = 0.1: stable focal loss early on
        nn.init.normal_(self.reg.weight, std=0.001)
        nn.init.constant_(self.reg.bias, 0.5)
        nn.init.normal_(self.wh.weight, std=0.001)
        nn.init.constant_(self.wh.bias, 2.0)          # log(size / 4) = 2  ->  ~30 px boxes

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, 3, S, S) RGB 0..255. Returns raw (heat logit, reg, log wh)."""
        c2, c3, c4, c5 = self.backbone(self.norm(x))
        p = self.lat[3](c5)
        p = self.smooth[0](self.lat[2](c4) + self.up(p))
        p = self.smooth[1](self.lat[1](c3) + self.up(p))
        p = self.smooth[2](self.lat[0](c2) + self.up(p))
        f = self.head(p)
        return self.heat(f), self.reg(f), self.wh(f)


class CenterNetExport(nn.Module):
    def __init__(self, net: CenterNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        heat, reg, wh = self.net(x)
        return torch.cat([torch.sigmoid(heat), reg, wh], dim=1)


class ReIDNet(nn.Module):
    def __init__(self, num_ids: int, width: float = 1.0, dim: int = 256):
        super().__init__()
        self.backbone = Backbone(width, last_stride=1)
        c5 = self.backbone.channels[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c5, dim)
        self.neck = nn.BatchNorm1d(dim)
        self.neck.bias.requires_grad_(False)          # "BNNeck" (Luo et al. 2019)
        self.classifier = nn.Linear(dim, max(1, num_ids), bias=False)

    def features(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """(embedding before the neck, after the neck)."""
        f = self.fc(torch.flatten(self.pool(self.backbone(x)[-1]), 1))
        return f, self.neck(f)

    def forward(self, x):
        f, n = self.features(x)
        return f, self.classifier(n)


def fold_linear_bn(fc: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    """One Linear equal to Linear -> BatchNorm1d (eval mode). No BatchNorm1d
    in the ONNX file: cv2.dnn is happiest with a plain Gemm."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    out = nn.Linear(fc.in_features, fc.out_features)
    with torch.no_grad():
        out.weight.copy_(fc.weight * scale[:, None])
        out.bias.copy_((fc.bias - bn.running_mean) * scale + bn.bias)
    return out


class ReIDExport(nn.Module):
    def __init__(self, net: ReIDNet):
        super().__init__()
        net = copy.deepcopy(net).eval()
        self.backbone, self.pool = net.backbone, net.pool
        self.fc = fold_linear_bn(net.fc, net.neck)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.backbone(x)[-1]), 1))


# ---- losses -----------------------------------------------------------------------

def focal_loss(logit: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """CenterNet's penalty-reduced focal loss on the heatmap."""
    pred = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1.0 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** 2 * pos
    neg_loss = torch.log(1 - pred) * pred ** 2 * (1 - gt) ** 4 * neg
    return -(pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)


def gather_at(fmap: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) at flat indices (B, M) -> (B, M, C)."""
    b, c = fmap.shape[:2]
    flat = fmap.view(b, c, -1).permute(0, 2, 1)                  # B, HW, C
    return flat.gather(1, ind.unsqueeze(-1).expand(-1, -1, c))


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1)
    return (torch.abs(pred - target) * m).sum() / (m.sum() * pred.shape[-1]).clamp(min=1.0)


def detector_loss(outputs, heat, reg, logwh, ind, mask, wh_weight: float = 1.0):
    h, r, w = outputs
    lh = focal_loss(h, heat)
    lr = masked_l1(gather_at(r, ind), reg, mask)
    lw = masked_l1(gather_at(w, ind), logwh, mask)
    return lh + lr + wh_weight * lw, {"heat": float(lh), "reg": float(lr), "wh": float(lw)}


def batch_hard_triplet(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """For each crop: its farthest same-person crop must be closer than its
    nearest stranger by `margin` (cosine distance space)."""
    e = F.normalize(emb, dim=1)
    d = (2.0 - 2.0 * e @ e.t()).clamp(min=1e-12).sqrt()
    same = labels[:, None].eq(labels[None, :]).float()
    hardest_pos = (d * same).max(dim=1).values
    hardest_neg = (d + same * 1e4).min(dim=1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


def reid_loss(feat, logits, labels, smoothing: float = 0.1, margin: float = 0.3):
    ce = F.cross_entropy(logits, labels, label_smoothing=smoothing)
    tri = batch_hard_triplet(feat, labels, margin)
    return ce + tri, {"ce": float(ce), "triplet": float(tri)}


class ModelEMA:
    """Exponential moving average of the weights: the exported model is this
    smoothed copy, usually a bit better than the last raw step."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))   # warm up
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def load_backbone(model: nn.Module, checkpoint: str) -> Sequence[str]:
    """Start a network's backbone from another checkpoint of ours (e.g. re-ID
    from the detector, which already learned what people look like from above).
    Returns the names that were loaded."""
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("model", state)
    own = model.state_dict()
    picked = {k: v for k, v in state.items() if k.startswith("backbone.") and k in own and own[k].shape == v.shape}
    model.load_state_dict(picked, strict=False)
    return sorted(picked)
