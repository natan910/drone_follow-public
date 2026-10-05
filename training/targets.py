"""
What the detector is trained to output, built from boxes (numpy only, so it is
tested without PyTorch). The exact inverse lives in
perception/person_detector.py: decode_center(). A test checks the round trip.

For each person, at the output cell holding the centre of their box:
    heat   1.0 at that cell, a Gaussian bump around it (so near-misses cost less)
    reg    where exactly the centre is inside the cell (0..1, 0..1)
    logwh  log(box width / stride), log(box height / stride)
"""

import math
from typing import Sequence, Tuple

import numpy as np

STRIDE = 4
MAX_OBJECTS = 64


def gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> float:
    """CenterNet's rule: how far the centre can move and still give a box with
    IoU >= min_overlap. Smallest of three cases (see the CornerNet paper)."""
    a1, b1 = 1, h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2 = 4, 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(heat: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """Max-blend a Gaussian of this radius into heat (H x W) at (cx, cy), in place."""
    d = 2 * radius + 1
    sigma = d / 6
    ys, xs = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma)).astype(np.float32)
    h, w = heat.shape
    l, r = min(cx, radius), min(w - cx, radius + 1)
    t, b = min(cy, radius), min(h - cy, radius + 1)
    if l + r <= 0 or t + b <= 0:
        return
    patch = heat[cy - t:cy + b, cx - l:cx + r]
    np.maximum(patch, g[radius - t:radius + b, radius - l:radius + r], out=patch)


def encode_center(boxes: Sequence[Sequence[float]], input_size: int, stride: int = STRIDE,
                  max_objects: int = MAX_OBJECTS
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Boxes (l, t, r, b) in network-input pixels -> training targets:
        heat  (1, H, W) float32
        reg   (M, 2)    float32   sub-cell centre offset
        logwh (M, 2)    float32   log(size / stride)
        ind   (M,)      int64     flat index y * W + x of the centre cell
        mask  (M,)      float32   1 for real objects, 0 for padding
    with H = W = input_size // stride and M = max_objects."""
    n = input_size // stride
    heat = np.zeros((1, n, n), np.float32)
    reg = np.zeros((max_objects, 2), np.float32)
    logwh = np.zeros((max_objects, 2), np.float32)
    ind = np.zeros(max_objects, np.int64)
    mask = np.zeros(max_objects, np.float32)
    # biggest first: if two centres land in one cell, the smaller person wins it
    order = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    k = 0
    for l, t, r, b in order:
        bw, bh = (r - l) / stride, (b - t) / stride
        if bw <= 0 or bh <= 0 or k >= max_objects:
            continue
        cx, cy = (l + r) / 2 / stride, (t + b) / 2 / stride
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < n and 0 <= iy < n):
            continue
        draw_gaussian(heat[0], ix, iy, max(0, int(gaussian_radius(bh, bw))))
        reg[k] = (cx - ix, cy - iy)
        logwh[k] = (math.log(bw), math.log(bh))
        ind[k] = iy * n + ix
        mask[k] = 1.0
        k += 1
    return heat, reg, logwh, ind, mask


def as_network_output(heat: np.ndarray, reg: np.ndarray, logwh: np.ndarray, ind: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
    """Targets laid out exactly like the exported model's output (1, 5, H, W):
    used by the round-trip test, and handy for eyeballing targets."""
    n = heat.shape[-1]
    out = np.zeros((1, 5, n, n), np.float32)
    out[0, 0] = heat[0]
    for k in np.nonzero(mask)[0]:
        y, x = divmod(int(ind[k]), n)
        out[0, 1, y, x], out[0, 2, y, x] = reg[k]
        out[0, 3, y, x], out[0, 4, y, x] = logwh[k]
    return out
