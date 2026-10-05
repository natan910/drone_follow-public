"""
Building blocks for recognising the target by their whole body.

    AppearanceEmbedder   crop of a person -> unit vector; similar looks -> high dot product
        ColorHistogramEmbedder  built in. Clothing colour, torso and legs separately.
                                Viewpoint-proof (a shirt is the same colour from
                                behind) but fooled by two people dressed alike.
        OnnxReidEmbedder        a person re-identification network (OpenCV Zoo's
                                YouTu re-ID, or an OSNet export from torchreid,
                                or your own distilled student) through cv2.dnn.
        FusedEmbedder           weighted mix of several; the dot product of the
                                fused vectors IS the weighted mean of the parts'.
    AppearanceGallery    a small bank of looks; score = best match against any
    FaceBodyModel        where the face sits inside a body box, learned online,
                         so a body box can stand in for the face in the geometry
    MotionGate           constant-velocity prediction of where the target should
                         be next, so a lookalike across the frame can't win
"""

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from perception.person_detector import Box, load_onnx


def unit_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)


def clip_box(box: Box, w: int, h: int) -> Box:
    l, t, r, b = box
    return max(0, l), max(0, t), min(w, r), min(h, b)


def iou(a: Box, b: Box) -> float:
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---- embedders ----------------------------------------------------------------

class AppearanceEmbedder(ABC):
    # Thresholds on the cosine similarity, calibrated per embedder.
    default_acquire: float
    default_keep: float

    @abstractmethod
    def embed(self, image: np.ndarray, boxes: Sequence[Box]) -> np.ndarray:
        """One L2-normalised row per box (len(boxes) x D)."""


class ColorHistogramEmbedder(AppearanceEmbedder):
    """Hellinger-normalised colour histograms of the torso and the legs.

    Coloured pixels vote for (hue, saturation) bins; greys, blacks and whites
    (where hue is meaningless) vote for a brightness bin instead, so a black
    jacket and a white one are told apart. The crop is trimmed at the sides and
    the head/hair is skipped, to keep background and hair out of it. The dot
    product of two of these is the Bhattacharyya coefficient per stripe."""

    default_acquire = 0.80
    default_keep = 0.70

    H_BINS, S_BINS, V_BINS = 12, 3, 6
    STRIPES = ((0.18, 0.55), (0.55, 1.0))  # torso, legs (fractions of box height)
    SIDE_TRIM = 0.18
    SIZE = (24, 48)                         # work on a tiny copy: plenty for colour

    def embed(self, image: np.ndarray, boxes: Sequence[Box]) -> np.ndarray:
        h, w = image.shape[:2]
        dim = len(self.STRIPES) * (self.H_BINS * self.S_BINS + self.V_BINS)
        out = np.zeros((len(boxes), dim), np.float32)
        for i, box in enumerate(boxes):
            l, t, r, b = clip_box(box, w, h)
            trim = int((r - l) * self.SIDE_TRIM)
            crop = image[t:b, l + trim:r - trim]
            if crop.shape[0] < 8 or crop.shape[1] < 4:
                continue
            hsv = cv2.cvtColor(cv2.resize(crop, self.SIZE, interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2HSV)
            out[i] = np.concatenate([self._stripe(hsv[int(a * self.SIZE[1]):int(z * self.SIZE[1])])
                                     for a, z in self.STRIPES])
        return unit_rows(out)

    def _stripe(self, hsv: np.ndarray) -> np.ndarray:
        hch, sch, vch = (hsv[..., k].ravel().astype(np.int32) for k in range(3))
        chroma = (sch > 50) & (vch > 50)
        hb = hch * self.H_BINS // 180
        sb = np.minimum((sch - 50) * self.S_BINS // 206, self.S_BINS - 1)
        idx = np.where(chroma, hb * self.S_BINS + sb,
                       self.H_BINS * self.S_BINS + vch * self.V_BINS // 256)
        hist = np.bincount(idx, minlength=self.H_BINS * self.S_BINS + self.V_BINS).astype(np.float32)
        return np.sqrt(hist / max(hist.sum(), 1.0))  # Hellinger: dot product = Bhattacharyya


class OnnxReidEmbedder(AppearanceEmbedder):
    """A person re-ID network. Defaults match both OpenCV Zoo's YouTu re-ID and
    torchreid's OSNet: 128 x 256 RGB input, ImageNet mean/std. Crops are
    embedded in one batch per frame."""

    default_acquire = 0.65
    default_keep = 0.55

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_path: str, input_w: int = 128, input_h: int = 256, net=None):
        self.net = net if net is not None else load_onnx(model_path, "re-ID network (see README: body re-ID)")
        self.size = (input_w, input_h)

    def embed(self, image: np.ndarray, boxes: Sequence[Box]) -> np.ndarray:
        if not boxes:
            return np.zeros((0, 1), np.float32)
        h, w = image.shape[:2]
        crops = []
        for box in boxes:
            l, t, r, b = clip_box(box, w, h)
            crop = image[t:max(b, t + 2), l:max(r, l + 2)]
            rgb = cv2.cvtColor(cv2.resize(crop, self.size), cv2.COLOR_BGR2RGB).astype(np.float32)
            crops.append((rgb / 255.0 - self.MEAN) / self.STD)
        blob = np.ascontiguousarray(np.stack(crops).transpose(0, 3, 1, 2))  # N x 3 x H x W
        self.net.setInput(blob)
        return unit_rows(np.asarray(self.net.forward()).reshape(len(boxes), -1))


class FusedEmbedder(AppearanceEmbedder):
    """Concatenates sqrt(weight) * each part, so that
    dot(fused_a, fused_b) = sum(w_i * cos_i) / sum(w_i). Thresholds combine the same way."""

    def __init__(self, parts: Sequence[Tuple[AppearanceEmbedder, float]]):
        total = sum(w for _, w in parts)
        self.parts = [(e, w / total) for e, w in parts]
        self.default_acquire = sum(e.default_acquire * w for e, w in self.parts)
        self.default_keep = sum(e.default_keep * w for e, w in self.parts)

    def embed(self, image: np.ndarray, boxes: Sequence[Box]) -> np.ndarray:
        if not boxes:
            return np.zeros((0, 1), np.float32)
        return np.concatenate([math.sqrt(w) * e.embed(image, boxes) for e, w in self.parts], axis=1)


# ---- gallery ------------------------------------------------------------------

class AppearanceGallery:
    """A few distinct looks of one person (front, back, jacket open, ...).
    A new look close to a stored one refines it (running mean); otherwise it is
    added, evicting the look that has gone longest without being refreshed.
    With `ttl_s`, looks also expire."""

    def __init__(self, capacity: int = 12, merge_above: float = 0.93,
                 ttl_s: Optional[float] = None, blend: float = 0.2):
        self.capacity, self.merge_above, self.ttl_s, self.blend = capacity, merge_above, ttl_s, blend
        self._bank: Optional[np.ndarray] = None
        self._times: List[float] = []

    def __len__(self) -> int:
        return 0 if self._bank is None else len(self._bank)

    def clear(self) -> None:
        self._bank, self._times = None, []

    def add(self, v: np.ndarray, now: float) -> None:
        v = unit_rows(v.reshape(1, -1))
        if self._bank is None or self._bank.shape[1] != v.shape[1]:
            self._bank, self._times = v, [now]
            return
        sims = self._bank @ v[0]
        j = int(np.argmax(sims))
        if sims[j] >= self.merge_above:
            self._bank[j] = unit_rows((1 - self.blend) * self._bank[j] + self.blend * v[0])
            self._times[j] = now
        elif len(self._bank) < self.capacity:
            self._bank = np.vstack([self._bank, v])
            self._times.append(now)
        else:
            k = int(np.argmin(self._times))
            self._bank[k], self._times[k] = v[0], now

    def expire(self, now: float) -> None:
        if self.ttl_s is None or self._bank is None:
            return
        keep = [i for i, t in enumerate(self._times) if now - t <= self.ttl_s]
        if not keep:
            self.clear()
        elif len(keep) < len(self._times):
            self._bank = self._bank[keep]
            self._times = [self._times[i] for i in keep]

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Best similarity to any stored look, per row. -1 when empty."""
        n = len(embeddings)
        if self._bank is None or n == 0 or embeddings.shape[1] != self._bank.shape[1]:
            return np.full(n, -1.0, np.float32)
        return (embeddings @ self._bank.T).max(axis=1)


# ---- face <-> body geometry -------------------------------------------------------

class FaceBodyModel:
    """Where the face box sits inside a person box, as fractions of that box:
    face centre (fx, fy), face height / body height (rh), face height / body
    width (rw). Starts from standing-adult averages and is refined (moving
    average) every time both are seen together, so it adapts to this camera's
    tilt and this detector's box habits. Lets a body box stand in for the face
    in the distance/height geometry (perception/geometry.py)."""

    def __init__(self, fx: float = 0.5, fy: float = 0.07, rh: float = 0.11, rw: float = 0.37,
                 alpha: float = 0.15, edge_px: int = 3):
        self.fx, self.fy, self.rh, self.rw = fx, fy, rh, rw
        self.alpha, self.edge = alpha, edge_px

    def _cut(self, box: Box, w: int, h: int) -> Tuple[bool, bool, bool, bool]:
        l, t, r, b = box
        e = self.edge
        return l <= e, t <= e, r >= w - 1 - e, b >= h - 1 - e  # left, top, right, bottom cut off

    def learn(self, face: Box, body: Box, w: int, h: int) -> None:
        cl, ct, cr, cb = self._cut(body, w, h)
        bw, bh = body[2] - body[0], body[3] - body[1]
        fh = face[3] - face[1]
        if bw <= 0 or bh <= 0 or fh <= 0:
            return
        a = self.alpha
        ease = lambda old, new: (1 - a) * old + a * new
        if not (cl or cr):
            self.fx = ease(self.fx, ((face[0] + face[2]) / 2 - body[0]) / bw)
            self.rw = ease(self.rw, fh / bw)
        if not (ct or cb):
            self.fy = ease(self.fy, ((face[1] + face[3]) / 2 - body[1]) / bh)
            self.rh = ease(self.rh, fh / bh)

    def face_box(self, body: Box, w: int, h: int) -> Box:
        """Estimated face box for a person box, trusting whichever of the box's
        height or width isn't cut off by the frame edge."""
        cl, ct, cr, cb = self._cut(body, w, h)
        l, t, r, b = body
        bw, bh = r - l, b - t
        if ct or cb:
            if not (cl or cr):
                fh = self.rw * bw
            else:
                fh = self.rh * bh      # cut both ways (very close): best effort
            full_h = fh / self.rh
            top = b - full_h if (ct and not cb) else t
        else:
            fh, full_h, top = self.rh * bh, bh, t
        cx = l + self.fx * bw
        cy = top + self.fy * full_h
        fw = 0.8 * fh
        return (int(cx - fw / 2), int(cy - fh / 2), int(cx + fw / 2), int(cy + fh / 2))


# ---- motion gate ----------------------------------------------------------------

class MotionGate:
    """Constant-velocity prediction of the target's face-equivalent position in
    normalised image coordinates (x, y in 0..1, size = face height / frame
    height). Distances are measured in body heights, so the gate means the same
    thing near and far."""

    def __init__(self, rh: float = 0.11, max_age_s: float = 4.0, smoothing: float = 0.5):
        self.rh, self.max_age_s, self.k = rh, max_age_s, smoothing
        self.reset()

    def reset(self) -> None:
        self._state: Optional[Tuple[float, float, float]] = None
        self._vel = (0.0, 0.0)
        self._t: Optional[float] = None

    @property
    def last_time(self) -> Optional[float]:
        return self._t

    def update(self, x: float, y: float, size: float, now: float) -> None:
        if self._state is not None and self._t is not None and 0 < now - self._t <= self.max_age_s:
            dt = now - self._t
            vx, vy = (x - self._state[0]) / dt, (y - self._state[1]) / dt
            self._vel = (self.k * vx + (1 - self.k) * self._vel[0],
                         self.k * vy + (1 - self.k) * self._vel[1])
        else:
            self._vel = (0.0, 0.0)
        self._state, self._t = (x, y, size), now

    def predict(self, now: float) -> Optional[Tuple[float, float, float]]:
        if self._state is None or self._t is None or now - self._t > self.max_age_s:
            return None
        dt = now - self._t
        x, y, s = self._state
        return x + self._vel[0] * dt, y + self._vel[1] * dt, s

    def age(self, now: float) -> float:
        return math.inf if self._t is None else now - self._t

    def distance(self, x: float, y: float, now: float) -> Optional[float]:
        """How far (x, y) is from the prediction, in body heights; None = no prediction."""
        p = self.predict(now)
        if p is None:
            return None
        body_h = max(p[2] / self.rh, 0.05)
        return math.hypot(x - p[0], y - p[1]) / body_h
