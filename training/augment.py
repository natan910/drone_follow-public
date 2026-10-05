"""
Image augmentation for training (numpy + cv2 only, tested without PyTorch).

Detector: random zoom / shift / flip / colour / motion blur, then the same
square input the drone uses. Eval uses perception's own letterbox(), so the
benchmark and the flight code see identical pixels.
Re-ID: crop the person, resize to 128 x 256, flip, shift, random erasing.
"""

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from perception.person_detector import letterbox

PAD_GREY = 114
REID_W, REID_H = 128, 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def letterbox_boxes(image: np.ndarray, boxes: Sequence[Sequence[float]], size: int
                    ) -> Tuple[np.ndarray, List[List[float]]]:
    out, scale, px, py = letterbox(image, size)
    return out, [[l * scale + px, t * scale + py, r * scale + px, b * scale + py] for l, t, r, b in boxes]


def random_affine(image: np.ndarray, boxes: Sequence[Sequence[float]], size: int,
                  rng: np.random.Generator, zoom: Tuple[float, float] = (0.5, 1.5),
                  min_visible: float = 0.35, min_px: float = 4.0
                  ) -> Tuple[np.ndarray, List[List[float]]]:
    """Scale the image around a random point into a size x size canvas.
    zoom 1.0 = the letterbox fit; > 1 zooms in (people bigger, some cut off),
    < 1 zooms out (people smaller: what the drone sees from higher up).
    Boxes cut to less than min_visible of their area are dropped."""
    h, w = image.shape[:2]
    s = size / max(w, h) * rng.uniform(*zoom)
    nw, nh = w * s, h * s
    # any placement that keeps some of the image on the canvas
    tx = rng.uniform(min(0, size - nw), max(0, size - nw))
    ty = rng.uniform(min(0, size - nh), max(0, size - nh))
    m = np.array([[s, 0, tx], [0, s, ty]], np.float32)
    out = cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(PAD_GREY,) * 3)
    kept = []
    for l, t, r, b in boxes:
        L, T, R, B = l * s + tx, t * s + ty, r * s + tx, b * s + ty
        area = (R - L) * (B - T)
        cl, ct, cr, cb = max(0, L), max(0, T), min(size, R), min(size, B)
        if cr - cl < min_px or cb - ct < min_px or area <= 0:
            continue
        if (cr - cl) * (cb - ct) / area < min_visible:
            continue
        kept.append([cl, ct, cr, cb])
    return out, kept


def hflip(image: np.ndarray, boxes: Sequence[Sequence[float]]) -> Tuple[np.ndarray, List[List[float]]]:
    w = image.shape[1]
    return image[:, ::-1].copy(), [[w - r, t, w - l, b] for l, t, r, b in boxes]


def colour_jitter(image: np.ndarray, rng: np.random.Generator, hue: float = 5,
                  sat: Tuple[float, float] = (0.7, 1.3), val: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hue, hue)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(*sat), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(*val), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def motion_blur(image: np.ndarray, rng: np.random.Generator, max_len: int = 9) -> np.ndarray:
    """A drone yawing or bumping in wind smears the frame in one direction."""
    k = int(rng.integers(3, max_len + 1))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def detector_sample(image: np.ndarray, boxes: Sequence[Sequence[float]], size: int,
                    rng: Optional[np.random.Generator]) -> Tuple[np.ndarray, List[List[float]]]:
    """BGR image + boxes -> (size x size RGB uint8, boxes in its pixels).
    rng None = eval: plain letterbox, exactly like flight."""
    if rng is None:
        img, bx = letterbox_boxes(image, boxes, size)
    else:
        img, bx = random_affine(image, boxes, size, rng)
        if rng.random() < 0.5:
            img, bx = hflip(img, bx)
        if rng.random() < 0.8:
            img = colour_jitter(img, rng)
        if rng.random() < 0.2:
            img = motion_blur(img, rng)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), bx


# ---- re-ID ----------------------------------------------------------------------

def crop_person(image: np.ndarray, box: Sequence[float], pad: float = 0.0,
                max_h: int = 320) -> np.ndarray:
    """The person box (pad = extra margin; 0 matches flight, where
    OnnxReidEmbedder crops the exact box), capped in size for the crop cache."""
    h, w = image.shape[:2]
    l, t, r, b = box
    pw, ph = (r - l) * pad, (b - t) * pad
    L, T = int(max(0, l - pw)), int(max(0, t - ph))
    R, B = int(min(w, r + pw)), int(min(h, b + ph))
    crop = image[T:max(B, T + 2), L:max(R, L + 2)]
    if crop.shape[0] > max_h:
        k = max_h / crop.shape[0]
        crop = cv2.resize(crop, (max(2, int(crop.shape[1] * k)), max_h), interpolation=cv2.INTER_AREA)
    return crop


def reid_sample(crop: np.ndarray, rng: Optional[np.random.Generator]) -> np.ndarray:
    """BGR crop -> (3, 256, 128) float32, ImageNet-normalised RGB: exactly what
    perception.body_reid.OnnxReidEmbedder feeds the network at flight time."""
    img = cv2.resize(crop, (REID_W, REID_H), interpolation=cv2.INTER_LINEAR)
    if rng is not None:
        if rng.random() < 0.5:
            img = img[:, ::-1]
        pad = 10
        big = cv2.copyMakeBorder(np.ascontiguousarray(img), pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                 value=(0, 0, 0))
        x, y = int(rng.integers(0, 2 * pad + 1)), int(rng.integers(0, 2 * pad + 1))
        img = big[y:y + REID_H, x:x + REID_W]
        if rng.random() < 0.5:
            # colour matters for re-ID (the shirt): brightness only, no hue shift
            img = colour_jitter(img, rng, hue=0, sat=(0.9, 1.1), val=(0.75, 1.25))
        if rng.random() < 0.5:
            img = random_erase(img, rng)
    rgb = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)


def random_erase(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blank a random rectangle: teaches the net that a person half-hidden by a
    tree or another person is still the same person."""
    h, w = img.shape[:2]
    for _ in range(10):
        area = rng.uniform(0.02, 0.3) * h * w
        aspect = rng.uniform(0.3, 3.3)
        eh, ew = int(round(np.sqrt(area * aspect))), int(round(np.sqrt(area / aspect)))
        if 0 < eh < h and 0 < ew < w:
            y, x = int(rng.integers(0, h - eh)), int(rng.integers(0, w - ew))
            img = img.copy()
            img[y:y + eh, x:x + ew] = rng.integers(0, 256, (eh, ew, 3), dtype=np.uint8)
            return img
    return img
