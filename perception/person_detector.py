"""
Find every person in a frame (who they are is body_reid's job, not this one's).

    "hog"   OpenCV's built-in HOG + linear SVM pedestrian detector. Nothing to
            download, works everywhere, but slow-ish and weak on small, sitting
            or top-down people. Good for trying things out.
    "yolo"  Any YOLO-family ONNX model run through cv2.dnn (no PyTorch needed).
            Much better and, at 320 px input, fast enough for a Pi 4/5. Three
            output layouts are understood (set `yolo_format` in config.py):
              "v8"     YOLOv8 / YOLO11 export: (1, 4 + classes, N), no objectness
              "v5"     YOLOv5 / YOLOv7 export: (1, N, 5 + classes), boxes decoded
              "yolox"  YOLOX (e.g. OpenCV Zoo's object_detection_yolox): (1, N, 5 +
                       classes), raw grid offsets decoded here
              "auto"   "v8" if the output is channels-first, otherwise "v5"
            Licences differ: YOLOX and NanoDet are Apache-2.0; Ultralytics
            YOLOv8/11 are AGPL-3.0 (fine for a personal project, read it before
            shipping anything).
    "own"   Our own network (training/models.py, trained on our own footage by
            training/train.py): a CenterNet-style single-class detector. The
            ONNX file takes RGB 0..255 (normalisation is inside the model) and
            outputs one (1, 5, H/4, W/4) map: person score (0..1), x/y offset
            inside the cell, log(box width / 4), log(box height / 4). Decoded
            here (the exp is done in numpy: fewer ONNX ops for cv2.dnn to get wrong).

Every detector returns boxes as (left, top, right, bottom) pixels in the image
it was given, plus a confidence. Detection on a crop is the caller's business:
see detect_in_roi().
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]      # left, top, right, bottom
Scored = Tuple[Box, float]           # (box, confidence)


def load_onnx(path: str, what: str):
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what}: model file {path!r} not found")
    return cv2.dnn.readNet(path)


class PersonDetector(ABC):
    # For detectors whose cost grows with pixels: the person height (px) they
    # work best at. detect_in_roi shrinks a crop so the expected person is about
    # this tall. None = fixed-input network, shrinking buys nothing.
    preferred_person_px: Optional[int] = None

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Scored]:
        """Every person in a BGR image."""


def nms(boxes: Sequence[Box], scores: Sequence[float], iou: float) -> List[int]:
    """Indices to keep after non-maximum suppression."""
    if not boxes:
        return []
    xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    keep = cv2.dnn.NMSBoxes(xywh, [float(s) for s in scores], 0.0, iou)
    return sorted((int(i) for i in np.array(keep).ravel()), key=lambda i: -scores[i])


def detect_in_roi(detector: PersonDetector, image: np.ndarray, roi: Box,
                  expected_person_px: Optional[float] = None) -> List[Scored]:
    """Run the detector on a crop and return boxes in full-frame pixels.
    For fixed-input networks a crop makes a distant person bigger in the
    network's eyes; for pixel-bound detectors (HOG) the crop is also shrunk
    so the person we expect is about `preferred_person_px` tall."""
    h, w = image.shape[:2]
    l, t, r, b = max(0, roi[0]), max(0, roi[1]), min(w, roi[2]), min(h, roi[3])
    if r - l < 16 or b - t < 16:
        return []
    crop = image[t:b, l:r]
    k = 1.0
    want = detector.preferred_person_px
    if want and expected_person_px and expected_person_px > 1.2 * want:
        k = want / expected_person_px
        crop = cv2.resize(crop, (max(1, int((r - l) * k)), max(1, int((b - t) * k))),
                          interpolation=cv2.INTER_AREA)
    return [((int(bl / k) + l, int(bt / k) + t, int(br / k) + l, int(bb / k) + t), s)
            for (bl, bt, br, bb), s in detector.detect(crop)]


# ---- HOG --------------------------------------------------------------------

class HOGPersonDetector(PersonDetector):
    preferred_person_px = 160   # the 64 x 128 window, plus margin for the box

    def __init__(self, max_width: int = 480, hit_threshold: float = 0.0, nms_iou: float = 0.45):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.max_width, self.hit_threshold, self.nms_iou = max_width, hit_threshold, nms_iou

    def detect(self, image: np.ndarray) -> List[Scored]:
        h, w = image.shape[:2]
        scale = min(1.0, self.max_width / w)
        small = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image
        if small.shape[0] < 128 or small.shape[1] < 64:
            return []  # the HOG window is 64 x 128: nothing smaller can be found
        rects, weights = self.hog.detectMultiScale(small, hitThreshold=self.hit_threshold,
                                                   winStride=(8, 8), padding=(8, 8), scale=1.05)
        boxes = [(int(x / scale), int(y / scale), int((x + bw) / scale), int((y + bh) / scale))
                 for x, y, bw, bh in (rects if len(rects) else [])]
        scores = [float(s) for s in np.array(weights).ravel()]
        return [(boxes[i], scores[i]) for i in nms(boxes, scores, self.nms_iou)]


# ---- YOLO (ONNX through cv2.dnn) ---------------------------------------------

def letterbox(image: np.ndarray, size: int) -> Tuple[np.ndarray, float, int, int]:
    """Fit the image into size x size keeping its aspect ratio, padding with grey.
    Returns (padded, scale, pad_x, pad_y) so boxes can be mapped back."""
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    out = np.full((size, size, 3), 114, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, scale, px, py


def _yolox_grid(size: int, strides=(8, 16, 32)) -> Tuple[np.ndarray, np.ndarray]:
    grids, mults = [], []
    for s in strides:
        n = size // s
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        grids.append(np.stack((xv, yv), -1).reshape(-1, 2))
        mults.append(np.full((n * n, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(mults).astype(np.float32)


def decode_yolo(raw: np.ndarray, fmt: str, input_size: int, class_id: int = 0,
                conf: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    """Raw network output -> (boxes as N x 4 [cx, cy, w, h] in network-input
    pixels, scores N) for one class, already thresholded. Pure numpy."""
    out = np.asarray(raw, np.float32)
    out = out.reshape(out.shape[-2], out.shape[-1]) if out.ndim == 3 else out
    if fmt == "auto":
        fmt = "v8" if out.shape[0] < out.shape[1] else "v5"
    if fmt == "v8":
        out = out.T                                  # -> N x (4 + classes)
        boxes, scores = out[:, :4], out[:, 4 + class_id]
    elif fmt in ("v5", "yolox"):
        boxes = out[:, :4].copy()
        scores = out[:, 4] * out[:, 5 + class_id]
        if fmt == "yolox":
            grid, stride = _yolox_grid(input_size)
            if len(grid) != len(boxes):
                raise ValueError(f"YOLOX output has {len(boxes)} rows, expected {len(grid)} for "
                                 f"input {input_size}: check yolo_input_size")
            boxes[:, :2] = (boxes[:, :2] + grid) * stride
            boxes[:, 2:4] = np.exp(boxes[:, 2:4]) * stride
    else:
        raise ValueError(f"unknown yolo_format {fmt!r}")
    keep = scores >= conf
    return boxes[keep], scores[keep]


class YoloPersonDetector(PersonDetector):
    def __init__(self, model_path: str, input_size: int = 320, fmt: str = "auto",
                 conf: float = 0.4, nms_iou: float = 0.45, class_id: int = 0,
                 rgb: bool = True, scale_01: bool = True, net: Optional[Any] = None):
        """`net` is injectable for tests; normally the ONNX file is loaded here.
        YOLOv5/v8/11 exports want RGB scaled to 0..1 (the defaults); OpenCV
        Zoo's YOLOX wants BGR 0..255 (rgb=False, scale_01=False)."""
        self.net = net if net is not None else load_onnx(model_path, "person detector (see README: body re-ID)")
        self.size, self.fmt, self.conf, self.nms_iou = input_size, fmt, conf, nms_iou
        self.class_id, self.rgb, self.scale_01 = class_id, rgb, scale_01

    def detect(self, image: np.ndarray) -> List[Scored]:
        h, w = image.shape[:2]
        padded, scale, px, py = letterbox(image, self.size)
        blob = cv2.dnn.blobFromImage(padded, 1 / 255.0 if self.scale_01 else 1.0,
                                     (self.size, self.size), swapRB=self.rgb)
        self.net.setInput(blob)
        raw = self.net.forward()
        centres, scores = decode_yolo(raw, self.fmt, self.size, self.class_id, self.conf)
        boxes: List[Box] = []
        for cx, cy, bw, bh in centres:
            l = int(np.clip((cx - bw / 2 - px) / scale, 0, w - 1))
            t = int(np.clip((cy - bh / 2 - py) / scale, 0, h - 1))
            r = int(np.clip((cx + bw / 2 - px) / scale, 0, w - 1))
            b = int(np.clip((cy + bh / 2 - py) / scale, 0, h - 1))
            boxes.append((l, t, r, b))
        s = [float(v) for v in scores]
        return [(boxes[i], s[i]) for i in nms(boxes, s, self.nms_iou) if boxes[i][2] > boxes[i][0]
                and boxes[i][3] > boxes[i][1]]


# ---- our own detector (CenterNet-style, ONNX through cv2.dnn) -------------------

CENTER_STRIDE = 4


def decode_center(raw: np.ndarray, conf: float = 0.4, top_k: int = 50,
                  stride: int = CENTER_STRIDE) -> Tuple[np.ndarray, np.ndarray]:
    """(1, 5, h, w) output -> (boxes N x 4 [cx, cy, w, h] in network-input pixels,
    scores N). A cell is a detection if its score is a local maximum (3 x 3) and
    >= conf: the "NMS" of this family. Pure numpy + cv2.dilate."""
    out = np.asarray(raw, np.float32)
    if out.ndim == 4:
        out = out[0]
    if out.shape[0] != 5:
        raise ValueError(f"own detector output should have 5 channels, got shape {raw.shape}")
    heat = np.ascontiguousarray(out[0])
    peaks = (heat >= conf) & (heat >= cv2.dilate(heat, np.ones((3, 3), np.uint8)))
    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    scores = heat[ys, xs]
    order = np.argsort(-scores)[:top_k]
    ys, xs, scores = ys[order], xs[order], scores[order]
    cx = (xs + out[1, ys, xs]) * stride
    cy = (ys + out[2, ys, xs]) * stride
    bw = np.exp(np.clip(out[3, ys, xs], -10, 10)) * stride
    bh = np.exp(np.clip(out[4, ys, xs], -10, 10)) * stride
    boxes = np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)
    return boxes, scores.astype(np.float32)


class CenterPersonDetector(PersonDetector):
    """Our own detector. Same letterbox as YOLO; RGB 0..255 in (the model
    normalises), so there is nothing to get wrong about scaling."""

    def __init__(self, model_path: str, input_size: int = 320, conf: float = 0.4,
                 nms_iou: float = 0.6, net: Optional[Any] = None):
        self.net = net if net is not None else load_onnx(model_path, "own person detector (see TRAINING.md)")
        self.size, self.conf, self.nms_iou = input_size, conf, nms_iou

    def detect(self, image: np.ndarray) -> List[Scored]:
        h, w = image.shape[:2]
        padded, scale, px, py = letterbox(image, self.size)
        blob = cv2.dnn.blobFromImage(padded, 1.0, (self.size, self.size), swapRB=True)
        self.net.setInput(blob)
        centres, scores = decode_center(self.net.forward(), self.conf)
        boxes: List[Box] = []
        for cx, cy, bw, bh in centres:
            boxes.append((int(np.clip((cx - bw / 2 - px) / scale, 0, w - 1)),
                          int(np.clip((cy - bh / 2 - py) / scale, 0, h - 1)),
                          int(np.clip((cx + bw / 2 - px) / scale, 0, w - 1)),
                          int(np.clip((cy + bh / 2 - py) / scale, 0, h - 1))))
        s = [float(v) for v in scores]
        # peaks already de-duplicate; a light NMS catches two peaks on one tall person
        return [(boxes[i], s[i]) for i in nms(boxes, s, self.nms_iou)
                if boxes[i][2] > boxes[i][0] and boxes[i][3] > boxes[i][1]]
