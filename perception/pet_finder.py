"""
Mission 2, pet follow: the same follow brain, but the "target" is one enrolled
animal instead of a person.

    main.py ... --mission pet --pet-species dog      (then send a photo of the dog from the phone)

What changes, what doesn't:
- Detector: the YOLOX model you already use for people is a COCO model; COCO
  also has bird, cat, dog, horse, sheep, cow. YoloClassesDetector runs it once
  and keeps the classes asked for. Needs --person-detector yolo (our own
  detector knows only people).
- Identity: animals have no face for the face matcher, so identity = appearance
  (fur colour histogram, perception/body_reid.py ColorHistogramEmbedder). The
  enrolment photo is never forgotten; new looks are learned only from matches
  far above threshold (anti-drift, same idea as TargetFinder).
- Output: the same Detection the tracker / controller already use. A confident
  match is source="face" (= identity confirmed this frame, which is what the
  tracker's lock needs); a weaker one is "track". Distance comes from the
  animal's box height and PetConfig.body_height_m.
- Height: hover_height_m (4 m) replaces the person hover height. Never fly low
  over animals: rotor noise and downwash scare them; some bite drones.
  Wildlife: parks and protected areas restrict drones near animals.

Same interface as FaceMatcher / TargetFinder (has_target, set_target,
clear_target, find, stats), so platforms/real.py doesn't change.
"""

import time
from typing import Any, Callable, List, Optional, Sequence

import cv2
import numpy as np

from datatypes import Detection
from missions.mission_config import COCO_CLASSES, PetConfig
from perception.person_detector import (Box, PersonDetector, Scored, decode_yolo, letterbox,
                                        load_onnx, nms)


class YoloClassesDetector(PersonDetector):
    """YoloPersonDetector for several classes at once (one network pass)."""

    def __init__(self, model_path: str, class_ids: Sequence[int], input_size: int = 640, fmt: str = "yolox",
                 conf: float = 0.35, nms_iou: float = 0.45, rgb: bool = False, scale_01: bool = False,
                 net: Optional[Any] = None):
        self.net = net if net is not None else load_onnx(model_path, "animal detector (the COCO YOLOX model)")
        self.class_ids = list(class_ids)
        self.size, self.fmt, self.conf, self.nms_iou = input_size, fmt, conf, nms_iou
        self.rgb, self.scale_01 = rgb, scale_01
        self.last_classes: List[int] = []

    def detect(self, image: np.ndarray) -> List[Scored]:
        h, w = image.shape[:2]
        padded, scale, px, py = letterbox(image, self.size)
        blob = cv2.dnn.blobFromImage(padded, 1 / 255.0 if self.scale_01 else 1.0,
                                     (self.size, self.size), swapRB=self.rgb)
        self.net.setInput(blob)
        raw = self.net.forward()
        boxes: List[Box] = []
        scores: List[float] = []
        classes: List[int] = []
        for cid in self.class_ids:
            centres, s = decode_yolo(raw, self.fmt, self.size, cid, self.conf)
            for (cx, cy, bw, bh), sc in zip(centres, s):
                l = int(np.clip((cx - bw / 2 - px) / scale, 0, w - 1))
                t = int(np.clip((cy - bh / 2 - py) / scale, 0, h - 1))
                r = int(np.clip((cx + bw / 2 - px) / scale, 0, w - 1))
                b = int(np.clip((cy + bh / 2 - py) / scale, 0, h - 1))
                if r > l and b > t:
                    boxes.append((l, t, r, b))
                    scores.append(float(sc))
                    classes.append(cid)
        keep = nms(boxes, scores, self.nms_iou)
        self.last_classes = [classes[i] for i in keep]
        return [(boxes[i], scores[i]) for i in keep]


class _Gallery:
    """Looks of the enrolled animal: the pinned enrolment look(s) + a rolling set of learned ones."""

    def __init__(self, size: int):
        self.size = size
        self.pinned: List[np.ndarray] = []
        self.learned: List[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.pinned) + len(self.learned)

    def clear(self) -> None:
        self.pinned, self.learned = [], []

    def add(self, emb: np.ndarray, pinned: bool = False) -> None:
        v = np.asarray(emb, np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        if pinned:
            self.pinned.append(v)
        else:
            self.learned.append(v)
            del self.learned[:-self.size]

    def score(self, embs: np.ndarray) -> np.ndarray:
        e = np.asarray(embs, np.float32)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        g = np.stack(self.pinned + self.learned)
        return (e @ g.T).max(axis=1)


class PetFinder:
    def __init__(self, detector: PersonDetector, embedder, config: Optional[PetConfig] = None,
                 face_height_m: float = 0.22, clock: Callable[[], float] = time.monotonic):
        self.detector, self.embedder = detector, embedder
        self.cfg = config or PetConfig()
        self.face_height_m = face_height_m
        self.clock = clock
        c = self.cfg
        self.acquire = c.acquire_threshold or getattr(embedder, "default_acquire", 0.8)
        self.keep = c.keep_threshold or getattr(embedder, "default_keep", 0.7)
        self.gallery = _Gallery(c.gallery_size)
        self._last_hit = -float("inf")
        self._last_learn = -float("inf")
        self.stats: dict = {}

    # ---- FaceMatcher-compatible interface --------------------------------------
    @property
    def has_target(self) -> bool:
        return len(self.gallery) > 0

    def set_target(self, image: np.ndarray) -> int:
        """Enrol from a photo of the animal (the biggest one in the photo). Raises if none is found."""
        found = self.detector.detect(image)
        if not found:
            names = "/".join(self.cfg.species)
            raise ValueError(f"no {names} found in the photo: send a clearer, closer one")
        box = max(found, key=lambda s: (s[0][2] - s[0][0]) * (s[0][3] - s[0][1]))[0]
        self.gallery.clear()
        self.gallery.add(self.embedder.embed(image, [box])[0], pinned=True)
        self._last_hit = self._last_learn = -float("inf")
        return len(found)

    def clear_target(self) -> None:
        self.gallery.clear()

    def find(self, frame: np.ndarray) -> Optional[Detection]:
        if not self.has_target:
            self.stats = {"source": None}
            return None
        now = self.clock()
        c = self.cfg
        found = sorted(self.detector.detect(frame), key=lambda s: -s[1])[:c.max_candidates]
        self.stats = {"source": None, "animals": len(found)}
        if not found:
            return None
        boxes = [b for b, _ in found]
        sims = self.gallery.score(self.embedder.embed(frame, boxes))
        order = np.argsort(-sims)
        best = int(order[0])
        runner_up = float(sims[order[1]]) if len(order) > 1 else -1.0
        fresh = now - self._last_hit <= c.keep_window_s
        threshold = self.keep if fresh else self.acquire
        self.stats.update(best_sim=round(float(sims[best]), 3), threshold=threshold)
        if sims[best] < threshold or runner_up > sims[best] - c.margin:
            return None
        self._last_hit = now
        confirmed = sims[best] >= self.acquire + c.confirm_margin
        if confirmed and now - self._last_learn >= c.learn_every_s:
            self.gallery.add(self.embedder.embed(frame, [boxes[best]])[0])
            self._last_learn = now
        source = "face" if confirmed else "track"
        self.stats["source"] = source
        h, w = frame.shape[:2]
        return pet_detection(boxes[best], w, h, c.body_height_m, self.face_height_m, source)


def pet_detection(box: Box, w: int, h: int, body_height_m: float, face_height_m: float,
                  source: str) -> Detection:
    """The rest of the system sizes things by a FACE box (distance = face_height_m / its
    angular size). Build the face box an animal of this box height would have, at the
    top of the box (its head end); bbox = the whole animal, like TargetFinder's body tracks."""
    l, t, r, b = box
    fh = max(1.0, (b - t) * face_height_m / body_height_m)
    cx, cy = (l + r) / 2, t + fh / 2
    return Detection(bbox=(t, r, b, l), offset_x=(cx - w / 2) / (w / 2), offset_y=-(cy - h / 2) / (h / 2),
                     size=fh / h, source=source)


def make_pet_finder(reid_cfg, pet_cfg: PetConfig, face_height_m: float = 0.22,
                    detector: Optional[PersonDetector] = None, embedder=None) -> PetFinder:
    unknown = [s for s in pet_cfg.species if s not in COCO_CLASSES]
    if unknown:
        raise ValueError(f"unknown species {unknown}: pick from {sorted(COCO_CLASSES)}")
    if detector is None:
        if reid_cfg.detector != "yolo":
            raise ValueError("--mission pet needs --person-detector yolo (a COCO model knows animals)")
        detector = YoloClassesDetector(reid_cfg.yolo_model, [COCO_CLASSES[s] for s in pet_cfg.species],
                                       reid_cfg.yolo_input_size, reid_cfg.yolo_format, pet_cfg.detector_conf,
                                       rgb=reid_cfg.yolo_rgb, scale_01=reid_cfg.yolo_scale_01)
    if embedder is None:
        from perception.body_reid import ColorHistogramEmbedder
        embedder = ColorHistogramEmbedder()
    return PetFinder(detector, embedder, pet_cfg, face_height_m)
