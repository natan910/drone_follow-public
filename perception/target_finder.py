"""
Face first, body second: find the enrolled person in a frame even when their
face isn't visible (turned around, seen from above, too far for the face model).

    TargetFinder.find(frame)
        face matched?  -> return it (source="face"). Every `enroll_every_n`
                          such frames, also find the body the face belongs to
                          and LEARN its look (long-term gallery) and where the
                          face sits in it (FaceBodyModel).
        no face?       -> detect people (in a crop around where the target is
                          predicted to be, then the whole frame if needed),
                          keep those the motion gate allows, embed at most
                          `max_candidates` of them, and accept the best if it
                          (a) is similar enough to a known look,
                          (b) clearly beats the runner-up (never guess between
                              two similar people), and
                          (c) is where the target could plausibly be.
                          Returned as source="track" with the face position
                          estimated from the body box, so the rest of the
                          system (tracker, geometry, controller) is unchanged.

Anti-drift: the long-term gallery is only ever taught by face-confirmed frames.
Confident body-only matches teach a short-term gallery whose looks expire after
`short_term_ttl_s` (this is what carries the lock through a turn-around, when
the back looks different from the front). And tracking/target_tracker.py stops
trusting "track" detections `track_only_timeout_s` after the last face anyway.

Same interface as FaceMatcher (has_target / set_target / clear_target / find),
so platforms/real.py doesn't care which one it has.
"""

import time
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config import BodyReIDConfig
from datatypes import Detection
from perception.body_reid import (AppearanceEmbedder, AppearanceGallery, FaceBodyModel,
                                  MotionGate, iou)
from perception.face_matcher import FaceMatcher, make_detection
from perception.person_detector import Box, PersonDetector, Scored, detect_in_roi


def face_box_of(det: Detection) -> Box:
    top, right, bottom, left = det.bbox
    return left, top, right, bottom


class TargetFinder:
    def __init__(self, face: FaceMatcher, detector: PersonDetector, embedder: AppearanceEmbedder,
                 config: Optional[BodyReIDConfig] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.face, self.detector, self.embedder = face, detector, embedder
        self.cfg = config or BodyReIDConfig()
        self.clock = clock
        c = self.cfg
        self.acquire = c.acquire_threshold if c.acquire_threshold is not None else embedder.default_acquire
        self.keep = c.keep_threshold if c.keep_threshold is not None else embedder.default_keep
        self.long_term = AppearanceGallery(c.gallery_size, c.merge_above)
        self.short_term = AppearanceGallery(c.short_term_size, c.merge_above, ttl_s=c.short_term_ttl_s)
        self.geometry = FaceBodyModel()
        self.motion = MotionGate(self.geometry.rh, c.motion_max_age_s)
        self._since_enroll = c.enroll_every_n  # learn on the very first face frame
        self.stats: Dict[str, object] = {}

    # ---- FaceMatcher-compatible interface --------------------------------------
    @property
    def has_target(self) -> bool:
        return self.face.has_target

    def set_target(self, image: np.ndarray) -> int:
        """Enrol from a photo. If the photo shows their body too (a full-length
        shot), the body gallery starts with it: they can be picked up from behind
        before the drone has ever seen their face live."""
        faces = self.face.set_target(image)   # raises if there is no face: nothing changes
        self._forget_body()
        det = self.face.find(image)           # the enrolled face, located in the photo
        if det is not None:
            self._learn(image, det, self.clock())
        return faces

    def clear_target(self) -> None:
        self.face.clear_target()
        self._forget_body()

    def find(self, frame: np.ndarray) -> Optional[Detection]:
        now = self.clock()
        det = self.face.find(frame)
        if det is not None:
            self._note_face(frame, det, now)
            return det
        if not self.face.has_target or len(self.long_term) + len(self.short_term) == 0:
            self.stats = {"source": None}
            return None
        return self._find_body(frame, now)

    # ---- face visible: learn -----------------------------------------------
    def _note_face(self, frame: np.ndarray, det: Detection, now: float) -> None:
        h, w = frame.shape[:2]
        l, t, r, b = face_box_of(det)
        self.motion.update((l + r) / 2 / h, (t + b) / 2 / h, (b - t) / h, now)
        self._since_enroll += 1
        learned = False
        if self._since_enroll >= self.cfg.enroll_every_n:
            learned = self._learn(frame, det, now)
        self.stats = {"source": "face", "learned": learned,
                      "looks": len(self.long_term), "short_looks": len(self.short_term)}

    def _learn(self, frame: np.ndarray, det: Detection, now: float) -> bool:
        """Find the body this face belongs to and remember what it looks like.
        Skipped (returns False) whenever the answer is ambiguous or the body is
        partly hidden behind someone else: a polluted gallery is worse than a
        thin one."""
        h, w = frame.shape[:2]
        self._since_enroll = 0                # success or not, next try in enroll_every_n frames
        face = face_box_of(det)
        people = [b for b, _ in self.detector.detect(frame)]
        owner = self._owner(face, people)
        if owner is None:
            return False
        if any(iou(owner, other) > 0.3 for other in people if other is not owner):
            return False
        self.geometry.learn(face, owner, w, h)
        self.long_term.add(self.embedder.embed(frame, [owner])[0], now)
        return True

    @staticmethod
    def _owner(face: Box, people: List[Box]) -> Optional[Box]:
        """The one person box whose head region holds this face, or None if
        there is none or more than one."""
        fx, fy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
        holders = [p for p in people
                   if p[0] <= fx <= p[2] and p[1] - (face[3] - face[1]) <= fy <= p[1] + 0.45 * (p[3] - p[1])]
        return holders[0] if len(holders) == 1 else None

    def _forget_body(self) -> None:
        self.long_term.clear()
        self.short_term.clear()
        self.geometry = FaceBodyModel()
        self.motion.reset()
        self._since_enroll = self.cfg.enroll_every_n

    # ---- no face: recognise the body ------------------------------------------
    def _find_body(self, frame: np.ndarray, now: float) -> Optional[Detection]:
        self.short_term.expire(now)
        h, w = frame.shape[:2]
        roi = self._roi(now, w, h)
        pick = None
        if roi is not None:
            p = self.motion.predict(now)
            expected = p[2] / self.geometry.rh * h if p is not None else None   # body height, px
            pick = self._choose(frame, detect_in_roi(self.detector, frame, roi, expected), now)
        if pick is None and (roi is None or not self._roi_covers_most(roi, w, h)):
            pick = self._choose(frame, self.detector.detect(frame), now)
        if pick is None:
            return None

        body, sim, embedding = pick
        face = self.geometry.face_box(body, w, h)
        self.motion.update((face[0] + face[2]) / 2 / h, (face[1] + face[3]) / 2 / h,
                           (face[3] - face[1]) / h, now)
        threshold = self.keep if self._fresh(now) else self.acquire
        if sim >= threshold + self.cfg.self_update_margin:
            self.short_term.add(embedding, now)
        l, t, r, b = body
        return replace(make_detection(face, w, h, source="track"), bbox=(t, r, b, l))

    def _fresh(self, now: float) -> bool:
        return self.motion.age(now) <= self.cfg.keep_window_s

    def _roi(self, now: float, w: int, h: int) -> Optional[Box]:
        """A square crop around where the target is predicted to be."""
        p = self.motion.predict(now)
        if p is None:
            return None
        x, y, s = p
        body_h = s / self.geometry.rh * h
        half = self.cfg.roi_scale * body_h / 2 + self.cfg.gate_growth_per_s * self.motion.age(now) * body_h
        cx, cy = x * h, y * h + body_h * (0.5 - self.geometry.fy)   # centre on the body, not the face
        return int(cx - half), int(cy - half), int(cx + half), int(cy + half)

    @staticmethod
    def _roi_covers_most(roi: Box, w: int, h: int) -> bool:
        l, t, r, b = max(0, roi[0]), max(0, roi[1]), min(w, roi[2]), min(h, roi[3])
        return (r - l) * (b - t) >= 0.8 * w * h

    def _choose(self, frame: np.ndarray, found: List[Scored], now: float
                ) -> Optional[Tuple[Box, float, np.ndarray]]:
        h, w = frame.shape[:2]
        c = self.cfg
        fresh = self._fresh(now)
        allowed = c.gate_body_heights + c.gate_growth_per_s * min(self.motion.age(now), c.motion_max_age_s)

        ranked: List[Tuple[float, Box]] = []
        for body, conf in found:
            face = self.geometry.face_box(body, w, h)
            d = self.motion.distance((face[0] + face[2]) / 2 / h, (face[1] + face[3]) / 2 / h, now)
            if d is None:
                ranked.append((-conf, body))          # no prediction: most confident first
            elif d <= allowed:
                ranked.append((d, body))              # closest to the prediction first
        ranked.sort(key=lambda x: x[0])
        boxes = [b for _, b in ranked[:c.max_candidates]]
        self.stats = {"source": None, "people": len(found), "gated": len(ranked), "embedded": len(boxes)}
        if not boxes:
            return None

        embeddings = self.embedder.embed(frame, boxes)
        sims = np.maximum(self.long_term.score(embeddings), self.short_term.score(embeddings))
        order = np.argsort(-sims)
        best = int(order[0])
        runner_up = float(sims[order[1]]) if len(order) > 1 else -1.0
        threshold = self.keep if fresh else self.acquire
        self.stats.update(best_sim=round(float(sims[best]), 3), threshold=threshold)
        if sims[best] < threshold or runner_up > sims[best] - c.margin:
            return None
        self.stats["source"] = "track"
        return boxes[best], float(sims[best]), embeddings[best]


# ---- construction -------------------------------------------------------------

def make_person_detector(c: BodyReIDConfig) -> PersonDetector:
    from perception.person_detector import CenterPersonDetector, HOGPersonDetector, YoloPersonDetector
    if c.detector == "hog":
        return HOGPersonDetector()
    if c.detector == "own":
        return CenterPersonDetector(c.own_model, c.own_input_size, c.detector_conf)
    if c.detector == "yolo":
        return YoloPersonDetector(c.yolo_model, c.yolo_input_size, c.yolo_format, c.detector_conf,
                                  rgb=c.yolo_rgb, scale_01=c.yolo_scale_01)
    raise ValueError(f"unknown person detector {c.detector!r}")


def make_embedder(c: BodyReIDConfig) -> AppearanceEmbedder:
    from perception.body_reid import ColorHistogramEmbedder, FusedEmbedder, OnnxReidEmbedder
    if c.embedder == "color":
        return ColorHistogramEmbedder()
    deep = OnnxReidEmbedder(c.reid_model, c.reid_input_w, c.reid_input_h)
    if c.embedder == "onnx":
        return deep
    if c.embedder == "fused":
        return FusedEmbedder([(deep, c.fused_deep_weight), (ColorHistogramEmbedder(), 1 - c.fused_deep_weight)])
    raise ValueError(f"unknown embedder {c.embedder!r}")


def make_finder(matcher_cfg, reid_cfg: BodyReIDConfig):
    """The face matcher alone, or wrapped in a TargetFinder when body re-ID is
    on. The reference photo (if any) is enrolled through whichever it is, so a
    full-length photo also seeds the body gallery."""
    import cv2
    from dataclasses import replace as dc_replace
    from perception.face_matcher import make_matcher
    face = make_matcher(dc_replace(matcher_cfg, reference_image=None))
    finder = (TargetFinder(face, make_person_detector(reid_cfg), make_embedder(reid_cfg), reid_cfg)
              if reid_cfg.enabled else face)
    if matcher_cfg.reference_image:
        image = cv2.imread(matcher_cfg.reference_image)
        if image is None:
            raise ValueError(f"Could not read reference image: {matcher_cfg.reference_image}")
        finder.set_target(image)
    return finder
