"""
Find one specific person's face in a frame.

A FaceMatcher is backend-agnostic: it holds the target's embedding, asks an
Embedder for every face in a frame, and returns the best match above a
threshold. A backend only has to say "here are the faces, with an embedding
each". Adding your own distilled model means writing one small Embedder.

    "insightface"  buffalo_l (SCRFD detector + ResNet50 ArcFace embedder)
                   pip install insightface onnxruntime
    "opencv"       YuNet detector + SFace embedder, pure OpenCV >= 4.8.
                   Download from https://github.com/opencv/opencv_zoo into ./models/:
                     face_detection_yunet_2023mar.onnx
                     face_recognition_sface_2021dec.onnx

Privacy: only the embedding (a vector of numbers) is kept. The enrolment photo
is never stored by this module.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import MatcherConfig
from datatypes import Detection

Box = Tuple[int, int, int, int]  # left, top, right, bottom


@dataclass(frozen=True)
class FaceObservation:
    box: Box
    embedding: np.ndarray  # L2-normalised, so a dot product is cosine similarity


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    return v / max(float(np.linalg.norm(v)), 1e-9)


def make_detection(box: Box, frame_w: int, frame_h: int, source: str = "face") -> Detection:
    """Pixel box -> Detection with normalised 2-D offset and size.
    offset_x: -1 left edge .. +1 right edge. offset_y: -1 bottom .. +1 top
    (image y grows downward, so it is negated to make "up" positive)."""
    left, top, right, bottom = box
    return Detection(bbox=(top, right, bottom, left),
                     offset_x=((left + right) / 2 - frame_w / 2) / (frame_w / 2),
                     offset_y=-((top + bottom) / 2 - frame_h / 2) / (frame_h / 2),
                     size=(bottom - top) / frame_h,
                     source=source)


class Embedder(ABC):
    default_threshold: float

    @abstractmethod
    def faces(self, image: np.ndarray) -> List[FaceObservation]:
        """Every face in a BGR image, each with its embedding."""


class InsightFaceEmbedder(Embedder):
    default_threshold = 0.4  # cosine similarity; raise it if you get false matches

    def __init__(self):
        from insightface.app import FaceAnalysis  # imported lazily: heavy
        # Skip the landmark and gender/age models: we only need detect + embed.
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                                providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def faces(self, image: np.ndarray) -> List[FaceObservation]:
        return [FaceObservation(tuple(int(v) for v in f.bbox), unit(f.normed_embedding))
                for f in self.app.get(image)]


class OpenCVEmbedder(Embedder):
    default_threshold = 0.363  # OpenCV's documented SFace cosine threshold

    def __init__(self, yunet_path: str, sface_path: str):
        self.detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.8, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    def faces(self, image: np.ndarray) -> List[FaceObservation]:
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, rows = self.detector.detect(image)
        if rows is None:
            return []
        out = []
        for r in rows:  # x, y, w, h, 10 landmark coords, score
            feat = self.recognizer.feature(self.recognizer.alignCrop(image, r))
            x, y, bw, bh = (int(v) for v in r[:4])
            out.append(FaceObservation((x, y, x + bw, y + bh), unit(feat)))
        return out


class FaceMatcher:
    def __init__(self, embedder: Embedder, threshold: Optional[float] = None):
        self.embedder = embedder
        self.threshold = embedder.default_threshold if threshold is None else threshold
        self._target: Optional[np.ndarray] = None

    @property
    def has_target(self) -> bool:
        return self._target is not None

    def set_target(self, image: np.ndarray) -> int:
        """Enrol (or replace) the person to look for, from a photo. Uses the
        biggest face if there are several. Returns how many faces the photo had."""
        faces = self.embedder.faces(image)
        if not faces:
            raise ValueError("no face found in that photo")
        biggest = max(faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))
        self._target = biggest.embedding
        return len(faces)

    def clear_target(self) -> None:
        self._target = None

    def find(self, frame: np.ndarray) -> Optional[Detection]:
        """The target's Detection in this frame, or None (also if nobody is enrolled)."""
        if self._target is None:
            return None
        best, best_sim = None, self.threshold
        for face in self.embedder.faces(frame):
            sim = float(np.dot(self._target, face.embedding))
            if sim >= best_sim:
                best, best_sim = face, sim
        if best is None:
            return None
        h, w = frame.shape[:2]
        return make_detection(best.box, w, h)


def make_matcher(cfg: MatcherConfig) -> FaceMatcher:
    if cfg.backend == "insightface":
        embedder: Embedder = InsightFaceEmbedder()
    elif cfg.backend == "opencv":
        embedder = OpenCVEmbedder(cfg.yunet_model, cfg.sface_model)
    else:
        raise ValueError(f"Unknown matcher backend: {cfg.backend!r}")
    matcher = FaceMatcher(embedder, cfg.threshold)
    if cfg.reference_image:
        image = cv2.imread(cfg.reference_image)
        if image is None:
            raise ValueError(f"Could not read reference image: {cfg.reference_image}")
        matcher.set_target(image)
    return matcher
