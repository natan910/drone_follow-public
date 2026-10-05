"""
Pseudo-labels: run a big, slow, accurate "teacher" detector over every
recorded frame, offline on the Mac, and save its person boxes as labels.
Humans then only CORRECT boxes (tools/label_review.py) instead of drawing them.

Rules:
  * a frame a human verified is never overwritten;
  * frames already labelled by a teacher are skipped unless redo=True;
  * track ids (same person across frames) are recomputed for the whole session
    every run, from teacher and human boxes alike.

The default teacher is YOLOX-S at 640 px (Apache-2.0, see config.py), the same
file the drone uses today. It never has to run on the Pi, so a bigger teacher
is fine if you ever get one with a licence that allows it.
"""

from typing import Callable, Optional

import cv2

from dataset.labels import link_tracks, teacher_row
from dataset.layout import image_path, read_frames, read_labels, write_labels
from perception.person_detector import PersonDetector

LOW_CONF = 0.25        # teacher keeps boxes down to this; training decides what to trust


def autolabel_session(root: str, sid: str, detector: PersonDetector, by: str,
                      min_score: float = 0.5, redo: bool = False,
                      progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Label one session in place. Returns counts."""
    frames = read_frames(root, sid)
    labels = read_labels(root, sid)
    ran = skipped = missing = 0
    for i, fr in enumerate(frames):
        name = fr["file"]
        old = labels.get(name)
        if old is not None and (old.get("verified") or not redo):
            skipped += 1
            continue
        img = cv2.imread(image_path(root, sid, name))
        if img is None:
            missing += 1
            continue
        labels[name] = teacher_row(name, detector.detect(img), by)
        ran += 1
        if progress is not None and ran % 50 == 0:
            progress(i + 1, len(frames))
    ordered = [labels[f["file"]] for f in frames if f["file"] in labels]
    times = [f.get("t", float(k)) for k, f in enumerate(frames) if f["file"] in labels]
    for row in link_tracks(ordered, times, min_score):
        labels[row["file"]] = row
    write_labels(root, sid, labels)
    return {"session": sid, "labelled": ran, "kept": skipped, "missing_images": missing,
            "frames": len(frames)}


def make_teacher(model_path: str, input_size: int = 640, fmt: str = "yolox",
                 rgb: bool = False, scale_01: bool = False, conf: float = LOW_CONF) -> PersonDetector:
    from perception.person_detector import YoloPersonDetector
    return YoloPersonDetector(model_path, input_size, fmt, conf, rgb=rgb, scale_01=scale_01)
