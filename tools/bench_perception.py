"""
How long does each perception stage take on THIS machine? Run it on the Pi
before flying, and whenever you change detector, embedder or input size.

    python tools/bench_perception.py                          # built-ins, synthetic frames
    python tools/bench_perception.py --image crowd.jpg        # a real photo with people in it
    python tools/bench_perception.py --detector yolo --yolo-model models/yolov8n.onnx \\
        --embedder fused --reid-model models/person_reid_youtu_2021nov.onnx

Reports milliseconds per call (median and 90th percentile) for: person
detection on the full frame, detection on a typical tracking crop, and
embedding 1 and 4 people. The per-frame cost of body tracking is roughly
crop-detect + embed(1..4); a full-frame pass only happens when the crop misses.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BodyReIDConfig  # noqa: E402
from perception.person_detector import detect_in_roi  # noqa: E402
from perception.target_finder import make_embedder, make_person_detector  # noqa: E402


def timed(fn, repeats):
    fn()  # warm-up (first inference allocates)
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1000)
    return np.median(samples), np.percentile(samples, 90)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--detector", default="hog", choices=["hog", "yolo"])
    p.add_argument("--yolo-model")
    p.add_argument("--yolo-format", default="auto")
    p.add_argument("--yolo-size", type=int, default=320)
    p.add_argument("--embedder", default="color", choices=["color", "onnx", "fused"])
    p.add_argument("--reid-model")
    p.add_argument("--repeats", type=int, default=30)
    a = p.parse_args()

    cfg = BodyReIDConfig(detector=a.detector, embedder=a.embedder, yolo_format=a.yolo_format,
                         yolo_input_size=a.yolo_size)
    if a.yolo_model:
        cfg.yolo_model = a.yolo_model
    if a.reid_model:
        cfg.reid_model = a.reid_model

    if a.image:
        frame = cv2.imread(a.image)
        if frame is None:
            sys.exit(f"cannot read {a.image}")
        frame = cv2.resize(frame, (a.width, int(frame.shape[0] * a.width / frame.shape[1])))
    else:
        frame = np.random.default_rng(0).integers(0, 255, (a.width * 3 // 4, a.width, 3), np.uint8)
    h, w = frame.shape[:2]
    person = (w // 2 - h // 8, h // 4, w // 2 + h // 8, h - 10)
    crop = (w // 2 - h // 3, 0, w // 2 + h // 3, h)

    detector, embedder = make_person_detector(cfg), make_embedder(cfg)
    print(f"cv2 {cv2.__version__}, {cv2.getNumThreads()} threads, frame {w}x{h}, "
          f"detector={a.detector}, embedder={a.embedder}")
    rows = [
        ("detect: full frame", lambda: detector.detect(frame)),
        ("detect: tracking crop", lambda: detect_in_roi(detector, frame, crop, person[3] - person[1])),
        ("embed: 1 person", lambda: embedder.embed(frame, [person])),
        ("embed: 4 people", lambda: embedder.embed(frame, [person] * 4)),
    ]
    for name, fn in rows:
        med, p90 = timed(fn, a.repeats)
        print(f"  {name:24s} {med:7.1f} ms  (p90 {p90:.1f})")


if __name__ == "__main__":
    main()
