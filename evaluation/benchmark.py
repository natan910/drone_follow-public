"""
Run detectors / re-ID embedders over the frozen eval split and score them.
Uses the SAME perception classes the drone flies with (person_detector.py,
body_reid.py), so a model is judged exactly as it will behave in the air.
"""

import os
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from evaluation.metrics import average_precision, recall_by_size, reid_scores, suggest_thresholds
from perception.body_reid import AppearanceEmbedder
from perception.person_detector import PersonDetector

ALT_BINS = ((None, 3.0, "alt<3m"), (3.0, 6.0, "alt3-6m"), (6.0, 10.0, "alt6-10m"), (10.0, None, "alt>10m"))
LOW_CONF = 0.05          # detectors run this permissive for the full precision/recall curve


def bench_detector(detector: PersonDetector, items: Sequence[dict], root: str,
                   flight_conf: float = 0.4, loader: Callable[[str], Optional[np.ndarray]] = cv2.imread
                   ) -> Dict[str, object]:
    frames, by_alt, ms = [], defaultdict(list), []
    for it in items:
        img = loader(os.path.join(root, it["image"]))
        if img is None:
            continue
        t = time.perf_counter()
        pred = detector.detect(img)
        ms.append((time.perf_counter() - t) * 1000)
        pair = (pred, it["boxes"])
        frames.append(pair)
        alt = it.get("alt_m")
        if alt is not None:
            for lo, hi, name in ALT_BINS:
                if (lo is None or alt >= lo) and (hi is None or alt < hi):
                    by_alt[name].append(pair)
    at_flight = [([p for p in pred if p[1] >= flight_conf], gt) for pred, gt in frames]
    out: Dict[str, object] = {
        "frames": len(frames),
        "ap50": average_precision(frames, 0.5)["ap"],
        "ap75": average_precision(frames, 0.75)["ap"],
        "at_flight_conf": {"conf": flight_conf, **average_precision(at_flight, 0.5)},
        "recall_by_size": recall_by_size(at_flight),
        "ap50_by_altitude": {k: {"ap50": average_precision(v, 0.5)["ap"], "frames": len(v)}
                             for k, v in sorted(by_alt.items())},
        "ms_median": round(float(np.median(ms)), 1) if ms else None,
    }
    return out


def bench_reid(embedder: AppearanceEmbedder, items: Sequence[dict], root: str,
               loader: Callable[[str], Optional[np.ndarray]] = cv2.imread) -> Dict[str, object]:
    by_image: Dict[str, List[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_image[it["image"]].append(i)
    rows: Dict[int, np.ndarray] = {}
    ms = []
    for image, idx in by_image.items():
        img = loader(os.path.join(root, image))
        if img is None:
            continue
        t = time.perf_counter()
        e = embedder.embed(img, [tuple(items[i]["box"]) for i in idx])
        ms.append((time.perf_counter() - t) * 1000 / len(idx))
        for i, v in zip(idx, e):
            rows[i] = v
    keep = sorted(rows)
    if not keep:
        return {"crops": 0}
    emb = np.stack([rows[i] for i in keep])
    ids = [items[i]["id"] for i in keep]
    groups = [items[i]["group"] for i in keep]
    sessions = [items[i]["session"] for i in keep]
    return {"crops": len(keep), "ids": len(set(ids)),
            "same_session": reid_scores(emb, ids, groups, sessions),
            "cross_session": reid_scores(emb, ids, groups, sessions, cross_session=True),
            "thresholds": suggest_thresholds(emb, ids, groups),
            "ms_per_crop": round(float(np.median(ms)), 2) if ms else None}


def headline(kind: str, result: Dict[str, object]) -> float:
    """The one number the gate compares. Detector: AP50. Re-ID: mAP across
    sessions when there are such queries (the hard, useful case), else within."""
    if kind == "detector":
        return float(result.get("ap50", float("nan")))
    cross = result.get("cross_session", {})
    if isinstance(cross, dict) and cross.get("queries"):
        return float(cross["map"])
    same = result.get("same_session", {})
    return float(same.get("map", float("nan"))) if isinstance(same, dict) else float("nan")
