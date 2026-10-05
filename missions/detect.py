"""
R9: detection off the flight loop, one network pass for people AND objects.

Before: the perimeter ran YOLOX-640 on the main loop, once a second; each run
(~0.3-0.5 s on a Pi 5 CPU) froze the loop, so the drone's commands stuttered.

Now:
  DetectorWorker    runs the detector in its own thread. The main loop hands it
                    a copy of the frame + where the drone was at that instant,
                    and picks the result up on a later step. Never waits. If the
                    worker is still busy, the new frame is skipped (counted).
                    Only the worker thread touches the network (CLAUDE.md
                    threading rule: one owner per model).
  CocoSceneDetector one YOLOX pass -> people + the R7 object labels, NMS per class
                    (a person sitting on a chair keeps both boxes).
  ImageSceneSource  frame -> Scene: people projected to the ground with the pose
                    and camera tilt AT CAPTURE (the drone moved while the
                    detector ran), the owner skipped, objects listed.

threaded=False runs the detector inline (tests, and --detect-sync to debug).
cv2.dnn releases Python's GIL while it computes, so the thread really runs in
parallel with the main loop on the Pi's 4 cores.

Faster still (later, see PATROL.md R9): our own 320 px detector (TRAINING.md),
or a Hailo AI HAT+ on the Pi 5.
"""

import threading
import time
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from datatypes import Observation
from missions.entities import ObjectSighting, PersonSighting, Scene
from missions.geo import foot_of, ground_point
from missions.mission_config import COCO_NAMES, PerimeterConfig

Labeled = Tuple[Tuple[int, int, int, int], float, str]     # (l, t, r, b), score, label


def labeled(dets) -> List[Labeled]:
    """Person detectors return (box, score); CocoSceneDetector (box, score, label)."""
    return [(d[0], float(d[1]), d[2] if len(d) > 2 else "person") for d in dets]


class CocoSceneDetector:
    """People + chosen COCO objects from one YOLOX (or YOLOv5/8) pass. Same model file
    as --person-detector yolo; our own detector knows only people."""

    def __init__(self, reid_cfg, labels: Sequence[str] = (), conf: float = 0.35, nms_iou: float = 0.45,
                 net=None):
        from perception.person_detector import load_onnx
        unknown = [l for l in labels if l not in COCO_NAMES]
        if unknown:
            raise ValueError(f"not COCO labels: {unknown}")
        self.ids = [0] + sorted({COCO_NAMES.index(l) for l in labels} - {0})
        self.net = net if net is not None else load_onnx(reid_cfg.yolo_model, "COCO detector (--person-detector yolo model)")
        self.size, self.fmt = reid_cfg.yolo_input_size, reid_cfg.yolo_format
        self.rgb, self.scale_01 = reid_cfg.yolo_rgb, reid_cfg.yolo_scale_01
        self.conf, self.nms_iou = conf, nms_iou

    def detect(self, image: np.ndarray) -> List[Labeled]:
        from perception.person_detector import decode_yolo, letterbox, nms
        h, w = image.shape[:2]
        padded, scale, px, py = letterbox(image, self.size)
        blob = cv2.dnn.blobFromImage(padded, 1 / 255.0 if self.scale_01 else 1.0, (self.size, self.size),
                                     swapRB=self.rgb)
        self.net.setInput(blob)
        raw = self.net.forward()
        out: List[Labeled] = []
        for cid in self.ids:
            centres, scores = decode_yolo(raw, self.fmt, self.size, cid, self.conf)
            boxes, s = [], []
            for (cx, cy, bw, bh), sc in zip(centres, scores):
                l = int(np.clip((cx - bw / 2 - px) / scale, 0, w - 1))
                t = int(np.clip((cy - bh / 2 - py) / scale, 0, h - 1))
                r = int(np.clip((cx + bw / 2 - px) / scale, 0, w - 1))
                b = int(np.clip((cy + bh / 2 - py) / scale, 0, h - 1))
                if r > l and b > t:
                    boxes.append((l, t, r, b))
                    s.append(float(sc))
            out += [(boxes[i], s[i], COCO_NAMES[cid]) for i in nms(boxes, s, self.nms_iou)]
        return out


class DetectorWorker:
    def __init__(self, detect_fn: Callable[[np.ndarray], list], threaded: bool = True, max_errors: int = 5,
                 clock: Callable[[], float] = time.perf_counter):
        self._fn, self.threaded, self.max_errors, self._clock = detect_fn, threaded, max_errors, clock
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._job = None
        self._result = None
        self._busy = False
        self._stop = False
        self.runs = self.errors = self.skipped = self._consecutive = 0
        self.last_ms: Optional[float] = None
        self.avg_ms: Optional[float] = None
        self.last_error: Optional[str] = None
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._loop, name="detector", daemon=True)
            self._thread.start()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def submit(self, frame: np.ndarray, meta: dict) -> bool:
        with self._lock:
            if self._busy:
                self.skipped += 1
                return False
            self._busy = True
            self._job = (frame, meta)
        if self.threaded:
            self._wake.set()
        else:
            self._run_job()
        return True

    def poll(self):
        """(meta, detections, ms) of a finished run, or None. Raises after max_errors failures in a row."""
        with self._lock:
            r, self._result = self._result, None
            if self._consecutive >= self.max_errors:
                raise RuntimeError(f"detector failed {self._consecutive} times in a row: {self.last_error}")
        return r

    def stats(self) -> dict:
        with self._lock:
            return {"runs": self.runs, "last_ms": None if self.last_ms is None else round(self.last_ms),
                    "avg_ms": None if self.avg_ms is None else round(self.avg_ms), "skipped": self.skipped,
                    "errors": self.errors, "threaded": self.threaded}

    def close(self, timeout_s: float = 2.0) -> None:
        self._stop = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout_s)

    def _loop(self) -> None:
        while not self._stop:
            self._wake.wait(0.5)
            self._wake.clear()
            if self._job is not None:
                self._run_job()

    def _run_job(self) -> None:
        with self._lock:
            job, self._job = self._job, None
        if job is None:
            return
        frame, meta = job
        t0 = self._clock()
        try:
            dets, err = list(self._fn(frame)), None
        except Exception as e:                       # reported through poll(), never kills the thread
            dets, err = None, e
        ms = (self._clock() - t0) * 1000.0
        with self._lock:
            self._busy = False
            if err is not None:
                self.errors += 1
                self._consecutive += 1
                self.last_error = f"{type(err).__name__}: {err}"
                return
            self._consecutive = 0
            self.runs += 1
            self.last_ms = ms
            self.avg_ms = ms if self.avg_ms is None else 0.8 * self.avg_ms + 0.2 * ms
            self._result = (meta, dets, ms)


def covers(person, target) -> bool:
    """person box (l, t, r, b) contains the centre of the target's box."""
    cx, cy = (target[0] + target[2]) / 2, (target[1] + target[3]) / 2
    return person[0] <= cx <= person[2] and person[1] <= cy <= person[3]


def _ltrb(bbox_trbl) -> tuple:
    t, r, b, l = bbox_trbl
    return l, t, r, b


class ImageSceneSource:
    """Camera frames in, Scenes out (at most one per detector run)."""

    def __init__(self, detector, hfov_deg: float, aspect: float, config: Optional[PerimeterConfig] = None,
                 threaded: bool = True):
        self.detector = detector
        self.cfg = config or PerimeterConfig()
        self.hfov, self.aspect = hfov_deg, aspect
        self.worker = DetectorWorker(detector.detect, threaded=threaded)
        self._last_submit = -float("inf")

    def step(self, frame: Optional[np.ndarray], obs: Observation) -> Optional[Scene]:
        scene = self._take()
        c = self.cfg
        if frame is not None and obs.now - self._last_submit >= c.detect_every_s and not self.worker.busy:
            owner = _ltrb(obs.detection.bbox) if (c.ignore_target and obs.detection is not None) else None
            copy = frame.copy()          # the camera reuses its buffer; the overlay is drawn on it later
            meta = {"t": obs.now, "pose": obs.pose, "pitch": obs.camera_pitch_deg, "owner": owner, "frame": copy}
            if self.worker.submit(copy, meta):
                self._last_submit = obs.now
                if not self.worker.threaded:
                    scene = self._take() or scene
        return scene

    def stats(self) -> dict:
        return self.worker.stats()

    def close(self) -> None:
        self.worker.close()

    def _take(self) -> Optional[Scene]:
        r = self.worker.poll()
        return None if r is None else self._scene(*r)

    def _scene(self, meta: dict, dets, ms: float) -> Scene:
        c = self.cfg
        frame = meta["frame"]
        h, w = frame.shape[:2]
        people, objects, far = [], [], 0
        for box, score, label in labeled(dets):
            if label == "person":
                if score < c.min_score:
                    continue
                if meta["owner"] is not None and covers(box, meta["owner"]):
                    continue
                u, v = foot_of(box)
                p = ground_point(u, v, w, h, meta["pose"], meta["pitch"], self.hfov, self.aspect,
                                 min_down_deg=c.min_down_deg, max_range_m=c.max_range_m)
                if p is None:
                    far += 1
                    continue
                people.append(PersonSighting(p[0], p[1], score, box))
            elif score >= c.object_min_score:
                objects.append(ObjectSighting(label, score, box))
        return Scene(meta["t"], meta["pose"], meta["pitch"], people, objects, frame, ms, far)
