"""
Save raw camera frames plus what the drone knew at that moment, for training
our own detector and re-ID models later.

    rec = Recorder("datasets", SessionInfo(new_session_id("subject-01"), subject="subject-01"))
    ...every loop:  rec.maybe_record(frame, obs, decision)
    rec.close()

Costs the main loop one frame copy per saved frame: JPEG encoding and disk
writes happen on a background thread. If the disk is slow the queue fills and
frames are DROPPED (counted), never waited for: recording must not slow flight.

Saves at most `fps` frames per second, stops for good at `max_gb` or when the
disk has less than `min_free_gb` left (a full SD card can crash the Pi).
Frames are saved before anything is drawn on them (call it before the overlay).
"""

import json
import os
import queue
import shutil
import threading
import time
from dataclasses import asdict
from typing import Callable, Optional

import cv2
import numpy as np

from dataset.layout import SessionInfo, frames_path, write_session


def frame_meta(obs=None, decision=None) -> dict:
    """What we log with each frame. Everything optional: a standalone recording
    (tools/record.py) has no autopilot, so no pose or mode."""
    m: dict = {}
    if obs is not None:
        p = obs.pose
        m.update(x=round(p.x, 2), y=round(p.y, 2), alt_m=round(p.z, 2),
                 yaw_deg=round(float(np.degrees(p.yaw)), 1),
                 pitch_deg=round(obs.camera_pitch_deg, 1))
        if obs.detection is not None:
            top, right, bottom, left = obs.detection.bbox
            m["live_box"] = [left, top, right, bottom]       # what the drone itself followed
            m["live_source"] = obs.detection.source           # "face" or "track"
    if decision is not None:
        m["mode"] = decision.mode.name
    return m


class Recorder:
    def __init__(self, root: str, info: SessionInfo, fps: float = 4.0, max_gb: float = 20.0,
                 min_free_gb: float = 2.0, jpeg_quality: int = 90, threaded: bool = True,
                 clock: Callable[[], float] = time.monotonic,
                 free_bytes: Optional[Callable[[str], int]] = None,
                 queue_size: int = 8):
        self.root, self.info = root, info
        self.dir = write_session(root, info)
        self.interval = 1.0 / fps if fps > 0 else 0.0
        self.max_bytes = int(max_gb * 1e9)
        self.min_free = int(min_free_gb * 1e9)
        self.quality = jpeg_quality
        self.clock = clock
        self.free_bytes = free_bytes or (lambda path: shutil.disk_usage(path).free)
        self.frames = self.bytes = self.dropped = 0
        self.stopped: Optional[str] = None           # why recording stopped, if it did
        self._next_index = 0
        self._next_t: Optional[float] = None
        self._t0 = clock()
        self._lock = threading.Lock()
        self._q: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        if threaded:
            self._q = queue.Queue(maxsize=queue_size)
            self._thread = threading.Thread(target=self._drain, name="recorder", daemon=True)
            self._thread.start()

    # ---- main loop side ------------------------------------------------------
    def maybe_record(self, frame: Optional[np.ndarray], obs=None, decision=None) -> bool:
        """Queue this frame if it's time for the next one. True = queued."""
        if frame is None or self.stopped:
            return False
        now = self.clock()
        if self._next_t is not None and now < self._next_t - 1e-6:
            return False
        # keep a steady average rate even when the loop is not a multiple of it
        # (10 Hz loop, 4 fps recording -> 4 fps, not 3.3)
        base = self._next_t if self._next_t is not None and now - self._next_t < self.interval else now
        self._next_t = base + self.interval
        if self._next_index % 50 == 0 and self.free_bytes(self.dir) < self.min_free:
            self._stop(f"less than {self.min_free / 1e9:.1f} GB free on disk")
            return False
        if not self.info.width:
            self._set_size(frame)
        idx = self._next_index
        self._next_index += 1
        item = (idx, frame.copy(), round(now - self._t0, 3), frame_meta(obs, decision))
        if self._q is None:
            self._write(*item)
            return True
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            with self._lock:
                self.dropped += 1
            return False

    def close(self) -> None:
        if self._q is not None and self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=10)
            self._q = None

    def status(self) -> dict:
        with self._lock:
            return {"session": self.info.session_id, "frames": self.frames,
                    "mb": round(self.bytes / 1e6, 1), "dropped": self.dropped, "stopped": self.stopped}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- writer side -------------------------------------------------------
    def _set_size(self, frame: np.ndarray) -> None:
        self.info.height, self.info.width = frame.shape[:2]
        with open(os.path.join(self.dir, "session.json"), "w") as f:
            json.dump(asdict(self.info), f, indent=2)

    def _stop(self, reason: str) -> None:
        if not self.stopped:
            self.stopped = reason
            print(f"Recorder: stopped ({reason}). Flight is unaffected.")

    def _drain(self) -> None:
        assert self._q is not None
        while True:
            item = self._q.get()
            if item is None:
                return
            try:
                self._write(*item)
            except OSError as e:          # disk gone / full: stop recording, never crash the loop
                self._stop(f"write failed: {e}")

    def _write(self, idx: int, frame: np.ndarray, t: float, meta: dict) -> None:
        if self.stopped:
            return
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        name = f"{idx:06d}.jpg"
        with open(os.path.join(self.dir, "frames", name), "wb") as f:
            f.write(jpg.tobytes())
        with open(frames_path(self.root, self.info.session_id), "a") as f:
            f.write(json.dumps({"file": name, "t": t, **meta}) + "\n")
        with self._lock:
            self.frames += 1
            self.bytes += len(jpg)
            over = self.bytes >= self.max_bytes
        if over:
            self._stop(f"session reached {self.max_bytes / 1e9:.0f} GB")
