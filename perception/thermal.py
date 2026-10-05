"""
Mission 3, thermal spotter: find warm bodies (people, animals) from the air,
e.g. someone lost in a field at dusk. Reports WHERE, does not follow.

Hardware (not bought yet): FLIR Lepton 3.5 (160 x 120, radiometric) on a
PureThermal USB board. On the Pi it is a USB webcam that sends 16-bit frames
whose pixel values are temperatures in centi-kelvin (radiometric "TLinear" mode,
the PureThermal default).

    LeptonSource   background thread: grabs the newest thermal frame, as deg C
    HeatSpotter    one frame -> warm blobs: warmer than the scene's median by
                   `above_background_c`, inside [min_c, max_c], person-sized
    ThermalWatch   blobs -> ground points (missions/geo.py) -> a spot is reported
                   after `confirm_hits` sightings within `match_m`, once per
                   `cooldown_s`. Status + optional snapshot like the perimeter alerts.

Honest limits: a person 20 m away is a few pixels; warm ground, animals, car
bonnets and sun-heated rocks all look similar. It points you at places to
LOOK, a human decides. Best at night / early morning (cold background).
"""

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np

from missions.geo import ground_point
from missions.mission_config import ThermalConfig


def centikelvin_to_c(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32) / 100.0 - 273.15


@dataclass
class HeatBlob:
    box: tuple                 # (l, t, r, b) pixels
    peak_c: float
    mean_c: float
    area_px: int


class HeatSpotter:
    def __init__(self, config: Optional[ThermalConfig] = None):
        self.cfg = config or ThermalConfig()

    def detect(self, temps_c: np.ndarray) -> List[HeatBlob]:
        import cv2
        c = self.cfg
        t = np.asarray(temps_c, np.float32)
        background = float(np.median(t))
        lo = max(c.min_c, background + c.above_background_c)
        mask = ((t >= lo) & (t <= c.max_c)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < c.min_area_px or area > c.max_area_frac * t.size:
                continue
            vals = t[labels == i]
            blobs.append(HeatBlob((x, y, x + w, y + h), float(vals.max()), float(vals.mean()), area))
        return sorted(blobs, key=lambda b: -b.peak_c)


class LeptonSource:
    """Newest thermal frame in deg C, grabbed on its own thread. `capture` injectable (tests):
    any object with read() -> (ok, frame) and release()."""

    def __init__(self, index: int = 1, capture=None, threaded: bool = True):
        if capture is None:
            import cv2
            capture = cv2.VideoCapture(index)
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
            capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            if not capture.isOpened():
                raise RuntimeError(f"thermal camera: /dev/video{index} did not open")
        self.cap = capture
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.frames = 0
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._run, name="thermal", daemon=True)
            self._thread.start()

    def grab(self) -> bool:
        ok, raw = self.cap.read()
        if not ok or raw is None:
            return False
        raw = np.asarray(raw)
        if raw.dtype != np.uint16:                 # the camera fell back to 8-bit video: no temperatures
            return False
        if raw.ndim == 3:
            raw = raw[..., 0]
        with self._lock:
            self._latest = centikelvin_to_c(raw)
            self.frames += 1
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.grab():
                time.sleep(0.05)

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.cap.release()


@dataclass
class _Spot:
    x: float
    y: float
    hits: int
    last_t: float
    peak_c: float
    reported_t: float = -float("inf")


@dataclass
class Sighting:
    x: float
    y: float
    peak_c: float
    hits: int
    t: float
    wall_time: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(x=round(self.x, 1), y=round(self.y, 1), peak_c=round(self.peak_c, 1))
        return d


class ThermalWatch:
    def __init__(self, source: LeptonSource, config: Optional[ThermalConfig] = None,
                 spotter: Optional[HeatSpotter] = None, alerts_dir: Optional[str] = None):
        self.src = source
        self.cfg = config or ThermalConfig()
        self.spotter = spotter or HeatSpotter(self.cfg)
        self.alerts_dir = alerts_dir
        self._spots: List[_Spot] = []
        self._last = -float("inf")
        self.recent: List[Sighting] = []

    def step(self, obs) -> List[Sighting]:
        c = self.cfg
        if obs.now - self._last < c.every_s:
            return []
        self._last = obs.now
        temps = self.src.latest()
        if temps is None:
            return []
        h, w = temps.shape[:2]
        points = []
        for b in self.spotter.detect(temps):
            u, v = (b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2
            p = ground_point(u, v, w, h, obs.pose, c.pitch_deg, c.hfov_deg, c.aspect, max_range_m=c.max_range_m)
            if p is not None:
                points.append((p, b.peak_c))
        out = self._update(obs.now, points)
        for s in out:
            self.recent.append(s)
            self._save(temps, s)
            print(f"THERMAL: warm spot at ({s.x:.1f}, {s.y:.1f}) m, {s.peak_c:.1f} C")
        del self.recent[:-20]
        return out

    def _update(self, now: float, points) -> List[Sighting]:
        c = self.cfg
        self._spots = [s for s in self._spots if now - s.last_t <= c.forget_s]
        out = []
        for (x, y), peak in points:
            near = min(self._spots, key=lambda s: math.hypot(s.x - x, s.y - y), default=None)
            if near is None or math.hypot(near.x - x, near.y - y) > c.match_m:
                near = _Spot(x, y, 0, now, peak)
                self._spots.append(near)
            k = near.hits
            near.x, near.y = (near.x * k + x) / (k + 1), (near.y * k + y) / (k + 1)
            near.hits += 1
            near.last_t, near.peak_c = now, max(near.peak_c, peak)
            if near.hits >= c.confirm_hits and now - near.reported_t >= c.cooldown_s:
                near.reported_t = now
                out.append(Sighting(near.x, near.y, near.peak_c, near.hits, now))
        return out

    def status(self) -> dict:
        return {"mission": "thermal", "frames": self.src.frames, "candidates": len(self._spots),
                "sightings": [s.to_dict() for s in self.recent[-5:]], "sighting_count": len(self.recent)}

    def _save(self, temps: np.ndarray, s: Sighting) -> None:
        if not self.alerts_dir:
            return
        import cv2
        os.makedirs(self.alerts_dir, exist_ok=True)
        base = os.path.join(self.alerts_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime(s.wall_time)) + "_thermal")
        lo, hi = float(np.percentile(temps, 1)), float(np.percentile(temps, 99)) + 1e-3
        img = cv2.applyColorMap(np.clip((temps - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cv2.imwrite(base + ".jpg", cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
        with open(base + ".json", "w") as f:
            json.dump(s.to_dict(), f)
