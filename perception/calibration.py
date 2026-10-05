"""
One-tap optics calibration: measure the camera's real field of view from the
enrolled person's own face at a known distance. Done once, on the ground;
the result is saved and loaded automatically on every later boot.

Why the face: distance is estimated as size_at_1m / apparent_size, where
size_at_1m = face_height_m / (2 * tan(vfov/2)). Calibrating with the target's
own face at a measured distance fixes that whole product -- lens error AND
the fact that nobody's face is exactly the 0.22 m default -- in one go.

Flow: the operator taps "calibrate" (phone page or fleet console) and stands
`distance_m` from the camera, facing it. The next frames where the enrolled
face is confirmed get measured; the median of N samples gives the size
(robust to a blink or a half-turn). The result goes to the autopilot through
the next Observation, and to disk.
"""

import json
import os
import statistics
import time
from typing import List, Optional, Tuple

from datatypes import Detection
from perception.geometry import fov_from_measurement

PLAUSIBLE_HFOV_DEG = (20.0, 150.0)   # outside this, the measurement is wrong, not the lens
DISTANCE_RANGE_M = (0.5, 10.0)


class FovCalibrator:
    IDLE, MEASURING, DONE, FAILED = "idle", "measuring", "done", "failed"

    def __init__(self, face_height_m: float, samples_needed: int = 15, timeout_s: float = 15.0):
        self.face_height_m = face_height_m
        self.samples_needed, self.timeout_s = samples_needed, timeout_s
        self.state = self.IDLE
        self.distance_m: Optional[float] = None
        self.result: Optional[Tuple[float, float]] = None   # (hfov_deg, aspect)
        self.reason = ""
        self._sizes: List[float] = []
        self._aspect = 0.0
        self._started = 0.0

    def start(self, distance_m: float, now: float) -> None:
        lo, hi = DISTANCE_RANGE_M
        if not (lo <= distance_m <= hi):
            raise ValueError(f"stand between {lo:g} and {hi:g} m from the camera")
        self.state, self.distance_m, self._started = self.MEASURING, distance_m, now
        self._sizes, self.result, self.reason = [], None, ""

    def feed(self, detection: Optional[Detection], frame_shape: Optional[Tuple[int, int]],
             now: float) -> Optional[Tuple[float, float]]:
        """Call once per frame. Returns (hfov_deg, aspect) exactly once: on the
        frame that completes a successful measurement. None otherwise."""
        if self.state != self.MEASURING:
            return None
        if now - self._started > self.timeout_s:
            self._fail(f"saw the enrolled face clearly in only {len(self._sizes)} of "
                       f"{self.samples_needed} frames within {self.timeout_s:g} s -- send your "
                       "photo first, face the camera and hold still")
            return None
        if detection is None or detection.source != "face" or not frame_shape:
            return None
        h, w = frame_shape
        self._aspect = h / w
        self._sizes.append(detection.size)
        if len(self._sizes) < self.samples_needed:
            return None

        size = statistics.median(self._sizes)
        hfov, _ = fov_from_measurement(self.distance_m, self.face_height_m, size, self._aspect)
        lo, hi = PLAUSIBLE_HFOV_DEG
        if not (lo <= hfov <= hi):
            self._fail(f"measured {hfov:.0f} deg, which no normal lens has -- check the "
                       f"distance really was {self.distance_m:g} m")
            return None
        self.state, self.result = self.DONE, (hfov, self._aspect)
        return self.result

    def status(self) -> dict:
        s: dict = {"state": self.state}
        if self.state == self.MEASURING:
            s.update(samples=len(self._sizes), needed=self.samples_needed, distance_m=self.distance_m)
        elif self.state == self.DONE and self.result:
            s.update(hfov_deg=round(self.result[0], 1), aspect=round(self.result[1], 4))
        elif self.state == self.FAILED:
            s["reason"] = self.reason
        return s

    def _fail(self, reason: str) -> None:
        self.state, self.reason = self.FAILED, reason


def save_calibration(path: str, hfov_deg: float, aspect: float, **extra) -> None:
    """Atomic write (temp file + rename): a power cut mid-save leaves the old
    file intact rather than a half-written one."""
    data = {"hfov_deg": round(hfov_deg, 2), "aspect": round(aspect, 5),
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra}
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_calibration(path: str) -> Optional[dict]:
    """None if there is no file (never calibrated). Raises ValueError if the
    file exists but is unusable, so the caller can say so rather than
    silently flying on defaults."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        hfov, aspect = float(data["hfov_deg"]), float(data["aspect"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{path} is not a readable calibration file ({e})") from e
    lo, hi = PLAUSIBLE_HFOV_DEG
    if not (lo <= hfov <= hi) or not (0.2 <= aspect <= 2.0):
        raise ValueError(f"{path} holds implausible values (hfov {hfov}, aspect {aspect})")
    return data
