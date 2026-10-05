"""
How fast is the target walking, over the ground? Used as a feed-forward: the
follow controller alone is proportional, so it trails a walking person by
speed / horizontal_kp (0.5 m at a slow stroll), which is enough to push them
out of a steeply tilted camera's view. Adding their own velocity to the
command removes that lag.

World position of the target = drone position + the camera's relative
estimate rotated by the drone's heading. Differentiated between SIGHTINGS
only (not frames where the tracker is coasting on a held estimate, which
would read the drone's own motion as the target's), smoothed, capped, and
faded out if sightings stop.
"""

import math
from typing import Optional, Tuple

from datatypes import Pose
from perception.geometry import RelativePosition


class TargetVelocity:
    def __init__(self, smoothing: float = 0.25, max_speed: float = 1.5,
                 deadband: float = 0.08, max_gap_s: float = 0.6, fade_s: float = 2.0):
        self.k, self.max_speed, self.deadband = smoothing, max_speed, deadband
        self.max_gap_s, self.fade_s = max_gap_s, fade_s
        self.reset()

    def reset(self) -> None:
        self._last: Optional[Tuple[float, float, float]] = None  # x, y, t of last sighting
        self._v = (0.0, 0.0)

    @staticmethod
    def world(pose: Pose, rel: RelativePosition) -> Tuple[float, float]:
        s, c = math.sin(pose.yaw), math.cos(pose.yaw)
        # forward = (sin yaw, cos yaw) in (east, north); right = (cos yaw, -sin yaw)
        return pose.x + rel.forward * s + rel.right * c, pose.y + rel.forward * c - rel.right * s

    def sighting(self, pose: Pose, rel: RelativePosition, now: float) -> None:
        x, y = self.world(pose, rel)
        if self._last is not None:
            lx, ly, lt = self._last
            dt = now - lt
            if 0.02 < dt <= self.max_gap_s:
                vx, vy = (x - lx) / dt, (y - ly) / dt
                self._v = (self.k * vx + (1 - self.k) * self._v[0], self.k * vy + (1 - self.k) * self._v[1])
            elif dt > self.max_gap_s:
                self._v = (0.0, 0.0)
        self._last = (x, y, now)

    def velocity(self, now: float) -> Tuple[float, float]:
        """Smoothed (east, north) m/s; fades to zero when sightings stop."""
        if self._last is None:
            return 0.0, 0.0
        age = now - self._last[2]
        fade = max(0.0, 1.0 - max(0.0, age - self.max_gap_s) / self.fade_s)
        vx, vy = self._v[0] * fade, self._v[1] * fade
        speed = math.hypot(vx, vy)
        if speed < self.deadband:
            return 0.0, 0.0
        if speed > self.max_speed:
            vx, vy = vx * self.max_speed / speed, vy * self.max_speed / speed
        return vx, vy

    def body_frame(self, pose: Pose, now: float) -> Tuple[float, float]:
        """The same velocity as (forward, right) for this heading."""
        vx, vy = self.velocity(now)
        s, c = math.sin(pose.yaw), math.cos(pose.yaw)
        return vx * s + vy * c, vx * c - vy * s
