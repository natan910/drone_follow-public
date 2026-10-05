"""
Camera geometry: turns a 2-D image position (offset_x, offset_y, size) into a
3-D position of the target relative to the drone, and back.

Pinhole model. The camera can be tilted down by `pitch_deg` (0 = looking
level, 90 = straight down), so the SAME image position means a different real
position at different tilts — that's why the gimbal angle is an input.

Body frame returned: forward, right, up (metres, drone to the middle of the
detected face).  Distance comes from apparent size:
    size = face_height_m / (2 * depth * tan(vfov/2))
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from config import CameraConfig
from datatypes import TargetEstimate


def fov_from_measurement(distance_m: float, real_height_m: float, size_frac: float,
                         aspect: float) -> Tuple[float, float]:
    """Inverse of CameraModel's distance estimate: something `real_height_m`
    tall, `distance_m` away, fills `size_frac` of the frame height. Solve
    `size = real_height_m / (2 * depth * tan(vfov/2))` for the field of view.
    Returns (hfov_deg, vfov_deg); aspect is frame height / width."""
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    if real_height_m <= 0:
        raise ValueError("real_height_m must be positive")
    if not (0 < size_frac <= 1.0):
        raise ValueError("size_frac must be in (0, 1] -- it's a fraction of the frame height")
    if aspect <= 0:
        raise ValueError("aspect must be positive")
    tan_v = real_height_m / (2 * distance_m * size_frac)
    tan_h = tan_v / aspect
    return 2 * math.degrees(math.atan(tan_h)), 2 * math.degrees(math.atan(tan_v))


@dataclass(frozen=True)
class RelativePosition:
    forward: float
    right: float
    up: float

    @property
    def horizontal(self) -> float:
        return math.hypot(self.forward, self.right)


class CameraModel:
    def __init__(self, config: Optional[CameraConfig] = None):
        self.cfg = config or CameraConfig()
        self.tan_h = math.tan(math.radians(self.cfg.hfov_deg) / 2)
        self.tan_v = self.tan_h * self.cfg.aspect
        self.size_at_1m = self.cfg.face_height_m / (2 * self.tan_v)  # face height / frame height at 1 m
        self.face_to_head_top = self.cfg.face_to_head_top_m

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.tan_v))

    def locate(self, t: TargetEstimate, pitch_deg: float) -> RelativePosition:
        """Where the target (face centre) is, relative to the drone, given
        what the camera sees and how far it is tilted down."""
        depth = self.size_at_1m / max(t.size, 1e-3)   # distance along the optical axis
        x_c = t.offset_x * self.tan_h * depth          # right of the axis, in the image plane
        y_c = t.offset_y * self.tan_v * depth           # above the axis, in the image plane
        th = math.radians(pitch_deg)
        # Rotate the (depth-along-axis, y_c-in-image-plane) pair by the tilt to get
        # (forward, up) in the drone's level body frame. Sideways (right) is unaffected by tilt.
        return RelativePosition(forward=depth * math.cos(th) + y_c * math.sin(th),
                                right=x_c,
                                up=-depth * math.sin(th) + y_c * math.cos(th))

    def project(self, rel: RelativePosition, pitch_deg: float,
                clip: bool = True) -> Optional[TargetEstimate]:
        """Inverse of locate(): what the camera would see. None if the point is
        behind the camera or (with clip) falls outside the field of view."""
        th = math.radians(pitch_deg)
        depth = rel.forward * math.cos(th) - rel.up * math.sin(th)
        if depth < 0.05:
            return None
        y_c = rel.forward * math.sin(th) + rel.up * math.cos(th)
        ox, oy = rel.right / (depth * self.tan_h), y_c / (depth * self.tan_v)
        if clip and (abs(ox) >= 1.0 or abs(oy) >= 1.0):
            return None
        return TargetEstimate(ox, oy, self.size_at_1m / depth)

    def head_top_offset(self) -> float:
        """How much higher than the face centre the top of the head is, in metres."""
        return self.face_to_head_top
