"""
Vision-servo behaviour: turns a TrackerOutput into the body-frame command we
would LIKE to send, plus a gimbal aim request, in order to approach the
target, arrive `hover_height_above_target_m` above the top of their head, and
then keep station there as they move (Task.FOLLOW), or hand off to a fixed
position hold once arrived (Task.HOVER — see autonomy/autopilot.py).

Geometry, not image heuristics: perception/geometry.CameraModel turns the 2-D
image position (offset_x, offset_y, size) plus the camera's current tilt into
a 3-D position of the target relative to the drone (forward, right, up), and
every control decision is made on that. Because DriveCommand is body-frame,
the drone's yaw never needs to be known here — commanding `right_mps` toward
`rel.right` converges regardless of which way the nose points.

Yaw keeps the target inside the camera's cone (a forward camera with only a
tilting gimbal has no side-to-side pan). Close in, the bearing becomes jumpy
(directly overhead it is undefined), so the yaw command fades out in
proportion to distance inside `yaw_release_m` instead of switching off: tiny
wobbles overhead cause no spinning, but a person walking off BEHIND the drone,
out of the forward-looking camera's view, still gets it turning after them.

This class is stateless. Speed/acceleration limits and obstacle/ground
clearance are applied afterwards, uniformly, to whatever produced the command.
"""

import math
from dataclasses import dataclass
from typing import Optional

from config import CameraConfig, ControlConfig
from datatypes import DriveCommand, TargetEstimate, TrackerOutput, TrackState
from perception.geometry import CameraModel, RelativePosition


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class FollowOutput:
    cmd: DriveCommand
    camera_pitch_deg: float   # where the gimbal should aim this step
    height_above_target_m: Optional[float]  # None if the target's position isn't known
    horizontal_m: Optional[float] = None    # horizontal distance to the target, None if unknown
    rel: Optional[RelativePosition] = None  # where the target is relative to the drone, if known


class FollowController:
    def __init__(self, control: Optional[ControlConfig] = None,
                 camera: Optional[CameraConfig] = None):
        self.cfg = control or ControlConfig()
        self.camera_cfg = camera or CameraConfig()
        self.model = CameraModel(self.camera_cfg)

    def aim_for_search(self) -> float:
        """Gimbal angle to use while there is no target estimate at all."""
        return self.camera_cfg.search_pitch_deg if self.camera_cfg.gimbal else self.camera_cfg.fixed_pitch_deg

    def desired(self, out: TrackerOutput, current_pitch_deg: float,
                down_range_m: Optional[float] = None) -> FollowOutput:
        if out.target is None:
            return FollowOutput(DriveCommand(), self.aim_for_search(), None)

        rel = self.model.locate(out.target, current_pitch_deg)
        height = self._height_above_target(rel, down_range_m)

        if out.state == TrackState.TRACKING:
            return self._track(rel, height)
        if out.state == TrackState.LOST:
            return self._lost(rel, height)
        return FollowOutput(DriveCommand(), self.aim_for_search(), None)

    # ---- internals ----------------------------------------------------------
    def _height_above_target(self, rel: RelativePosition, down_range_m: Optional[float]) -> float:
        """How far above the target's head-top the drone currently is (metres,
        positive = above). Prefers a direct downward-rangefinder reading once
        the drone is nearly overhead — vision-from-a-distance is much noisier."""
        vision_height = -(rel.up + self.model.head_top_offset())
        if (down_range_m is not None and rel.horizontal <= self.cfg.down_valid_radius_m):
            return down_range_m
        return vision_height

    def _aim_pitch(self, rel: RelativePosition, height: float) -> float:
        if not self.camera_cfg.gimbal:
            return self.camera_cfg.fixed_pitch_deg
        # Point the camera at the target's chest (`aim_below_face_m` under the
        # face): the pitch-down angle from level is atan(drop / horizontal
        # distance) -- the drop is the numerator. (A target straight ahead and
        # level means a near-level camera; one straight below means pointing
        # straight down.)
        drop = height + self.model.head_top_offset() + self.camera_cfg.aim_below_face_m
        pitch = math.degrees(math.atan2(max(drop, 0.05), rel.horizontal))
        return clamp(pitch, self.camera_cfg.min_pitch_deg, self.camera_cfg.max_pitch_deg)

    def _track(self, rel: RelativePosition, height: float) -> FollowOutput:
        c = self.cfg

        # -- vertical: close the gap to the desired hover height --
        height_error = c.hover_height_above_target_m - height
        if abs(height_error) < c.height_deadband_m:
            up = 0.0
        else:
            up = clamp(c.vertical_kp * height_error, -c.max_descend_mps, c.max_climb_mps)

        # -- horizontal: close in toward directly above the target (or `standoff_m` short of it) --
        horiz = rel.horizontal
        ux, uy = (rel.forward / horiz, rel.right / horiz) if horiz > 1e-6 else (0.0, 0.0)
        gap = horiz - c.standoff_m
        if abs(gap) < c.horizontal_deadband_m:
            speed = 0.0
        else:
            speed = clamp(c.horizontal_kp * gap, -c.max_horizontal_mps, c.max_horizontal_mps)
            # Descend first, close in second: throttle back while still well above the
            # target height, so we don't overshoot past them while high in the air.
            excess = max(0.0, height - c.hover_height_above_target_m)
            speed *= clamp(1.0 - excess / max(c.approach_radius_m, 1e-6), 0.2, 1.0)
        forward, right = ux * speed, uy * speed

        # -- yaw: keep the target inside the camera's cone; fade out as we get overhead --
        yaw = 0.0
        bearing_deg = math.degrees(math.atan2(rel.right, rel.forward))
        if abs(bearing_deg) > c.yaw_deadband_deg:
            fade = min(1.0, horiz / max(c.yaw_release_m, 1e-6))
            yaw = clamp(c.yaw_kp * bearing_deg * fade, -c.max_yaw_rate_dps, c.max_yaw_rate_dps)

        return FollowOutput(DriveCommand(yaw, forward, right, up), self._aim_pitch(rel, height), height, horiz, rel)

    def _lost(self, rel: RelativePosition, height: float) -> FollowOutput:
        c = self.cfg
        bearing_deg = math.degrees(math.atan2(rel.right, rel.forward))
        if rel.horizontal <= c.lost_hold_within_m and abs(bearing_deg) <= 90:
            # Last seen close and in front: probably slipped underneath us, out of
            # the tilted camera's cone. Hold and look straight down rather than
            # flying off. (Last seen close but BEHIND: they walked off that way,
            # so turn after them, below.)
            return FollowOutput(DriveCommand(), self.camera_cfg.max_pitch_deg, height, rel.horizontal)
        # Turn toward where they went, with the camera raised to the search angle:
        # the last estimate is stale (they have moved on since), and a steep
        # "look where they were" pitch would miss them walking away across the floor.
        yaw = c.lost_search_yaw_dps if bearing_deg > 0 else -c.lost_search_yaw_dps
        return FollowOutput(DriveCommand(yaw_rate_dps=yaw), self.aim_for_search(), height, rel.horizontal)
