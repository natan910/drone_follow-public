"""
A toy 3-D-ish world for testing the whole autopilot with no camera, no drone,
and no flight-controller simulator.

It is NOT a flight simulator: the "drone" is a point with a heading, an
altitude, and first-order-lag velocities on all four axes (yaw, forward,
right, up). Obstacles are 2-D, floor-to-ceiling walls (boxes), so horizontal
line-of-sight is altitude-independent — a deliberate simplification. It exists
to catch logic bugs (wrong sign, oscillation, driving into walls, flying into
the ground, never finding or never quite reaching the target) before anything
with propellers is involved.

    boxes      axis-aligned rectangles: walls and obstacles (horizontal only)
    scan()     what a ring of horizontal range sensors, plus a downward one,
               would report (with noise + dropouts)
    detection  what the camera + face matcher would report for the person,
               using the SAME CameraModel the real autopilot uses, so a test
               that passes here is testing the real geometry, not a stand-in.
               With `realistic_face`, the face is only recognisable from in
               front of the person and not from steeply above; with
               `body_reid`, the body is still recognised then (source="track"),
               with noisier position and occasional misses, the way
               perception/target_finder.py would report it.

Conventions: x east, y north, z up (metres above takeoff), heading clockwise
from north (positive yaw rate = turn right).
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from config import CameraConfig
from datatypes import Detection, DriveCommand, Pose, RangeBeam, RangeScan, wrap_angle
from perception.geometry import CameraModel, RelativePosition


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    def distance_to(self, x: float, y: float) -> float:
        dx = max(self.x0 - x, 0.0, x - self.x1)
        dy = max(self.y0 - y, 0.0, y - self.y1)
        return math.hypot(dx, dy)


def ray_box(ox: float, oy: float, dx: float, dy: float, box: Box) -> Optional[float]:
    """Distance along the unit ray (dx, dy) from (ox, oy) to the box, or None."""
    tmin, tmax = 0.0, math.inf
    for o, d, lo, hi in ((ox, dx, box.x0, box.x1), (oy, dy, box.y0, box.y1)):
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return None
        else:
            t1, t2 = (lo - o) / d, (hi - o) / d
            tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
    return tmin if tmin <= tmax else None


def room(width: float, height: float, thickness: float = 0.3) -> List[Box]:
    """Four walls around a width x height room centred on the origin."""
    hw, hh, t = width / 2, height / 2, thickness
    return [Box(-hw - t, -hh - t, hw + t, -hh), Box(-hw - t, hh, hw + t, hh + t),
            Box(-hw - t, -hh, -hw, hh), Box(hw, -hh, hw + t, hh)]


class VirtualWorld:
    BEAM_BEARINGS_DEG = (-90, -45, 0, 45, 90)
    DOWN_BEAM_HALF_WIDTH_M = 0.5  # how far off horizontally the downward sensor can still "see" the person's head

    def __init__(self, boxes: Sequence[Box] = (), person: Optional[Tuple[float, float]] = None,
                 start: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 camera: Optional[CameraConfig] = None, person_height_m: float = 1.70,
                 view_range: float = 7.0, max_range: float = 4.0, lag: float = 0.3,
                 offset_noise: float = 0.02, range_noise: float = 0.02,
                 miss_prob: float = 0.05, dropout_prob: float = 0.02,
                 drone_radius: float = 0.35, battery_drain_pct_per_min: float = 3.0,
                 realistic_face: bool = False, face_half_angle_deg: float = 70.0,
                 face_max_elevation_deg: float = 55.0, person_heading_deg: float = 0.0,
                 body_reid: bool = False, reid_miss_prob: float = 0.1, body_offset_noise: float = 0.04,
                 seed: int = 1):
        self.boxes = list(boxes)
        self.person = person
        self.pvx = self.pvy = 0.0
        self.x, self.y, self.heading = start[0], start[1], math.radians(start[2])
        self.z = 0.0

        self.camera_cfg = camera or CameraConfig()
        self.camera_model = CameraModel(self.camera_cfg)
        self.camera_pitch_deg = self.camera_cfg.fixed_pitch_deg
        self.half_fov = math.radians(self.camera_cfg.hfov_deg) / 2   # kept for the viewer/debug tools
        self.size_at_1m = self.camera_model.size_at_1m                # kept: same meaning as before
        self.person_top_z = person_height_m
        self.person_face_z = person_height_m - self.camera_cfg.face_to_head_top_m

        self.view_range, self.max_range, self.lag = view_range, max_range, lag
        self.offset_noise, self.range_noise = offset_noise, range_noise
        self.miss_prob, self.dropout_prob = miss_prob, dropout_prob
        self.drone_radius = drone_radius
        self.realistic_face, self.body_reid = realistic_face, body_reid
        self.face_half_angle = math.radians(face_half_angle_deg)
        self.face_max_elevation = math.radians(face_max_elevation_deg)
        self.person_heading = math.radians(person_heading_deg)  # which way they face, clockwise from north
        self.reid_miss_prob, self.body_offset_noise = reid_miss_prob, body_offset_noise
        self.drain = battery_drain_pct_per_min
        self.rng = random.Random(seed)

        self.t = 0.0
        self.yaw_rate = 0.0     # rad/s, actual
        self.speed = 0.0        # m/s forward, actual
        self.right_speed = 0.0  # m/s sideways, actual
        self.up_speed = 0.0     # m/s vertical, actual
        self.collisions = 0     # steps spent touching an obstacle (horizontal)
        self.landed = False
        self.pilot_override = False

    # ---- ground truth -----------------------------------------------------
    @property
    def pose(self) -> Pose:
        return Pose(self.x, self.y, self.heading, self.z)

    @property
    def battery_pct(self) -> float:
        return max(0.0, 100.0 - self.drain * self.t / 60.0)

    def person_distance(self) -> float:
        """Horizontal distance to the person, ignoring altitude."""
        return math.hypot(self.person[0] - self.x, self.person[1] - self.y)

    def person_bearing(self) -> float:
        """Horizontal angle to the person relative to the nose (rad, + = right)."""
        return wrap_angle(math.atan2(self.person[0] - self.x, self.person[1] - self.y) - self.heading)

    def height_above_person(self) -> float:
        """Metres the drone is above the TOP of the person's head (can be negative)."""
        return self.z - self.person_top_z

    def _relative_to_person(self) -> RelativePosition:
        dx, dy = self.person[0] - self.x, self.person[1] - self.y
        forward = dx * math.sin(self.heading) + dy * math.cos(self.heading)
        right = dx * math.cos(self.heading) - dy * math.sin(self.heading)
        return RelativePosition(forward, right, self.person_face_z - self.z)

    def _first_hit(self, angle: float) -> Optional[float]:
        dx, dy = math.sin(angle), math.cos(angle)
        hits = [t for b in self.boxes if (t := ray_box(self.x, self.y, dx, dy, b)) is not None]
        return min(hits) if hits else None

    # ---- simulation step --------------------------------------------------
    def step(self, cmd: DriveCommand, dt: float) -> None:
        if self.landed or self.pilot_override:
            cmd = DriveCommand()
        k = min(1.0, dt / self.lag)
        self.yaw_rate += (math.radians(cmd.yaw_rate_dps) - self.yaw_rate) * k
        self.speed += (cmd.forward_mps - self.speed) * k
        self.right_speed += (cmd.right_mps - self.right_speed) * k
        self.up_speed += (cmd.up_mps - self.up_speed) * k
        if not self.landed:
            self.heading = wrap_angle(self.heading + self.yaw_rate * dt)
            self.x += (self.speed * math.sin(self.heading) + self.right_speed * math.cos(self.heading)) * dt
            self.y += (self.speed * math.cos(self.heading) - self.right_speed * math.sin(self.heading)) * dt
            self.z = max(0.0, self.z + self.up_speed * dt)
        if self.person is not None:
            self.person = (self.person[0] + self.pvx * dt, self.person[1] + self.pvy * dt)
            if self.pvx or self.pvy:
                self.person_heading = math.atan2(self.pvx, self.pvy)  # people face where they walk
        if any(b.distance_to(self.x, self.y) < self.drone_radius for b in self.boxes):
            self.collisions += 1
        self.t += dt

    def set_camera_pitch(self, pitch_down_deg: float) -> None:
        self.camera_pitch_deg = pitch_down_deg

    # ---- sensors ----------------------------------------------------------
    def scan(self) -> RangeScan:
        beams = []
        for deg in self.BEAM_BEARINGS_DEG:
            bearing = math.radians(deg)
            d = self._first_hit(self.heading + bearing)
            if d is not None and d <= self.max_range and self.rng.random() >= self.dropout_prob:
                beams.append(RangeBeam(bearing, max(0.05, d + self.rng.gauss(0, self.range_noise))))
            else:
                beams.append(RangeBeam(bearing, None))

        down = self.z  # distance to the flat floor, by default
        if self.person is not None and self.person_distance() <= self.DOWN_BEAM_HALF_WIDTH_M:
            down = max(0.05, self.height_above_person())  # the person's head is under the narrow beam instead
        if self.rng.random() < self.dropout_prob:
            down = None
        elif down is not None:
            down = max(0.02, down + self.rng.gauss(0, self.range_noise))

        return RangeScan(tuple(beams), self.max_range, down=down, up=None)  # no ceiling modelled

    def detection(self) -> Optional[Detection]:
        if self.person is None:
            return None
        dist_h = self.person_distance()
        if dist_h > self.view_range:
            return None
        world_bearing = math.atan2(self.person[0] - self.x, self.person[1] - self.y)
        blocker = self._first_hit(world_bearing)
        if blocker is not None and blocker < dist_h:
            return None  # a wall is in the way (horizontal blocking only, see module docstring)
        rel = self._relative_to_person()
        if self.face_visible():
            if self.rng.random() < self.miss_prob:
                return None
            estimate = self.camera_model.project(rel, self.camera_pitch_deg)
            if estimate is None:
                return None  # outside the camera's field of view at the current gimbal angle
            ox = estimate.offset_x + self.rng.gauss(0, self.offset_noise)
            oy = estimate.offset_y + self.rng.gauss(0, self.offset_noise)
            return Detection(bbox=(0, 0, 0, 0), offset_x=ox, offset_y=oy, size=estimate.size, source="face")

        if not self.body_reid or self.rng.random() < self.reid_miss_prob:
            return None
        # The body is recognised if its middle is in view; the face position is
        # then estimated from the body box, so it may even lie outside the frame.
        torso = RelativePosition(rel.forward, rel.right, rel.up - 0.5)
        if self.camera_model.project(torso, self.camera_pitch_deg) is None:
            return None
        estimate = self.camera_model.project(rel, self.camera_pitch_deg, clip=False)
        if estimate is None:
            return None
        n = self.body_offset_noise
        return Detection(bbox=(0, 0, 0, 0), offset_x=estimate.offset_x + self.rng.gauss(0, n),
                         offset_y=estimate.offset_y + self.rng.gauss(0, n),
                         size=estimate.size * (1 + self.rng.gauss(0, n)), source="track")

    def face_visible(self) -> bool:
        """Can the face be recognised from where the drone is? Always, unless
        `realistic_face`: then only from in front and not from steeply above."""
        if not self.realistic_face:
            return True
        dx, dy = self.x - self.person[0], self.y - self.person[1]
        toward_drone = math.atan2(dx, dy)
        if abs(wrap_angle(toward_drone - self.person_heading)) > self.face_half_angle:
            return False  # we are behind them
        elevation = math.atan2(self.z - self.person_face_z, math.hypot(dx, dy))
        return elevation <= self.face_max_elevation
