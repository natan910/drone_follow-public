"""
Shared data types passed between layers. No cv2, numpy, or drone libraries in
here, so every layer can import it without importing any other layer.

    camera -> perception --Detection--> tracking --TrackerOutput--> autopilot
    range sensors --RangeScan--> mapping / avoidance                   |
    driver --Pose--------------> mapping / navigation                  v
    phone --Task-----------------------------------------------> Decision(DriveCommand)
"""

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

BBox = Tuple[int, int, int, int]  # (top, right, bottom, left) in pixels


def wrap_angle(a: float) -> float:
    """Wrap an angle in radians to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class TrackState(Enum):
    SEARCHING = auto()
    TRACKING = auto()
    LOST = auto()


class Task(Enum):
    """What the operator has asked for, set live from the phone. Persists
    across frames until the operator sends a different one."""
    FOLLOW = auto()   # default: patrol until the target is found, close in, hover above it,
                      # then keep station above it as it moves
    HOVER = auto()    # close in and hover above the target, then stay at that spot even if
                      # the target walks off (does not chase)
    PATROL = auto()   # ignore the target entirely, keep mapping/patrolling
    HOLD = auto()     # freeze at the current position
    RETURN = auto()   # go home and land
    LAND = auto()     # land right here


class Mode(Enum):
    """What the autopilot is actually doing this frame (derived from Task + world state)."""
    IDLE = auto()     # the pilot has control: we send nothing
    HOLD = auto()     # standing still (operator request, safety, or nothing to do)
    TRACK = auto()    # vision-servoing: closing in on, climbing/descending to, or following the target
    HOVER = auto()    # arrived and parked at a fixed spot above where the target was (Task.HOVER only)
    LOST = auto()     # target just vanished: turn toward where it went
    SEARCH = auto()   # going to / scanning where the target was last seen
    INVESTIGATE = auto()  # a mission asked for a look: go to a standoff point near a spot, watch it
    EXPLORE = auto()  # heading for space the camera has never covered
    PATROL = auto()   # revisiting the parts of the map not seen for longest
    RETURN = auto()   # going home
    LAND = auto()


# ---- perception / tracking ------------------------------------------------

@dataclass(frozen=True)
class Detection:
    """One perception result for one frame. Position in the image is 2-D."""
    bbox: BBox
    offset_x: float  # horizontal: -1 = left edge, +1 = right edge
    offset_y: float  # vertical:   -1 = bottom edge, +1 = top edge (positive = up)
    size: float       # face height / frame height; a proxy for distance
    source: str = "face"  # "face": identity confirmed this frame.
                          # "track": followed visually only (e.g. from overhead, no face visible)


@dataclass(frozen=True)
class TargetEstimate:
    offset_x: float
    offset_y: float
    size: float


@dataclass(frozen=True)
class TrackerOutput:
    state: TrackState
    target: Optional[TargetEstimate]  # smoothed; in LOST, the last known value
    bbox: Optional[BBox] = None       # latest raw bbox, None if unseen this frame
    identity_age_s: float = 0.0       # seconds since a face last CONFIRMED who this is


# ---- motion ---------------------------------------------------------------

@dataclass(frozen=True)
class DriveCommand:
    """Body-frame motion request. The only thing a driver ever receives."""
    yaw_rate_dps: float = 0.0  # degrees/second, positive = turn right (clockwise)
    forward_mps: float = 0.0   # metres/second, positive = forward, negative = back
    right_mps: float = 0.0     # metres/second, positive = sideways to the right
    up_mps: float = 0.0        # metres/second, positive = climb, negative = descend

    @property
    def is_zero(self) -> bool:
        return not (self.yaw_rate_dps or self.forward_mps or self.right_mps or self.up_mps)

    @property
    def horizontal_speed(self) -> float:
        return math.hypot(self.forward_mps, self.right_mps)


@dataclass(frozen=True)
class Pose:
    """Position in metres east (x) / north (y) of home and height above the
    takeoff point (z); yaw in radians clockwise from north (turning right
    increases yaw)."""
    x: float
    y: float
    yaw: float
    z: float = 0.0


# ---- range sensing --------------------------------------------------------

@dataclass(frozen=True)
class RangeBeam:
    bearing: float             # radians in the body frame, positive = right
    distance: Optional[float]  # metres; None = nothing within max range


@dataclass(frozen=True)
class RangeScan:
    beams: Tuple[RangeBeam, ...]
    max_range: float
    down: Optional[float] = None  # metres to whatever is directly below (ground, or a person's
                                  # head if hovering over them); None = no valid reading
    up: Optional[float] = None    # metres to whatever is directly above; None = no valid reading

    def clearance_at(self, center: float, half_width: float) -> Optional[float]:
        """Closest distance among beams within +-half_width of `center`
        (radians, body frame, wraps correctly). max_range if those beams see
        nothing; None if no beam points that way (a blind spot)."""
        seen = [b for b in self.beams if abs(wrap_angle(b.bearing - center)) <= half_width]
        if not seen:
            return None
        return min(b.distance if b.distance is not None else self.max_range for b in seen)


# ---- autopilot in / out ---------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """Everything the autopilot knows about the world for one step."""
    now: float
    pose: Pose
    detection: Optional[Detection] = None
    scan: Optional[RangeScan] = None
    battery_pct: Optional[float] = None
    autonomy_permitted: bool = True  # False when the pilot has taken over
    frame_age: float = 0.0           # seconds since the last camera frame
    scan_age: float = 0.0            # seconds since the last range reading
    camera_pitch_deg: float = 0.0    # the camera's CURRENT tilt (0 = level, 90 = straight down)
    task: Optional[Task] = None      # the operator's current command, if any (default: Task.FOLLOW)
    hover_height_m: Optional[float] = None  # operator's live edit to the "hover above target" filter
    camera_fov: Optional[Tuple[float, float]] = None  # (hfov_deg, aspect) just measured on the ground


@dataclass(frozen=True)
class Decision:
    cmd: DriveCommand
    mode: Mode
    note: str = field(default="", compare=False)
    camera_pitch_deg: Optional[float] = field(default=None, compare=False)  # gimbal request, if any
