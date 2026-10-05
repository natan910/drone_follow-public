"""
R1 investigate: "go and look at that spot", as pure geometry for the autopilot.

A mission (missions/wiring.py PerimeterMission) asks Autopilot.investigate(x, y)
when a person is confirmed in a zone, or a sensor fires. The autopilot then:

  1. flies (planner route, obstacle avoider on) to the STANDOFF point: on the
     line from the spot toward the drone, standoff_m from the spot. Never
     overhead: an uninvolved person must not have a drone above them (EU rules,
     and safety), and from overhead a fixed 30-deg camera cannot see them anyway.
  2. holds there, turning to keep the spot straight ahead. If the person walks,
     the mission moves the spot; the standoff point moves with it.
  3. stops at `until` (timeout), when the mission says so, when the pilot takes
     over, or when the operator picks HOVER / HOLD / RETURN / LAND.
Then back to the normal patrol.
"""

import math
from dataclasses import dataclass
from typing import Tuple

XY = Tuple[float, float]


@dataclass
class Investigation:
    x: float
    y: float
    standoff_m: float
    until: float
    reason: str = ""
    started: float = 0.0
    holding: bool = False
    anchor: XY = (0.0, 0.0)


def standoff_point(drone: XY, spot: XY, standoff_m: float) -> XY:
    """On the spot->drone line, standoff_m from the spot (south of it if the drone is on it)."""
    dx, dy = drone[0] - spot[0], drone[1] - spot[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return spot[0], spot[1] - standoff_m
    return spot[0] + dx / d * standoff_m, spot[1] + dy / d * standoff_m


def bearing_to(x: float, y: float, spot: XY) -> float:
    """Radians clockwise from north, from (x, y) to the spot."""
    return math.atan2(spot[0] - x, spot[1] - y)


def face_rate(x: float, y: float, yaw: float, spot: XY, kp: float, max_dps: float,
              deadband_deg: float = 3.0) -> float:
    """Yaw rate (deg/s, + = right) that turns the nose toward the spot."""
    err = math.degrees((bearing_to(x, y, spot) - yaw + math.pi) % (2 * math.pi) - math.pi)
    if abs(err) < deadband_deg:
        return 0.0
    return max(-max_dps, min(max_dps, kp * err))
