"""
Turns a list of waypoints into a DriveCommand ("pure pursuit"): aim at a point
a little way ahead on the path, turn toward it, and drive forward in proportion
to how well we are already facing it. Turns in place when badly misaligned,
and slows down as it nears the end of the path. Also climbs or descends toward
the configured patrol altitude, independent of the horizontal motion.
"""

import math
from typing import List, Optional, Tuple

from config import PatrolConfig
from datatypes import DriveCommand, Pose, wrap_angle

XY = Tuple[float, float]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class WaypointFollower:
    def __init__(self, config: Optional[PatrolConfig] = None):
        self.cfg = config or PatrolConfig()

    def command(self, pose: Pose, path: List[XY]) -> DriveCommand:
        if not path:
            return DriveCommand()
        c = self.cfg
        tx, ty = self._lookahead_point(pose, path)
        heading_err = wrap_angle(math.atan2(tx - pose.x, ty - pose.y) - pose.yaw)
        yaw = clamp(c.yaw_kp * math.degrees(heading_err), -c.max_yaw_dps, c.max_yaw_dps)

        up = clamp(c.altitude_kp * (c.altitude_m - pose.z), -1.0, 1.0)

        if abs(math.degrees(heading_err)) > c.turn_in_place_deg:
            return DriveCommand(yaw_rate_dps=yaw, up_mps=up)  # face it first (but keep climbing/descending)

        gx, gy = path[-1]
        to_goal = math.hypot(gx - pose.x, gy - pose.y)
        slow = clamp(to_goal / c.slow_radius_m, 0.2, 1.0)
        fwd = c.cruise_mps * math.cos(heading_err) * slow
        return DriveCommand(yaw_rate_dps=yaw, forward_mps=fwd, up_mps=up)

    def _lookahead_point(self, pose: Pose, path: List[XY]) -> XY:
        """First waypoint (after the one nearest to us) at least `lookahead_m` away."""
        nearest = min(range(len(path)),
                      key=lambda i: math.hypot(path[i][0] - pose.x, path[i][1] - pose.y))
        for p in path[nearest:]:
            if math.hypot(p[0] - pose.x, p[1] - pose.y) >= self.cfg.lookahead_m:
                return p
        return path[-1]
