"""
World-frame station keeping: hold a fixed (x, y, z) point, using the drone's
own pose feedback rather than vision. This is what Task.HOVER hands off to
once the drone has arrived above the target and dwelled there for a moment —
from then on it holds that spot even if the target walks away, rather than
continuing to chase it (that continuous-chase behaviour is Task.FOLLOW,
handled by FollowController instead).

Body-frame commands are recovered from the world-frame position error by
rotating it through the drone's current yaw — the same convention every
driver's forward-kinematics uses (see control/print_driver.py):
    forward = vx*sin(yaw) + vy*cos(yaw)
    right   = vx*cos(yaw) - vy*sin(yaw)
"""

import math
from typing import Optional

from config import ControlConfig
from datatypes import DriveCommand, Pose


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class PositionHold:
    def __init__(self, config: Optional[ControlConfig] = None):
        self.cfg = config or ControlConfig()

    def command(self, pose: Pose, anchor: Pose) -> DriveCommand:
        c = self.cfg
        dx, dy, dz = anchor.x - pose.x, anchor.y - pose.y, anchor.z - pose.z

        dist = math.hypot(dx, dy)
        if dist < c.hold_deadband_m:
            forward = right = 0.0
        else:
            speed = clamp(c.hold_kp * dist, 0.0, c.hold_max_mps)
            vx, vy = dx / dist * speed, dy / dist * speed
            forward = vx * math.sin(pose.yaw) + vy * math.cos(pose.yaw)
            right = vx * math.cos(pose.yaw) - vy * math.sin(pose.yaw)

        up = 0.0 if abs(dz) < c.height_deadband_m else clamp(
            c.vertical_kp * dz, -c.max_descend_mps, c.max_climb_mps)

        return DriveCommand(yaw_rate_dps=0.0, forward_mps=forward, right_mps=right, up_mps=up)
