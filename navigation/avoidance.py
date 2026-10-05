"""
Reactive obstacle avoidance: the last line of defence, applied to every command.

It never plans anything. It only looks at the latest range readings and:
    - slows the drone as something ahead gets closer, and stops it short of
      `stop_distance_m`; the same applies sideways when the command strafes
      (the hover/follow controller commands `right_mps` directly)
    - while the drone is trying to drive forward, steers away from whichever
      side is closer; when head-on with no side to prefer, commits to the more
      open one (turning in place is always allowed and is never fought)
    - refuses to reverse unless a rear sensor confirms it is clear
    - refuses to move forward, or sideways toward an unseen side, at all if no
      sensor covers that arc
    - protects the vertical axis too: never lets a descent close the downward
      clearance past `min_clearance_below_m` (a hard floor — this is what
      keeps "hover 30 cm above the target" from ever becoming "land on them"),
      and never lets a climb close the upward clearance past `ceiling_clearance_m`

Because it acts on raw readings, it protects against things the map does not
know about yet (a person walking in, an obstacle the planner routed too close to).
"""

import math
from typing import Optional

from config import AvoidConfig
from datatypes import DriveCommand, RangeScan


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class ObstacleAvoider:
    def __init__(self, config: Optional[AvoidConfig] = None):
        self.cfg = config or AvoidConfig()
        self._escape_sign = 0.0  # committed turn direction while head-on

    def reset(self) -> None:
        self._escape_sign = 0.0

    def filter(self, cmd: DriveCommand, scan: Optional[RangeScan]) -> DriveCommand:
        if scan is None:
            return cmd  # the safety supervisor decides whether that is acceptable
        c = self.cfg
        front = scan.clearance_at(0.0, math.radians(c.front_half_angle_deg))
        yaw, fwd = cmd.yaw_rate_dps, cmd.forward_mps

        if fwd > 0:
            if front is None:
                fwd = 0.0  # blind ahead: do not drive into the unknown
            else:
                fwd *= clamp((front - c.stop_distance_m) /
                             (c.slow_distance_m - c.stop_distance_m), 0.0, 1.0)
        elif fwd < 0:
            rear = scan.clearance_at(math.pi, math.radians(35))
            if rear is None or rear < c.rear_clear_m:
                fwd = 0.0

        if cmd.forward_mps > 0:  # only steer when we are trying to drive toward things
            yaw += self._steering(scan, front)
        else:
            self._escape_sign = 0.0

        right = self._filter_lateral(cmd.right_mps, scan)
        up = self._filter_vertical(cmd.up_mps, scan)
        return DriveCommand(yaw_rate_dps=yaw, forward_mps=fwd, right_mps=right, up_mps=up)

    def _filter_lateral(self, right: float, scan: RangeScan) -> float:
        if right == 0.0:
            return 0.0
        c = self.cfg
        side = scan.clearance_at(math.copysign(math.pi / 2, right), math.radians(c.front_half_angle_deg))
        if side is None:
            return 0.0  # blind on that side: do not strafe into the unknown
        return right * clamp((side - c.stop_distance_m) / (c.slow_distance_m - c.stop_distance_m), 0.0, 1.0)

    def _filter_vertical(self, up: float, scan: RangeScan) -> float:
        c = self.cfg
        down = scan.down
        if down is not None and down <= c.min_clearance_below_m:
            return max(up, c.rescue_climb_mps)  # hard floor: force a climb regardless of what was asked
        if up < 0:
            floor = down if down is not None else c.blind_descend_limit_m
            if floor <= c.min_clearance_below_m:
                return 0.0
            return up * clamp((floor - c.min_clearance_below_m) /
                              max(c.blind_descend_limit_m - c.min_clearance_below_m, 1e-6), 0.0, 1.0)
        if up > 0 and scan.up is not None and scan.up <= c.ceiling_clearance_m:
            return 0.0  # something overhead: do not climb into it
        return up

    # ---- steering ---------------------------------------------------------
    def _steering(self, scan: RangeScan, front: Optional[float]) -> float:
        c = self.cfg
        if front is None or front >= c.slow_distance_m:
            self._escape_sign = 0.0
            push = 0.0
        else:
            push = 0.0
            for b in scan.beams:
                if b.distance is None or abs(b.bearing) > math.radians(100):
                    continue
                closeness = clamp((c.slow_distance_m - b.distance) /
                                  (c.slow_distance_m - c.stop_distance_m), 0.0, 1.0)
                push -= closeness * math.sin(b.bearing)  # obstacle on the right -> turn left

        steer = clamp(c.push_gain_dps * push, -c.max_push_dps, c.max_push_dps)
        if front is not None and front < c.escape_distance_m and abs(push) < 0.15:
            steer += self._escape(scan) * c.max_push_dps  # head-on: pick a side
        return steer

    def _escape(self, scan: RangeScan) -> float:
        """+1 = turn right, -1 = turn left. Picks the side with more room, then
        sticks with it until the way ahead is clear (so it cannot flip-flop)."""
        if self._escape_sign == 0.0:
            left, right = self._side_room(scan, -1), self._side_room(scan, +1)
            if left is not None and right is not None and left != right:
                self._escape_sign = 1.0 if right > left else -1.0
            else:
                self._escape_sign = 1.0
        return self._escape_sign

    @staticmethod
    def _side_room(scan: RangeScan, sign: int) -> Optional[float]:
        """Average free distance on one side (beams 30-110 degrees off the nose)."""
        lo, hi = math.radians(30), math.radians(110)
        d = [b.distance if b.distance is not None else scan.max_range
             for b in scan.beams if lo <= sign * b.bearing <= hi]
        return sum(d) / len(d) if d else None
