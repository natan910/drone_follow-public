"""
The last stop before a command reaches the driver: hard speed caps and
acceleration limits, applied identically whatever produced the command
(approaching, hovering, patrolling, avoiding, returning home) and to all four
axes (yaw, forward, right, up).
"""

from typing import Optional

from config import ShapingConfig
from datatypes import DriveCommand


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class SlewLimiter:
    """Moves a value toward a target no faster than an acceleration limit.
    Slowing down (toward zero) is allowed `brake_mult` times faster."""

    def __init__(self, accel: float, brake_mult: float = 3.0):
        self.accel, self.brake_mult = accel, brake_mult
        self.value = 0.0

    def step(self, target: float, dt: float) -> float:
        braking = abs(target) < abs(self.value) or target * self.value < 0
        limit = self.accel * (self.brake_mult if braking else 1.0) * dt
        self.value += clamp(target - self.value, -limit, limit)
        return self.value


class CommandShaper:
    MAX_DT = 0.25  # clamp so a long stall can't allow a big jump

    def __init__(self, config: Optional[ShapingConfig] = None):
        self.cfg = config or ShapingConfig()
        c = self.cfg
        self._yaw = SlewLimiter(c.max_yaw_accel_dps2, c.brake_multiplier)
        self._fwd = SlewLimiter(c.max_forward_accel_mps2, c.brake_multiplier)
        self._right = SlewLimiter(c.max_right_accel_mps2, c.brake_multiplier)
        self._up = SlewLimiter(c.max_vertical_accel_mps2, c.brake_multiplier)
        self._last: Optional[float] = None

    def reset(self) -> None:
        self._yaw.value = self._fwd.value = self._right.value = self._up.value = 0.0
        self._last = None

    def shape(self, cmd: DriveCommand, now: float) -> DriveCommand:
        dt = 0.1 if self._last is None else clamp(now - self._last, 0.0, self.MAX_DT)
        self._last = now
        c = self.cfg
        yaw = clamp(cmd.yaw_rate_dps, -c.max_yaw_rate_dps, c.max_yaw_rate_dps)
        fwd = clamp(cmd.forward_mps, -c.max_backward_mps, c.max_forward_mps)
        right = clamp(cmd.right_mps, -c.max_right_mps, c.max_right_mps)
        up = clamp(cmd.up_mps, -c.max_down_mps, c.max_up_mps)
        return DriveCommand(self._yaw.step(yaw, dt), self._fwd.step(fwd, dt),
                            self._right.step(right, dt), self._up.step(up, dt))
