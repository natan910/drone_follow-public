"""Quad-X mixer with saturation priorities.

Inputs are physical: collective thrust fraction and body torques (N m).  Output is per-motor
ESC command.  When the demand does not fit in [idle, 1] the priority is
    1. roll/pitch torque   2. yaw torque   3. collective thrust
(collective is shifted to keep attitude authority: "airmode"; yaw is scaled down first).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ControlConfig, VehicleParams


@dataclass
class MixResult:
    cmd: np.ndarray            # ESC commands 0..1, shape (4,)
    thrust: np.ndarray         # per-motor thrust fractions actually commanded
    sat_rp: bool               # roll/pitch demand was scaled down
    sat_yaw: bool              # yaw demand was scaled down
    sat_thrust: bool           # collective was shifted / clipped


class Mixer:
    def __init__(self, veh: VehicleParams, cfg: ControlConfig):
        d, k = veh.d, veh.yaw_coeff
        x = np.array([d, -d, d, -d])       # FR, BL, FL, BR
        y = np.array([d, -d, -d, d])
        spin = np.array([+1.0, +1.0, -1.0, -1.0])
        # rows: total thrust, roll torque, pitch torque, yaw torque  (per unit motor thrust)
        self.A = np.vstack([np.ones(4), -y, x, k * spin])
        self.Ainv = np.linalg.inv(self.A)
        self.tmax = veh.max_thrust
        self.expo = veh.thrust_expo
        self.lo = self._curve(cfg.ground_idle if cfg.ground_idle > 0 else veh.idle)
        self.hi = 1.0
        self.min_collective = cfg.min_collective
        self.airmode = cfg.airmode
        self.idle_cmd = cfg.ground_idle if cfg.ground_idle > 0 else veh.idle

    def _curve(self, u: float) -> float:
        e = self.expo
        return (1.0 - e) * u + e * u * u

    def inv_curve(self, x):
        e = self.expo
        x = np.clip(x, 0.0, 1.0)
        if e < 1e-6:
            return x
        return ((e - 1.0) + np.sqrt((1.0 - e) ** 2 + 4.0 * e * x)) / (2.0 * e)

    def _spread(self, d: np.ndarray) -> float:
        return float(d.max() - d.min())

    def mix(self, collective: float, torque) -> MixResult:
        """collective: mean per-motor thrust fraction (0..1). torque: (roll, pitch, yaw) in N m."""
        tau = np.asarray(torque, dtype=float) / self.tmax
        d_rp = self.Ainv @ np.array([0.0, tau[0], tau[1], 0.0])
        d_y = self.Ainv @ np.array([0.0, 0.0, 0.0, tau[2]])
        rng = self.hi - self.lo
        sat_rp = sat_yaw = False

        s = 1.0
        k = 1.0
        if self._spread(d_rp + d_y) > rng:
            sat_yaw = True
            if self._spread(d_rp) <= rng:
                lo_s, hi_s = 0.0, 1.0
                for _ in range(24):
                    mid = 0.5 * (lo_s + hi_s)
                    if self._spread(d_rp + mid * d_y) <= rng:
                        lo_s = mid
                    else:
                        hi_s = mid
                s = lo_s
            else:
                s = 0.0
                k = rng / self._spread(d_rp)
                sat_rp = True
        d = k * d_rp + s * d_y

        c_des = max(collective, self.min_collective)
        c_lo = self.lo - float(d.min())
        c_hi = self.hi - float(d.max())
        if self.airmode:
            c = min(max(c_des, c_lo), c_hi)
            sat_thrust = abs(c - c_des) > 1e-9
            x = c + d
        else:
            sat_thrust = False
            x = c_des + d
            if x.min() < self.lo - 1e-9 or x.max() > self.hi + 1e-9:
                sat_rp = True
            x = np.clip(x, self.lo, self.hi)
        x = np.clip(x, self.lo, self.hi)
        return MixResult(cmd=self.inv_curve(x), thrust=x, sat_rp=sat_rp, sat_yaw=sat_yaw, sat_thrust=sat_thrust)

    def realised(self, thrust_fraction) -> np.ndarray:
        """(total thrust N, roll, pitch, yaw torque N m) produced by per-motor thrust fractions."""
        return self.A @ (np.asarray(thrust_fraction) * self.tmax)
