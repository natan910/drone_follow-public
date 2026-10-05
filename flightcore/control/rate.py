"""Body-rate PID.  Output: torque demand in N m (angular-acceleration demand x nominal inertia).

P acts on the low-passed gyro, D acts on the (further filtered) measured angular acceleration
(no derivative kick on setpoint steps), I is clamped and frozen by the mixer's saturation flags.
Gyroscopic term w x (I w) is fed forward.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import ControlConfig, VehicleParams
from ..mathutil import cross3


class RateController:
    def __init__(self, cfg: ControlConfig, veh: VehicleParams):
        self.kp = np.array(cfg.rate_kp)
        self.ki = np.array(cfg.rate_ki)
        self.kd = np.array(cfg.rate_kd)
        self.imax = np.array(cfg.rate_i_max)
        self.f_gyro = cfg.rate_lpf_hz
        self.f_d = cfg.rate_dterm_lpf_hz
        self.I = np.array(veh.inertia)
        self.reset()

    def reset(self):
        self.integ = np.zeros(3)
        self.w_f = np.zeros(3)
        self.wdot_f = np.zeros(3)
        self._prev_w = None

    @staticmethod
    def _alpha(fc: float, dt: float) -> float:
        rc = 1.0 / (2.0 * math.pi * fc)
        return dt / (rc + dt)

    def update(self, rate_sp, gyro, dt: float, freeze_i=(False, False, False)) -> np.ndarray:
        gyro = np.asarray(gyro, dtype=float)
        a = self._alpha(self.f_gyro, dt)
        if self._prev_w is None:
            self.w_f = gyro.copy()
            self._prev_w = self.w_f.copy()
        else:
            self.w_f = self.w_f + a * (gyro - self.w_f)
        wdot = (self.w_f - self._prev_w) / dt
        self._prev_w = self.w_f.copy()
        self.wdot_f = self.wdot_f + self._alpha(self.f_d, dt) * (wdot - self.wdot_f)

        err = np.asarray(rate_sp, dtype=float) - self.w_f
        for i in range(3):
            if not freeze_i[i]:
                self.integ[i] = min(max(self.integ[i] + self.ki[i] * err[i] * dt, -self.imax[i]), self.imax[i])
        alpha_cmd = self.kp * err + self.integ - self.kd * self.wdot_f
        return self.I * alpha_cmd + cross3(self.w_f, self.I * self.w_f)
