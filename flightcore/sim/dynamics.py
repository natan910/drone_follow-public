"""Rigid-body quadrotor plant (X frame, ArduPilot motor order FR, BL, FL, BR).

Independent of the controller code on purpose: it builds its own force/torque model from
`VehicleParams`, so tests can give the plant different parameters than the controller assumes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import VehicleParams
from ..mathutil import GRAVITY_NED, G, cross3, quat_mul, quat_normalize, quat_to_R, quat_to_euler, quat_from_euler


def motor_geometry(p: VehicleParams):
    """Return (x, y, spin) for the 4 motors. spin +1 = CCW seen from above (body gets +yaw torque)."""
    d = p.d
    x = np.array([d, -d, d, -d])          # FR, BL, FL, BR
    y = np.array([d, -d, -d, d])
    spin = np.array([+1.0, +1.0, -1.0, -1.0])
    return x, y, spin


@dataclass
class WindModel:
    mean: np.ndarray = field(default_factory=lambda: np.zeros(3))   # NED m/s
    gust_std: float = 0.0
    gust_tau: float = 2.0
    _gust: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def step(self, dt: float, rng: np.random.Generator) -> np.ndarray:
        if self.gust_std > 0.0:
            a = math.exp(-dt / self.gust_tau)
            self._gust = a * self._gust + math.sqrt(1 - a * a) * self.gust_std * rng.standard_normal(3)
        return self.mean + self._gust


class Plant:
    def __init__(
        self,
        params: VehicleParams | None = None,
        *,
        motor_scale=(1.0, 1.0, 1.0, 1.0),
        wind: WindModel | None = None,
        rng: np.random.Generator | None = None,
        p0=(0.0, 0.0, 0.0),
        yaw0: float = 0.0,
        ground_z: float = 0.0,
    ):
        self.par = params or VehicleParams()
        self.motor_scale = np.asarray(motor_scale, dtype=float)
        self.wind = wind or WindModel()
        self.rng = rng or np.random.default_rng(0)
        self.ground_z = ground_z
        self.p = np.array(p0, dtype=float)
        self.v = np.zeros(3)
        self.q = quat_from_euler(0.0, 0.0, yaw0)
        self.w = np.zeros(3)
        self.m = np.zeros(4)               # thrust fractions (after motor lag)
        self.on_ground = True
        self.crashed = False
        self.max_impact = 0.0
        self.last_impact = 0.0
        self.t = 0.0
        self.f_body = np.array([0.0, 0.0, -G])   # specific force at the end of last step
        self._wind_now = np.zeros(3)
        x, y, s = motor_geometry(self.par)
        self._x, self._y, self._s = x, y, s
        self._I = np.array(self.par.inertia)

    # ---------------------------------------------------------------- physics
    def _curve(self, u):
        e = self.par.thrust_expo
        u = np.clip(u, 0.0, 1.0)
        return (1.0 - e) * u + e * u * u

    def _deriv(self, x, target, wind):
        par = self.par
        p, v, q, w, m = x[0:3], x[3:6], x[6:10], x[10:13], x[13:17]
        R = quat_to_R(q)
        thrust = par.max_thrust * self.motor_scale * m
        Ft = thrust.sum()
        vb = R.T @ (v - wind)
        drag_b = -par.mass * np.array([par.drag_xy * vb[0], par.drag_xy * vb[1], par.drag_z * vb[2]])
        F_b = np.array([0.0, 0.0, -Ft]) + drag_b
        a = GRAVITY_NED + R @ F_b / par.mass
        tau = np.array(
            [
                -(self._y * thrust).sum(),
                (self._x * thrust).sum(),
                par.yaw_coeff * (self._s * thrust).sum(),
            ]
        )
        Iw = self._I * w
        wdot = (tau - cross3(w, Iw) - par.ang_damp * w) / self._I
        qdot = 0.5 * quat_mul(q, np.array([0.0, w[0], w[1], w[2]]))
        mdot = (target - m) / par.motor_tau
        out = np.empty(17)
        out[0:3] = v
        out[3:6] = a
        out[6:10] = qdot
        out[10:13] = wdot
        out[13:17] = mdot
        return out, (F_b / par.mass)

    def step(self, cmd, dt: float) -> None:
        cmd = np.asarray(cmd, dtype=float)
        target = self._curve(cmd)
        wind = self.wind.step(dt, self.rng)
        self._wind_now = wind
        x = np.concatenate([self.p, self.v, self.q, self.w, self.m])

        if self.on_ground:
            # motors still spin up; check for liftoff
            self.m = self.m + (target - self.m) * (1.0 - math.exp(-dt / self.par.motor_tau))
            R = quat_to_R(self.q)
            thrust = self.par.max_thrust * self.motor_scale * self.m
            net_up = thrust.sum() - self.par.mass * G / max(R[2, 2], 1e-3)
            if net_up > 0.0:
                self.on_ground = False
            else:
                self.v[:] = 0.0
                self.w[:] = 0.0
                self.f_body = -R.T @ GRAVITY_NED
                self.t += dt
                return
            x = np.concatenate([self.p, self.v, self.q, self.w, self.m])

        k1, _ = self._deriv(x, target, wind)
        k2, _ = self._deriv(x + 0.5 * dt * k1, target, wind)
        k3, _ = self._deriv(x + 0.5 * dt * k2, target, wind)
        k4, _ = self._deriv(x + dt * k3, target, wind)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[6:10] = quat_normalize(x[6:10])
        self.p, self.v, self.q, self.w, self.m = x[0:3], x[3:6], x[6:10], x[10:13], x[13:17]
        self.t += dt

        _, f = self._deriv(x, target, wind)
        self.f_body = f

        if self.p[2] >= self.ground_z:
            impact = float(self.v[2])
            roll, pitch, yaw = quat_to_euler(self.q)
            self.last_impact = impact
            self.max_impact = max(self.max_impact, impact)
            if impact > 4.0 or max(abs(roll), abs(pitch)) > math.radians(45.0):
                self.crashed = True
            self.p[2] = self.ground_z
            self.v[:] = 0.0
            self.w[:] = 0.0
            self.q = quat_from_euler(0.0, 0.0, yaw)
            self.on_ground = True
            self.f_body = -quat_to_R(self.q).T @ GRAVITY_NED

    # ---------------------------------------------------------------- helpers
    @property
    def altitude(self) -> float:
        return self.ground_z - self.p[2]

    def euler(self):
        return quat_to_euler(self.q)
