"""Position -> velocity -> acceleration demand (NED), plus heading setpoint handling."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import ControlConfig
from ..mathutil import G, wrap_pi


@dataclass
class Setpoint:
    """What the outer loops should do this tick.

    kind:  'idle'  motors at idle, nothing controlled (on ground, armed)
           'fly'   closed-loop flight
    pos:   NED target per axis; NaN entries are not position-controlled (use `vel` there)
    vel:   NED velocity m/s: direct command where pos is NaN / None, feed-forward elsewhere
    yaw:   absolute heading target; None -> integrate `yaw_rate`
    horizontal: False -> no horizontal control at all (level attitude, e.g. estimator degraded)
    """
    kind: str = "idle"
    pos: Optional[np.ndarray] = None
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: Optional[float] = None
    yaw_rate: float = 0.0
    horizontal: bool = True
    max_speed_xy: Optional[float] = None


@dataclass
class PosOut:
    a_sp: np.ndarray
    yaw_sp: float
    yaw_rate_ff: float
    v_sp: np.ndarray
    sat_xy: bool
    sat_z: bool


class PositionController:
    def __init__(self, cfg: ControlConfig):
        self.c = cfg
        self.tilt_max = math.radians(cfg.max_tilt_deg)
        self.reset(np.zeros(3), 0.0)

    def reset(self, v: np.ndarray, yaw: float, keep_integrators: bool = False):
        self.v_shaped = np.array(v, dtype=float)
        self.yaw_sp = float(yaw)
        if not keep_integrators:
            self.i_xy = np.zeros(2)
            self.i_z = 0.0

    def update(self, p, v, yaw_est: float, sp: Setpoint, dt: float) -> PosOut:
        c = self.c
        v_cmd = np.array(sp.vel, dtype=float)
        vmax_xy = sp.max_speed_xy if sp.max_speed_xy is not None else c.max_speed_xy

        if sp.pos is not None:
            pos = np.asarray(sp.pos, dtype=float)
            if sp.horizontal:
                for i in (0, 1):
                    if not math.isnan(pos[i]):
                        v_cmd[i] = c.pos_kp_xy * (pos[i] - p[i]) + sp.vel[i]
            if not math.isnan(pos[2]):
                v_cmd[2] = c.pos_kp_z * (pos[2] - p[2]) + sp.vel[2]
        if not sp.horizontal:
            v_cmd[0:2] = 0.0
        n = math.hypot(v_cmd[0], v_cmd[1])
        if n > vmax_xy:
            v_cmd[0:2] *= vmax_xy / n
        v_cmd[2] = min(max(v_cmd[2], -c.max_speed_up), c.max_speed_down)

        # slew-limit the velocity setpoint (this is the acceleration the plan may ask for)
        dv = v_cmd - self.v_shaped
        step_xy = c.max_accel_xy * dt
        nxy = math.hypot(dv[0], dv[1])
        if nxy > step_xy:
            dv[0:2] *= step_xy / nxy
        dv[2] = min(max(dv[2], -c.max_accel_z * dt), c.max_accel_z * dt)
        v_prev = self.v_shaped.copy()
        self.v_shaped = self.v_shaped + dv
        a_ff = (self.v_shaped - v_prev) / dt

        e = self.v_shaped - np.asarray(v)
        sat_xy = sat_z = False

        # ---- vertical
        i_z_new = min(max(self.i_z + c.vel_ki_z * e[2] * dt, -c.vel_i_max_z), c.vel_i_max_z)
        a_z = c.vel_kp_z * e[2] + i_z_new + a_ff[2]
        a_z_c = min(max(a_z, -c.max_accel_up), c.max_accel_down)
        if a_z_c != a_z:
            sat_z = True
            a_z = a_z_c                       # integrator stays frozen
        else:
            self.i_z = i_z_new

        # ---- horizontal
        if sp.horizontal:
            i_new = np.clip(self.i_xy + c.vel_ki_xy * e[0:2] * dt, -c.vel_i_max_xy, c.vel_i_max_xy)
            a_xy = c.vel_kp_xy * e[0:2] + i_new + a_ff[0:2]
            a_lim = max(G - a_z, 0.5 * G) * math.tan(self.tilt_max)
            na = math.hypot(a_xy[0], a_xy[1])
            if na > a_lim:
                a_xy = a_xy * (a_lim / na)
                sat_xy = True
            else:
                self.i_xy = i_new
        else:
            a_xy = np.zeros(2)
            self.i_xy[:] = 0.0

        # ---- heading
        yaw_rate_ff = 0.0
        if sp.yaw is not None:
            err = wrap_pi(sp.yaw - self.yaw_sp)
            step = 0.8 * c.max_rate[2] * dt
            d = min(max(err, -step), step)
            self.yaw_sp = wrap_pi(self.yaw_sp + d)
            yaw_rate_ff = d / dt
        else:
            self.yaw_sp = wrap_pi(self.yaw_sp + sp.yaw_rate * dt)
            yaw_rate_ff = sp.yaw_rate
        lag = wrap_pi(self.yaw_sp - yaw_est)
        lag_max = math.radians(c.yaw_lag_max_deg)
        if abs(lag) > lag_max:
            self.yaw_sp = wrap_pi(yaw_est + math.copysign(lag_max, lag))
            yaw_rate_ff = 0.0

        return PosOut(
            a_sp=np.array([a_xy[0], a_xy[1], a_z]),
            yaw_sp=self.yaw_sp,
            yaw_rate_ff=yaw_rate_ff,
            v_sp=self.v_shaped.copy(),
            sat_xy=sat_xy,
            sat_z=sat_z,
        )
