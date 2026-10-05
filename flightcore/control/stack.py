"""FlightController: Setpoint + NavEstimate -> ESC commands.

    position/velocity -> acceleration demand -> thrust vector -> attitude -> body rates -> torque -> mixer
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import ControlConfig, VehicleParams
from ..mathutil import G, GRAVITY_NED, quat_to_R, wrap_pi
from ..state import NavEstimate
from .attitude import AttitudeController
from .mixer import Mixer, MixResult
from .position import PositionController, Setpoint
from .rate import RateController


@dataclass
class ControlOutput:
    cmd: np.ndarray
    collective: float = 0.0
    a_sp: np.ndarray | None = None
    rate_sp: np.ndarray | None = None
    torque: np.ndarray | None = None
    v_sp: np.ndarray | None = None
    yaw_sp: float = 0.0
    hover_frac: float = 0.0
    sat_rp: bool = False
    sat_yaw: bool = False
    sat_thrust: bool = False


class FlightController:
    def __init__(self, cfg: ControlConfig, veh: VehicleParams):
        self.cfg = cfg
        self.veh = veh
        self.mixer = Mixer(veh, cfg)
        self.rate = RateController(cfg, veh)
        self.att = AttitudeController(cfg)
        self.pos = PositionController(cfg)
        self.hover_frac = veh.hover_frac           # learned online
        self._freeze = (False, False, False)
        self._flying = False

    # ------------------------------------------------------------------ helpers
    def reset(self, est: NavEstimate | None = None):
        self.rate.reset()
        v = est.v if est is not None else np.zeros(3)
        yaw = est.yaw if est is not None else 0.0
        self.pos.reset(v, yaw)
        self._freeze = (False, False, False)
        self._flying = False

    def bumpless(self, est: NavEstimate):
        """Re-centre setpoint shaping on the current state (mode change) but keep integrators."""
        self.pos.reset(est.v, self.pos.yaw_sp, keep_integrators=True)

    def shift_yaw(self, delta: float):
        """Estimated heading jumped: keep the heading setpoint pointing the same physical way."""
        self.pos.yaw_sp = wrap_pi(self.pos.yaw_sp + delta)

    def idle(self) -> ControlOutput:
        cmd = np.full(4, self.mixer.idle_cmd)
        self.rate.reset()
        self._flying = False
        return ControlOutput(cmd=cmd, hover_frac=self.hover_frac)

    # ------------------------------------------------------------------ main
    def update(self, est: NavEstimate, sp: Setpoint, dt: float) -> ControlOutput:
        if sp.kind == "idle":
            self.pos.reset(est.v, est.yaw)
            return self.idle()
        if not self._flying:
            self._flying = True
            self.pos.reset(est.v, est.yaw, keep_integrators=False)
            self.pos.yaw_sp = est.yaw

        po = self.pos.update(est.p, est.v, est.yaw, sp, dt)

        # desired specific force (world) = a_sp - g ; thrust points along -f
        f = po.a_sp - GRAVITY_NED
        fn = float(np.linalg.norm(f))
        z_d = -f / fn
        collective = self.hover_frac * fn / G

        R = quat_to_R(est.q)
        rate_sp, _Rd = self.att.update(R, z_d, po.yaw_sp, po.yaw_rate_ff)
        torque = self.rate.update(rate_sp, est.rate, dt, self._freeze)
        mix: MixResult = self.mixer.mix(collective, torque)
        self._freeze = (mix.sat_rp, mix.sat_rp, mix.sat_yaw)

        self._learn_hover(est, po, dt)
        return ControlOutput(
            cmd=mix.cmd,
            collective=collective,
            a_sp=po.a_sp,
            rate_sp=rate_sp,
            torque=torque,
            v_sp=po.v_sp,
            yaw_sp=po.yaw_sp,
            hover_frac=self.hover_frac,
            sat_rp=mix.sat_rp,
            sat_yaw=mix.sat_yaw,
            sat_thrust=mix.sat_thrust,
        )

    def emergency(self, q, rate, dt: float, thrust_scale: float) -> ControlOutput:
        """Attitude-only recovery: hold level, no heading change, slightly-below-hover thrust.
        Used when position/altitude are unknown or the primary attitude is untrusted."""
        if not self._flying:
            self._flying = True
            self.rate.reset()
        R = quat_to_R(q)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        rate_sp, _ = self.att.update(R, np.array([0.0, 0.0, 1.0]), yaw, 0.0)   # z_d = body-z target (down = level)
        torque = self.rate.update(rate_sp, rate, dt, self._freeze)
        collective = self.hover_frac * thrust_scale
        mix = self.mixer.mix(collective, torque)
        self._freeze = (mix.sat_rp, mix.sat_rp, mix.sat_yaw)
        self.pos.reset(np.zeros(3), yaw)
        return ControlOutput(cmd=mix.cmd, collective=collective, rate_sp=rate_sp, torque=torque,
                             hover_frac=self.hover_frac, sat_rp=mix.sat_rp, sat_yaw=mix.sat_yaw,
                             sat_thrust=mix.sat_thrust)

    # ------------------------------------------------------------------ hover-thrust learning
    def _learn_hover(self, est: NavEstimate, po, dt: float):
        c = self.cfg
        if not c.hover_learn:
            return
        steady = (
            est.alt_valid
            and abs(est.v[2]) < 0.3
            and abs(po.v_sp[2] - est.v[2]) < 0.3
            and est.tilt < math.radians(10.0)
            and not po.sat_z
        )
        if not steady:
            return
        i_z = self.pos.i_z
        h = self.hover_frac
        target = h * (G - i_z) / G
        k = dt / c.hover_lpf_tau
        h_new = h + k * (target - h)
        h_new = min(max(h_new, 0.5 * self.veh.hover_frac), 2.0 * self.veh.hover_frac)
        # keep the total command unchanged: move the correction from the integrator into h
        self.pos.i_z = G - (h / h_new) * (G - i_z)
        self.hover_frac = h_new
