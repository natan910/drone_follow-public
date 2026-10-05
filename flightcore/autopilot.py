"""FlightCore: the whole stack behind one object.

    core = FlightCore()
    for frame in sensor_frames:                 # one per loop tick (SensorFrame from hal.py)
        out = core.step(frame)                  # MotorOutput -> ESCs
    core.arm(); core.takeoff(2.0); core.set_velocity_body(1.0, 0, 0, 0)   # commands from the brain

Pipeline per tick:  sensors -> ESKF (+ Mahony backup) -> supervisor (modes, failsafes)
                    -> setpoint -> FlightController -> motor commands.
Thread-safe: commands may come from another thread than step() (one RLock, held for microseconds).
"""
from __future__ import annotations

import math
import threading
from typing import Optional

import numpy as np

from .config import FlightConfig
from .control import FlightController
from .estimation import Eskf, Mahony
from .hal import MotorOutput, SensorFrame
from .mathutil import quat_to_R
from .state import NavEstimate
from .supervisor import Context, Mode, Supervisor


class FlightCore:
    def __init__(self, cfg: FlightConfig | None = None):
        self.cfg = cfg or FlightConfig()
        self.est = Eskf(self.cfg.estimator)
        self.backup = Mahony()
        self.ctl = FlightController(self.cfg.control, self.cfg.vehicle)
        self.sup = Supervisor(self.cfg.supervisor)
        self._lock = threading.RLock()
        self.nav = NavEstimate()
        self.t = 0.0
        self.batt: Optional[float] = None
        self.last_out = MotorOutput(t=0.0, cmd=np.zeros(4), armed=False)
        self.last_ctl = None
        self._seen = {"baro": -math.inf, "gps": -math.inf, "mag": -math.inf}
        self._att_bad_since: Optional[float] = None
        self._att_fast_since: Optional[float] = None
        self._att_ok = True
        self._impact_until = -math.inf
        self._reset_seen = 0
        self._yaw_reset_seen = 0
        self._range_valid: tuple[float, float] = (-math.inf, math.inf)   # (t, m) latest in-range reading
        self._range_low_since: Optional[float] = None                    # below-minimum readings started
        self._backup_ready = False
        self.att_disagreement_deg = 0.0

    # ------------------------------------------------------------------ per tick
    def step(self, frame: SensorFrame) -> MotorOutput:
        with self._lock:
            return self._step(frame)

    def _step(self, frame: SensorFrame) -> MotorOutput:
        imu = frame.imu
        t, dt = imu.t, imu.dt
        self.t = t
        for name in ("baro", "gps", "mag"):
            if getattr(frame, name):
                self._seen[name] = t
        if frame.battery:
            self.batt = frame.battery[-1].frac

        if not self.est.initialized:
            if self.est.feed_init(imu, frame.baro, frame.mag, frame.gps):
                self.backup.init_from(self.est.x.q, self.est.x.gb)
                self._backup_ready = True
            return self._out(t, np.zeros(4), False)

        self.est.set_static(not self.sup.airborne)
        meas = list(frame.baro) + list(frame.gps) + list(frame.mag) + list(frame.range)
        self.est.step(imu, meas)
        qb = self.backup.update(imu.gyro, imu.accel, dt)
        if self.est.reset_seq != self._reset_seen:
            self._reset_seen = self.est.reset_seq
            self.sup.shift_position(self.est.reset_delta)
        if self.est.yaw_reset_seq != self._yaw_reset_seen:
            self._yaw_reset_seen = self.est.yaw_reset_seq
            d = self.est.yaw_reset_delta
            self.sup.shift_yaw(d)
            self.ctl.shift_yaw(d)
        nav = self.est.estimate()
        self.nav = nav

        # ---- attitude cross-check: ESKF vs independent Mahony
        cfgs = self.cfg.supervisor
        Re, Rb = quat_to_R(nav.q), quat_to_R(qb)
        cosang = float(np.clip(Re[:, 2] @ Rb[:, 2], -1.0, 1.0))
        self.att_disagreement_deg = math.degrees(math.acos(cosang))
        dis = self.att_disagreement_deg
        # Is the ESKF being vouched for by an independent source?  (latest GPS-velocity update accepted, <0.6 s ago)
        validated = nav.att_valid and self.est.gps_validated(cfgs.att_validated_window_s)
        if validated:
            # ESKF is consistent with GPS: keep the shadow filter aligned with it (attitude AND gyro bias)
            self.backup.nudge_toward(nav.q, cfgs.backup_sync_rate, dt)
            self.backup.bias = self.est.x.gb.copy()
            self._att_bad_since = self._att_fast_since = None
            disagree = False
        else:
            self._att_bad_since = self._timer(self._att_bad_since, dis > cfgs.attitude_disagree_deg, t)
            self._att_fast_since = self._timer(self._att_fast_since, dis > cfgs.attitude_disagree_fast_deg, t)
            disagree = (
                (self._att_bad_since is not None and t - self._att_bad_since > cfgs.attitude_disagree_time_s)
                or (self._att_fast_since is not None and t - self._att_fast_since > cfgs.attitude_disagree_fast_time_s)
            )
        self._att_ok = nav.att_valid and not disagree

        # ---- impact spike (ground contact) for landing detection / emergency stop
        if float(np.linalg.norm(imu.accel)) > cfgs.impact_accel:
            self._impact_until = t + 0.3

        tilt = nav.tilt if self._att_ok else math.acos(float(np.clip(Rb[2, 2], -1.0, 1.0)))
        hint = self._ground_hint(frame, t)
        ctx = Context(t=t, dt=dt, est=nav, tilt=tilt, batt_frac=self.batt, att_ok=self._att_ok,
                      impact=t < self._impact_until, ground_hint=hint)
        was_armed = self.sup.armed
        dec = self.sup.update(ctx)

        if dec.power == "off":
            if was_armed:
                self.ctl.reset(nav)
            return self._out(t, np.zeros(4), False)
        if dec.power == "idle":
            o = self.ctl.idle()
        elif dec.power == "emergency":
            if self._att_ok:
                q, rate = nav.q, nav.rate
            else:
                q, rate = qb, imu.gyro - self.backup.bias
            o = self.ctl.emergency(q, rate, dt, cfgs.emergency_thrust)
        else:
            o = self.ctl.update(nav, dec.sp, dt)
        self.last_ctl = o
        return self._out(t, o.cmd, True)

    @staticmethod
    def _timer(since: Optional[float], active: bool, t: float) -> Optional[float]:
        if not active:
            return None
        return t if since is None else since

    def _ground_hint(self, frame: SensorFrame, t: float) -> bool:
        """True when the raw rangefinder says we are on the ground: an in-range reading below
        `ground_hint_range`, or (sensor blind below its minimum) a below-minimum reading that follows
        a valid reading under 2 m within the last 3 s.  Never true from a no-return at altitude."""
        c = self.cfg.supervisor
        for r in frame.range:
            if r.valid:
                self._range_valid = (r.t, r.range_m)
                self._range_low_since = None
                if r.range_m < c.ground_hint_range:
                    self._range_low_since = r.t
            elif r.range_m < 0.5 and (r.t - self._range_valid[0]) < 3.0 and self._range_valid[1] < 2.0:
                self._range_low_since = r.t
            else:
                self._range_low_since = None
        return self._range_low_since is not None and (t - self._range_low_since) < 0.2

    def _out(self, t, cmd, armed) -> MotorOutput:
        self.last_out = MotorOutput(t=t, cmd=np.asarray(cmd, dtype=float), armed=armed)
        return self.last_out

    # ------------------------------------------------------------------ pre-arm
    def prearm_check(self) -> list[str]:
        """Return the list of reasons arming is refused (empty = OK)."""
        c = self.cfg.supervisor
        why: list[str] = []
        if not self.est.initialized:
            return ["estimator_not_initialised"]
        n = self.nav
        if n.diverged:
            why.append("estimator_diverged")
        if not n.att_valid:
            why.append("attitude_invalid")
        if not self._att_ok:
            why.append("attitude_cross_check")
        if c.require_mag and not n.yaw_valid:
            why.append("yaw_invalid")
        if not n.alt_valid:
            why.append("altitude_invalid")
        if c.require_gps:
            if not n.pos_valid:
                why.append("position_invalid")
            if self.t - self._seen["gps"] > max(1.0, c.sensor_timeout_s):
                why.append("gps_stale")
        if self.t - self._seen["baro"] > c.sensor_timeout_s:
            why.append("baro_stale")
        if c.require_mag and self.t - self._seen["mag"] > c.sensor_timeout_s:
            why.append("mag_stale")
        if self.batt is not None and self.batt < c.arm_min_battery:
            why.append("battery_low")
        if n.tilt > math.radians(c.arm_max_tilt_deg):
            why.append("not_level")
        if float(np.linalg.norm(n.rate)) > c.arm_max_rate or float(np.linalg.norm(n.v)) > 0.3:
            why.append("not_still")
        if self.sup.armed:
            why.append("already_armed")
        return why

    # ------------------------------------------------------------------ commands
    def arm(self) -> tuple[bool, list[str]]:
        with self._lock:
            why = self.prearm_check()
            if why:
                return False, why
            self.ctl.reset(self.nav)
            self.sup.arm(self.nav, self.t)
            return True, []

    def disarm(self, force: bool = False) -> bool:
        with self._lock:
            return self.sup.disarm(force)

    def kill(self):
        with self._lock:
            self.sup.kill("operator_kill")

    def takeoff(self, alt_m: float) -> bool:
        with self._lock:
            return self.sup.takeoff(alt_m, self.nav)

    def set_velocity_body(self, vx: float, vy: float, vz: float, yaw_rate: float) -> bool:
        """Body-heading frame (forward, right, down) m/s and yaw rate rad/s (+ = clockwise from above)."""
        with self._lock:
            return self.sup.command_velocity_body(vx, vy, vz, yaw_rate, self.t, self.nav)

    def goto(self, n: float, e: float, d: float, yaw: Optional[float] = None) -> bool:
        with self._lock:
            return self.sup.command_position([n, e, d], yaw, self.nav)

    def hold(self) -> bool:
        with self._lock:
            return self.sup.command_hold(self.nav)

    def land(self) -> bool:
        with self._lock:
            return self.sup.command_land(self.nav)

    def rtl(self) -> bool:
        with self._lock:
            return self.sup.command_rtl(self.nav)

    # ------------------------------------------------------------------ status
    @property
    def mode(self) -> Mode:
        return self.sup.mode

    @property
    def armed(self) -> bool:
        return self.sup.armed

    @property
    def airborne(self) -> bool:
        return self.sup.airborne

    def status(self) -> dict:
        with self._lock:
            n = self.nav
            return {
                "t": self.t,
                "mode": self.sup.mode.value,
                "armed": self.sup.armed,
                "airborne": self.sup.airborne,
                "failsafe": self.sup.latched,
                "reason": self.sup.reason,
                "battery": self.batt,
                "pos_ned": n.p.tolist(),
                "vel_ned": n.v.tolist(),
                "yaw": n.yaw,
                "valid": {"att": n.att_valid, "yaw": n.yaw_valid, "vel": n.vel_valid,
                          "pos": n.pos_valid, "alt": n.alt_valid},
                "att_disagreement_deg": self.att_disagreement_deg,
                "hover_frac": self.ctl.hover_frac,
                "events": list(self.sup.events[-10:]),
            }
