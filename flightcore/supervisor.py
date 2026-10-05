"""Flight-mode state machine, failsafes and landing detection.

Pure logic: `update(Context) -> Decision`, no I/O, no hardware.  Commands (arm, takeoff, velocity, ...)
only change state; the next `update` turns state into a Setpoint.

Modes
  DISARMED  motors off                      ARMED    on ground, motors at idle
  TAKEOFF   climb at fixed rate to height   HOLD     hold position (or velocity / attitude if degraded)
  GUIDED    velocity command (body frame)   GOTO     fly to NED position
  LAND      descend, disarm on touchdown    RTL      climb, fly home, land
  KILLED    motors off until disarmed (crash / flip / operator kill)

Failsafes (latching ones cannot be undone by a new flight command; only disarm resets them)
  battery <= warn -> RTL        battery <= critical -> LAND
  command silence  -> hold (a new command resumes), then LAND after hold_before_land_s (latched)
  estimator: no position -> velocity hold; no velocity -> level hold; no attitude/altitude -> emergency descent
  fence (radius / altitude) -> RTL or LAND
  flip / crash (tilt > kill_tilt for kill_tilt_time) -> KILL
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .config import SupervisorConfig
from .control.position import Setpoint
from .mathutil import wrap_pi
from .state import NavEstimate

NAN = float("nan")


class Mode(str, Enum):
    DISARMED = "disarmed"
    ARMED = "armed"
    TAKEOFF = "takeoff"
    HOLD = "hold"
    GUIDED = "guided"
    GOTO = "goto"
    LAND = "land"
    RTL = "rtl"
    KILLED = "killed"


FLYING = (Mode.TAKEOFF, Mode.HOLD, Mode.GUIDED, Mode.GOTO, Mode.LAND, Mode.RTL)


@dataclass
class Context:
    t: float
    dt: float
    est: NavEstimate
    tilt: float                      # rad, from whichever attitude is trusted
    batt_frac: Optional[float] = None
    att_ok: bool = True              # attitude usable for closed-loop control
    impact: bool = False             # accelerometer spike seen recently
    ground_hint: bool = False        # raw rangefinder says we are within a few cm of the ground


@dataclass
class Decision:
    power: str                       # 'off' | 'idle' | 'fly' | 'emergency'
    sp: Setpoint = field(default_factory=Setpoint)
    disarm: bool = False


class Supervisor:
    def __init__(self, cfg: SupervisorConfig):
        self.c = cfg
        self.mode = Mode.DISARMED
        self.latched: Optional[str] = None       # None | 'rtl' | 'land'
        self.reason = ""
        self.events: list[tuple[float, str]] = []
        self.home = np.zeros(3)
        self.hold_pos = np.zeros(3)
        self.hold_yaw = 0.0
        self.cmd_vel = np.zeros(3)               # body-heading frame: forward, right, down
        self.cmd_yaw_rate = 0.0
        self.cmd_t = -math.inf
        self.goto_pos = np.zeros(3)
        self.goto_yaw: Optional[float] = None
        self.takeoff_alt = 0.0
        self._t = 0.0
        self._mode_t = 0.0
        self._stale_since: Optional[float] = None
        self._tilt_since: Optional[float] = None
        self._contact_since: Optional[float] = None
        self._pos_lost_since: Optional[float] = None
        self._rtl_phase = "climb"
        self._emerg_since: Optional[float] = None
        self._hint_since: Optional[float] = None
        self._vz_hold: Optional[float] = None    # altitude captured while GUIDED commands zero vertical speed
        self._last_vz_cmd = 0.0

    # ------------------------------------------------------------------ helpers
    def _log(self, text: str):
        self.events.append((self._t, text))

    def _enter(self, mode: Mode, est: Optional[NavEstimate] = None):
        if mode != self.mode:
            self._log(f"{self.mode.value}->{mode.value}")
        self.mode = mode
        self._mode_t = self._t
        self._stale_since = None
        self._contact_since = None
        self._vz_hold = None
        if est is not None and mode in (Mode.HOLD, Mode.LAND):
            self._capture_hold(est)

    def _latch(self, kind: str, reason: str, est: NavEstimate):
        self.latched = kind
        self.reason = reason
        self._log(f"failsafe:{reason}->{kind}")
        if kind == "land":
            self._enter(Mode.LAND, est)
        else:
            self._rtl_phase = "climb"
            self._capture_hold(est)
            self._enter(Mode.RTL)

    def _capture_hold(self, est: NavEstimate):
        p = est.p.copy()
        if est.vel_valid:
            vh = est.v[0:2]
            sp = float(np.linalg.norm(vh))
            if sp > 0.05:
                p[0:2] += vh / sp * (sp * sp / (2.0 * self.c.brake_accel))
        self.hold_pos = p
        self.hold_yaw = est.yaw

    def shift_position(self, delta_ne) -> None:
        """The estimator jumped its horizontal position by `delta_ne` (GPS reset).  Move the stored
        targets by the same amount so the vehicle does not fly off chasing a stale coordinate.
        `home` is left alone: it is an absolute (GPS-frame) location."""
        d = np.asarray(delta_ne, dtype=float)
        self.hold_pos[0:2] += d
        self.goto_pos[0:2] += d
        self._log("position_reset")

    def shift_yaw(self, delta: float) -> None:
        """Estimated heading jumped by `delta` (compass reset): move stored heading targets with it."""
        self.hold_yaw = wrap_pi(self.hold_yaw + delta)
        if self.goto_yaw is not None:
            self.goto_yaw = wrap_pi(self.goto_yaw + delta)
        self._log("yaw_reset")

    # ------------------------------------------------------------------ commands
    @property
    def airborne(self) -> bool:
        return self.mode in FLYING

    @property
    def armed(self) -> bool:
        return self.mode not in (Mode.DISARMED, Mode.KILLED)

    def arm(self, est: NavEstimate, t: float):
        self._t = t
        self.latched = None
        self.reason = ""
        self.home = est.p.copy()
        self.hold_yaw = est.yaw
        self.hold_pos = est.p.copy()
        self._enter(Mode.ARMED)

    def disarm(self, force: bool = False) -> bool:
        if self.mode in FLYING and not force:
            return False
        self._enter(Mode.DISARMED)
        self.latched = None
        return True

    def kill(self, why: str = "kill"):
        self.reason = why
        self._enter(Mode.KILLED)

    def takeoff(self, alt: float, est: NavEstimate) -> bool:
        if self.mode != Mode.ARMED:
            return False
        self.takeoff_alt = float(alt)
        self.hold_pos = est.p.copy()
        self.hold_yaw = est.yaw
        self._enter(Mode.TAKEOFF)
        return True

    def _accepts_flight_cmd(self) -> bool:
        return self.mode in (Mode.HOLD, Mode.GUIDED, Mode.GOTO) and self.latched is None

    def command_velocity_body(self, vx: float, vy: float, vz: float, yaw_rate: float, t: float, est: NavEstimate) -> bool:
        if not self._accepts_flight_cmd():
            return False
        self.cmd_vel = np.array([vx, vy, vz], dtype=float)
        self.cmd_yaw_rate = float(yaw_rate)
        self.cmd_t = t
        if self.mode != Mode.GUIDED:
            self._enter(Mode.GUIDED)
        self._stale_since = None
        return True

    def command_position(self, pos_ned, yaw: Optional[float], est: NavEstimate) -> bool:
        if not self._accepts_flight_cmd():
            return False
        self.goto_pos = np.array(pos_ned, dtype=float)
        self.goto_yaw = yaw
        self._enter(Mode.GOTO)
        return True

    def command_hold(self, est: NavEstimate) -> bool:
        if not self._accepts_flight_cmd():
            return False
        self._enter(Mode.HOLD, est)
        return True

    def command_land(self, est: NavEstimate) -> bool:
        if self.mode not in FLYING:
            return False
        if self.mode != Mode.LAND:
            self._enter(Mode.LAND, est)
        return True

    def command_rtl(self, est: NavEstimate) -> bool:
        if self.mode not in FLYING or self.mode == Mode.LAND:
            return False
        if self.mode != Mode.RTL:
            self._rtl_phase = "climb"
            self._capture_hold(est)
            self._enter(Mode.RTL)
        return True

    # ------------------------------------------------------------------ setpoint builders
    def _hold_sp(self, est: NavEstimate, vz: Optional[float] = None) -> Setpoint:
        """Best hold the estimator allows.  vz (NED, m/s) replaces the vertical position hold."""
        pos = self.hold_pos.copy()
        vel = np.zeros(3)
        if vz is not None:
            pos[2] = NAN
            vel[2] = vz
        if est.pos_valid:
            return Setpoint(kind="fly", pos=pos, vel=vel, yaw=self.hold_yaw)
        pos[0:2] = NAN
        if est.vel_valid:
            return Setpoint(kind="fly", pos=pos, vel=vel, yaw=self.hold_yaw)
        return Setpoint(kind="fly", pos=pos, vel=vel, yaw=self.hold_yaw, horizontal=False)

    def _sp_takeoff(self, c: Context) -> Setpoint:
        est = c.est
        cfg = self.c
        err = self.takeoff_alt - est.altitude
        speed = min(cfg.takeoff_speed, max(cfg.takeoff_kp * err, 0.25))
        return self._hold_sp(est, vz=-speed)

    def _sp_guided(self, c: Context) -> Setpoint:
        est, cfg = c.est, self.c
        age = c.t - self.cmd_t
        if age <= cfg.setpoint_timeout_s:
            self._stale_since = None
            psi = est.yaw
            vx, vy, vz = self.cmd_vel
            vn = math.cos(psi) * vx - math.sin(psi) * vy
            ve = math.sin(psi) * vx + math.cos(psi) * vy
            if est.altitude >= cfg.max_altitude and vz < 0.0:
                vz = 0.0
            self.hold_yaw = psi
            self._capture_hold(est)
            pos = None
            if abs(vz) < 0.05:                       # no vertical command: hold the altitude, don't just null vz
                if self._vz_hold is None:
                    self._vz_hold = float(est.p[2])
                pos = np.array([NAN, NAN, self._vz_hold])
            else:
                self._vz_hold = None
            return Setpoint(
                kind="fly", pos=pos, vel=np.array([vn, ve, vz]), yaw=None,
                yaw_rate=self.cmd_yaw_rate, horizontal=est.vel_valid or est.pos_valid,
            )
        if self._stale_since is None:
            self._stale_since = c.t
            self._capture_hold(est)
            self._log("command_timeout")
        if c.t - self._stale_since > cfg.hold_before_land_s:
            self._latch("land", "command_lost", est)
            return self._sp_land(c)
        return self._hold_sp(est)

    def _sp_goto(self, c: Context) -> Setpoint:
        est = c.est
        if not est.pos_valid:
            return self._pos_lost(c)
        self._pos_lost_since = None
        yaw = self.goto_yaw if self.goto_yaw is not None else est.yaw
        return Setpoint(kind="fly", pos=self.goto_pos.copy(), yaw=yaw)

    def _pos_lost(self, c: Context) -> Setpoint:
        """A mode that needs position lost it: brief velocity hold, then land."""
        if self._pos_lost_since is None:
            self._pos_lost_since = c.t
            self._capture_hold(c.est)
        if c.t - self._pos_lost_since > self.c.est_pos_lost_hold_s:
            self._latch("land", "position_lost", c.est)
            return self._sp_land(c)
        return self._hold_sp(c.est)

    def _sp_land(self, c: Context) -> Setpoint:
        est, cfg = c.est, self.c
        h = est.range_height if est.range_height is not None else est.altitude
        if h >= cfg.land_fast_above:
            speed = cfg.land_speed_fast
        elif h <= cfg.land_slow_below:
            speed = cfg.land_speed
        else:
            a = (h - cfg.land_slow_below) / (cfg.land_fast_above - cfg.land_slow_below)
            speed = cfg.land_speed + a * (cfg.land_speed_fast - cfg.land_speed)
        self._last_vz_cmd = speed
        return self._hold_sp(est, vz=speed)

    def _sp_rtl(self, c: Context) -> Setpoint:
        est, cfg = c.est, self.c
        if not est.pos_valid:
            return self._pos_lost(c)
        self._pos_lost_since = None
        yaw = self.hold_yaw
        target_alt = max(cfg.rtl_altitude, est.altitude)
        if self._rtl_phase == "climb":
            if est.altitude >= target_alt - 0.5:
                self._rtl_phase = "cruise"
            pos = np.array([self.hold_pos[0], self.hold_pos[1], -target_alt])
            return Setpoint(kind="fly", pos=pos, yaw=yaw, max_speed_xy=cfg.rtl_speed)
        dist = float(np.linalg.norm(est.p[0:2] - self.home[0:2]))
        if dist < cfg.rtl_arrive_tol:
            self.hold_pos = np.array([self.home[0], self.home[1], est.p[2]])
            self._enter(Mode.LAND)
            return self._sp_land(c)
        pos = np.array([self.home[0], self.home[1], -target_alt])
        return Setpoint(kind="fly", pos=pos, yaw=yaw, max_speed_xy=cfg.rtl_speed)

    # ------------------------------------------------------------------ main tick
    def update(self, c: Context) -> Decision:
        self._t = c.t
        cfg, est = self.c, c.est

        if self.mode in (Mode.DISARMED, Mode.KILLED):
            return Decision("off")

        if self.mode == Mode.ARMED:
            if c.t - self._mode_t > cfg.arm_timeout_s:
                self._log("arm_timeout")
                self.disarm()
                return Decision("off", disarm=True)
            return Decision("idle")

        # ---- flying from here on -------------------------------------------------------------
        # flip / crash
        if c.tilt > math.radians(cfg.kill_tilt_deg):
            if self._tilt_since is None:
                self._tilt_since = c.t
            if c.t - self._tilt_since > cfg.kill_tilt_time_s:
                self.kill("flip_or_crash")
                return Decision("off")
        else:
            self._tilt_since = None

        # battery
        if c.batt_frac is not None:
            if c.batt_frac <= cfg.battery_crit and self.latched != "land":
                self._latch("land", "battery_critical", est)
            elif c.batt_frac <= cfg.battery_warn and self.latched is None and self.mode not in (Mode.LAND, Mode.RTL):
                self._latch("rtl", "battery_low", est)

        # fence
        if est.pos_valid and self.latched is None and cfg.fence_action != "none":
            r = float(np.linalg.norm(est.p[0:2] - self.home[0:2]))
            if r > cfg.max_radius or est.altitude > cfg.max_altitude + 2.0:
                self._latch("land" if cfg.fence_action == "land" else "rtl", "fence", est)

        # takeoff timeout
        if self.mode == Mode.TAKEOFF and c.t - self._mode_t > cfg.takeoff_timeout_s:
            self._latch("land", "takeoff_timeout", est)

        # estimator degradation -> emergency descent
        if not c.att_ok or not est.alt_valid:
            if self._emerg_since is None:
                self._emerg_since = c.t
                self._log("emergency_descent")
                if self.latched != "land":
                    self.latched = "land"
                    self.reason = "estimator_failure"
                    self._enter(Mode.LAND)
            self._hint_since = (c.t if self._hint_since is None else self._hint_since) if c.ground_hint else None
            hint = self._hint_since is not None and c.t - self._hint_since >= cfg.ground_hint_time_s
            if c.t - self._emerg_since > cfg.emergency_max_s or c.impact or hint:
                self._log("emergency_landed")
                self.disarm(force=True)
                return Decision("off", disarm=True)
            return Decision("emergency")
        self._emerg_since = None

        # ---- per-mode setpoint ---------------------------------------------------------------
        m = self.mode
        if m == Mode.TAKEOFF:
            if est.altitude >= self.takeoff_alt - cfg.takeoff_tol:
                self.hold_pos = np.array([self.hold_pos[0], self.hold_pos[1], -self.takeoff_alt])
                self._enter(Mode.HOLD)
                sp = self._hold_sp(est)
            else:
                sp = self._sp_takeoff(c)
        elif m == Mode.HOLD:
            sp = self._hold_sp(est)
        elif m == Mode.GUIDED:
            sp = self._sp_guided(c)
        elif m == Mode.GOTO:
            sp = self._sp_goto(c)
        elif m == Mode.RTL:
            sp = self._sp_rtl(c)
        else:
            sp = self._sp_land(c)

        # landing detection (LAND may have been entered inside the builders above)
        if self.mode == Mode.LAND:
            h = est.range_height if est.range_height is not None else est.altitude
            stopped = abs(est.v[2]) < 0.3
            pushing = (self._last_vz_cmd - est.v[2]) > 0.4       # asked to descend, not moving
            touch = (h < cfg.touchdown_alt and stopped) or (h < 1.0 and pushing and stopped) or c.impact
            if touch:
                if self._contact_since is None:
                    self._contact_since = c.t
                if c.t - self._contact_since >= cfg.touchdown_time_s:
                    self._log("landed")
                    self.disarm(force=True)
                    return Decision("off", disarm=True)
            else:
                self._contact_since = None
        return Decision("fly", sp)
