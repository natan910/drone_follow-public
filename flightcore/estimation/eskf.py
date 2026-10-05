"""16-state error-state Kalman filter (ESKF) for a multirotor.

Nominal state   p, v (NED), q (body->world), accel bias, gyro bias, baro bias
Error state     [dp(3) dv(3) dtheta(3) dab(3) dgb(3) dbb(1)],  dtheta is a WORLD-frame rotation:
                R_true = (I + [dtheta]x) R_est.  Injection: q <- exp(dtheta) (x) q.
    d(dp)/dt = dv
    d(dv)/dt = -[R f]x dtheta - R dab             (f = accel - ab)
    d(dtheta)/dt = -R dgb
    biases: random walks

Measurements: GPS pos/vel, baro (with bias state), tilt-compensated rangefinder height, magnetic
heading, airborne "drag" pseudo-measurement (multirotor specific force ~ -D * body velocity; keeps
velocity/bias bounded through GPS loss), on-ground ZUPT + gravity + zero-gyro.

Delayed measurements: the filter stores a snapshot (state + covariance) for every IMU tick over
`delay_window_s`.  A measurement stamped in the past is inserted into a time-ordered log and the
filter is *replayed* from the snapshot at its timestamp (re-propagating the buffered IMU and
re-fusing everything logged after it).  Exact for any latency and any arrival order.
"""
from __future__ import annotations

import bisect
import math
from collections import deque
from typing import Optional

import numpy as np

from ..config import EstimatorConfig
from ..hal import BaroSample, GpsSample, ImuSample, MagSample, RangeSample
from ..mathutil import (
    GRAVITY_NED,
    G,
    quat_from_euler,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    quat_to_R,
    skew,
    wrap_pi,
)
from ..state import NavEstimate

N = 16
_E = np.eye(N)


def _e(i: int) -> np.ndarray:
    return _E[i]


class _Nav:
    __slots__ = ("p", "v", "q", "ab", "gb", "bb")

    def __init__(self, p, v, q, ab, gb, bb):
        self.p, self.v, self.q, self.ab, self.gb, self.bb = p, v, q, ab, gb, bb

    def copy(self) -> "_Nav":
        return _Nav(self.p.copy(), self.v.copy(), self.q.copy(), self.ab.copy(), self.gb.copy(), float(self.bb))

    def finite(self) -> bool:
        return bool(
            np.isfinite(self.p).all() and np.isfinite(self.v).all() and np.isfinite(self.q).all()
            and np.isfinite(self.ab).all() and np.isfinite(self.gb).all() and math.isfinite(self.bb)
        )


class _Slot:
    __slots__ = ("t", "imu", "x", "P", "static", "k")

    def __init__(self, t, imu, x, P, static, k):
        self.t, self.imu, self.x, self.P, self.static, self.k = t, imu, x, P, static, k


class _Meas:
    __slots__ = ("t", "kind", "data", "done")

    def __init__(self, t, kind, data):
        self.t, self.kind, self.data, self.done = t, kind, data, False


# --------------------------------------------------------------------------------------- init
class StaticInitializer:
    """Sliding ~init_time_s window of stillness on the ground, then yields the initial state."""

    def __init__(self, cfg: EstimatorConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.imu = deque()       # (t, gyro, accel)
        self.baro = deque()      # (t, alt)
        self.mag = deque()       # (t, field)
        self.gps = deque()       # (t, pos_ned)
        self._dt = 0.002

    def add(self, imu: ImuSample, baro=(), mag=(), gps=()):
        self._dt = imu.dt
        self.imu.append((imu.t, imu.gyro, imu.accel))
        self.baro += [(imu.t, b.alt) for b in baro]
        self.mag += [(imu.t, m.field) for m in mag]
        self.gps += [(imu.t, g.pos_ned) for g in gps if g.fix]
        t0 = imu.t - self.cfg.init_time_s
        for q in (self.imu, self.baro, self.mag, self.gps):
            while q and q[0][0] < t0 - 1e-9:
                q.popleft()

    def ready(self) -> bool:
        c = self.cfg
        if len(self.imu) < 2 or not self.baro:
            return False
        if self.imu[-1][0] - self.imu[0][0] + self._dt < c.init_time_s - 1e-6:
            return False
        g = np.array([x[1] for x in self.imu])
        a = np.array([x[2] for x in self.imu])
        still = float(np.max(np.std(g, axis=0))) < c.init_gyro_std_max
        an = np.linalg.norm(a, axis=1)
        steady = abs(float(np.mean(an)) - G) < 1.0 and float(np.std(an)) < 1.0
        return bool(still and steady)

    def result(self) -> dict:
        g = np.mean([x[1] for x in self.imu], axis=0)
        f = np.mean([x[2] for x in self.imu], axis=0)
        roll = math.atan2(-f[1], -f[2])
        pitch = math.atan2(f[0], math.hypot(f[1], f[2]))
        have_mag = len(self.mag) > 0
        yaw = 0.0
        mag_norm = None
        if have_mag:
            m = np.mean([x[1] for x in self.mag], axis=0)
            mag_norm = float(np.linalg.norm(m))
            q0 = quat_from_euler(roll, pitch, 0.0)
            mw = quat_to_R(q0) @ m
            yaw = wrap_pi(self.cfg.mag_declination - math.atan2(mw[1], mw[0]))
        have_gps = len(self.gps) > 0
        p0 = np.zeros(3)
        if have_gps:
            p0[0:2] = np.mean([x[1][0:2] for x in self.gps], axis=0)
        return dict(
            q=quat_from_euler(roll, pitch, yaw),
            gb=g,
            p=p0,
            baro_ref=float(np.mean([x[1] for x in self.baro])),
            mag_norm=mag_norm,
            have_mag=have_mag,
            have_gps=have_gps,
        )


# --------------------------------------------------------------------------------------- filter
class Eskf:
    def __init__(self, cfg: EstimatorConfig | None = None):
        self.cfg = cfg or EstimatorConfig()
        self.init = StaticInitializer(self.cfg)
        self.initialized = False
        self.t = 0.0
        self.k = 0
        self.static = True
        self.x: _Nav | None = None
        self.P = np.zeros((N, N))
        self.slots: deque[_Slot] = deque()
        self.log: list[_Meas] = []
        self._logkeys: list[float] = []
        self.baro_ref = 0.0
        self.mag_norm_ref: Optional[float] = None
        self.have_mag = False
        self.have_gps = False
        self.diverged = False
        self._F = np.eye(N)
        self._F[0:3, 3:6] = 0.0
        self._pending_reset = None
        self.reset_seq = 0                                  # counts horizontal position resets
        self.reset_delta = np.zeros(2)                      # NE jump of the last reset (new - old)
        self.yaw_reset_seq = 0                              # counts yaw resets
        self.yaw_reset_delta = 0.0                          # rad, yaw jump of the last reset
        self._pending_yaw_reset: Optional[float] = None
        self._mag_rej_since: Optional[float] = None
        self._gps_vel_accepted = False                      # gate result of the latest GPS-velocity fusion
        self._last_inflate = -math.inf
        # health timers / bookkeeping (updated on first (live) fusion only)
        self.last_ok = {"gps_pos": -math.inf, "gps_vel": -math.inf, "alt": -math.inf, "mag": -math.inf,
                        "drag": -math.inf, "static": -math.inf}
        self.range_height: Optional[float] = None
        self.last_range_ok = -math.inf
        self._range_rej = 0
        self._range_holdoff = (math.inf, -math.inf)         # (start, until) in measurement time
        self._rej_since = {"gps_pos_h": None, "gps_vel_h": None}
        self.stats: dict[str, list[int]] = {}
        self.ratio: dict[str, float] = {}
        self.last_imu: Optional[ImuSample] = None
        self._drag_every = 1
        self._static_every = 1

    # ------------------------------------------------------------------ setup
    def feed_init(self, imu: ImuSample, baro=(), mag=(), gps=()) -> bool:
        """Call every tick until it returns True (vehicle must be still)."""
        if self.initialized:
            return True
        self.init.add(imu, baro, mag, gps)
        if self.init.ready():
            r = self.init.result()
            self.initialize(
                q=r["q"], p=r["p"], gb=r["gb"], baro_ref=r["baro_ref"], mag_norm=r["mag_norm"],
                have_mag=r["have_mag"], have_gps=r["have_gps"], t=imu.t, dt=imu.dt,
            )
        return self.initialized

    def initialize(self, q, p=None, v=None, gb=None, ab=None, bb=0.0, *, baro_ref=0.0, mag_norm=None,
                   have_mag=True, have_gps=True, t=0.0, dt=0.002):
        c = self.cfg
        self.x = _Nav(
            np.zeros(3) if p is None else np.array(p, dtype=float),
            np.zeros(3) if v is None else np.array(v, dtype=float),
            quat_normalize(q),
            np.zeros(3) if ab is None else np.array(ab, dtype=float),
            np.zeros(3) if gb is None else np.array(gb, dtype=float),
            float(bb),
        )
        s = np.zeros(N)
        s[0:2] = c.init_sigma_pos if have_gps else 10.0
        s[2] = c.init_sigma_pos
        s[3:6] = c.init_sigma_vel
        s[6:8] = c.init_sigma_tilt
        s[8] = c.init_sigma_yaw if have_mag else math.pi
        s[9:12] = c.init_sigma_ab
        s[12:15] = c.init_sigma_gb
        s[15] = c.init_sigma_bb
        self.P = np.diag(s * s)
        self.baro_ref = baro_ref
        self.mag_norm_ref = mag_norm
        self.have_mag = have_mag
        self.have_gps = have_gps
        self.t = t
        self.slots.clear()
        self.log.clear()
        self._logkeys.clear()
        self._drag_every = max(1, int(round(c.drag_period_s / dt)))
        self._static_every = max(1, int(round(c.static_period_s / dt)))
        self.initialized = True
        self.diverged = False

    def perturb_attitude(self, dtheta_world) -> None:
        """Fault injection for tests: rotate the attitude estimate (now and in all stored snapshots)
        by a world-frame rotation vector, without touching the covariance."""
        dq = quat_from_rotvec(np.asarray(dtheta_world, dtype=float))
        if self.x is not None:
            self.x.q = quat_normalize(quat_mul(dq, self.x.q))
        for s in self.slots:
            s.x.q = quat_normalize(quat_mul(dq, s.x.q))

    def set_static(self, static: bool):
        """True while the vehicle is on the ground and not moving (enables ZUPT/gravity/zero-gyro);
        False in flight (enables drag pseudo-measurement)."""
        self.static = bool(static)

    # ------------------------------------------------------------------ main entry
    def step(self, imu: ImuSample, measurements=()) -> None:
        if not self.initialized:
            return
        self._propagate(imu)
        self.t = imu.t
        self.k += 1
        self.last_imu = imu
        self.slots.append(_Slot(imu.t, imu, self.x.copy(), self.P.copy(), self.static, self.k))
        window = self.cfg.delay_window_s
        while len(self.slots) > 2 and self.slots[-1].t - self.slots[0].t > window:
            self.slots.popleft()
        floor = self.slots[0].t - self.slots[0].imu.dt
        cut = bisect.bisect_right(self._logkeys, floor)
        if cut:
            del self.log[:cut]
            del self._logkeys[:cut]

        earliest = len(self.slots) - 1
        for m in measurements:
            rec = self._admit(m, floor)
            if rec is None:
                continue
            idx = self._slot_index(rec.t)
            if idx is not None:
                earliest = min(earliest, idx)
        self._run_from(earliest)
        if self._pending_reset is not None:
            self._apply_reset()
        if self._pending_yaw_reset is not None:
            self._apply_yaw_reset()
        if not self.x.finite() or not np.isfinite(self.P).all() or float(np.max(np.diag(self.P))) > 1e8:
            self.diverged = True
        if self.k % 25 == 0:
            self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------ log / replay
    def _admit(self, m, floor: float) -> Optional[_Meas]:
        if isinstance(m, GpsSample):
            kind = "gps"
        elif isinstance(m, BaroSample):
            kind = "baro"
        elif isinstance(m, MagSample):
            kind = "mag"
        elif isinstance(m, RangeSample):
            kind = "range"
        else:
            return None
        if m.t <= floor:
            self.stats.setdefault(kind + "_stale", [0, 0])[1] += 1
            return None
        rec = _Meas(min(m.t, self.t), kind, m)
        i = bisect.bisect_right(self._logkeys, rec.t)
        self._logkeys.insert(i, rec.t)
        self.log.insert(i, rec)
        return rec

    def _slot_index(self, t: float) -> Optional[int]:
        n = len(self.slots)
        if t > self.slots[-1].t:
            return None
        i = n - 1
        while i > 0 and self.slots[i - 1].t >= t:
            i -= 1
        return i

    def _run_from(self, i: int):
        n = len(self.slots)
        if i < n - 1:
            self.x = self.slots[i].x.copy()
            self.P = self.slots[i].P.copy()
        for j in range(i, n):
            s = self.slots[j]
            if j > i:
                self._propagate(s.imu)
                s.x = self.x.copy()
                s.P = self.P.copy()
            self._fuse_tick(j)

    def _fuse_tick(self, j: int):
        s = self.slots[j]
        lo = self.slots[j - 1].t if j > 0 else s.t - s.imu.dt
        # log is time-sorted: locate window (lo, s.t]
        a = bisect.bisect_right(self._logkeys, lo)
        b = bisect.bisect_right(self._logkeys, s.t)
        for rec in self.log[a:b]:
            live = not rec.done
            rec.done = True
            if rec.kind == "gps":
                self._fuse_gps(rec.data, live)
            elif rec.kind == "baro":
                self._fuse_baro(rec.data, live)
            elif rec.kind == "mag":
                self._fuse_mag(rec.data, live)
            else:
                self._fuse_range(rec.data, live)
        live_tick = j == len(self.slots) - 1 and self.slots[j].k == self.k
        if s.static:
            if s.k % self._static_every == 0:
                self._fuse_static(j, live_tick)
        elif s.k % self._drag_every == 0 and self.cfg.use_drag_fusion:
            self._fuse_drag(j, live_tick)

    # ------------------------------------------------------------------ propagation
    def _propagate(self, imu: ImuSample):
        dt = imu.dt
        x = self.x
        w = imu.gyro - x.gb
        f = imu.accel - x.ab
        R = quat_to_R(x.q)
        Rf = R @ f
        a = Rf + GRAVITY_NED
        x.p = x.p + x.v * dt + 0.5 * a * dt * dt
        x.v = x.v + a * dt
        x.q = quat_normalize(quat_mul(x.q, quat_from_rotvec(w * dt)))

        F = self._F
        F[0:3, 3:6] = 0.0
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        F[3:6, 6:9] = -skew(Rf) * dt
        F[3:6, 9:12] = -R * dt
        F[6:9, 12:15] = -R * dt
        P = F @ self.P @ F.T
        c = self.cfg
        q = np.empty(N)
        q[0:3] = 0.0
        q[3:6] = (c.accel_noise ** 2) * dt
        q[6:9] = (c.gyro_noise ** 2) * dt
        q[9:12] = (c.accel_bias_walk ** 2) * dt
        q[12:15] = (c.gyro_bias_walk ** 2) * dt
        q[15] = (c.baro_bias_walk ** 2) * dt
        P.flat[:: N + 1] += q
        self.P = P

    # ------------------------------------------------------------------ update machinery
    def _update(self, ys, Hs, Rs, gate: float, name: str, live: bool) -> bool:
        P = self.P
        worst = 0.0
        for y, h, r in zip(ys, Hs, Rs):
            s = float(h @ P @ h) + r
            worst = max(worst, y * y / s)
        ok = worst <= gate * gate
        if live:
            self.ratio[name] = math.sqrt(worst)
            st = self.stats.setdefault(name, [0, 0])
            st[0 if ok else 1] += 1
        if not ok:
            return False
        dx = np.zeros(N)
        for y, h, r in zip(ys, Hs, Rs):
            y_eff = y - float(h @ dx)
            PHt = P @ h
            S = float(h @ PHt) + r
            K = PHt / S
            dx += K * y_eff
            P = P - np.outer(K, PHt)
        d = np.diag(P)
        if (d < 0).any():
            P = P + np.diag(np.maximum(-d, 0.0) + 1e-12)
        self.P = 0.5 * (P + P.T)
        self._inject(dx)
        return True

    def _inject(self, dx: np.ndarray):
        x = self.x
        x.p = x.p + dx[0:3]
        x.v = x.v + dx[3:6]
        dth = dx[6:9]
        if dth[0] != 0.0 or dth[1] != 0.0 or dth[2] != 0.0:
            x.q = quat_normalize(quat_mul(quat_from_rotvec(dth), x.q))
        x.ab = np.clip(x.ab + dx[9:12], -2.0, 2.0)
        x.gb = np.clip(x.gb + dx[12:15], -0.35, 0.35)
        x.bb = float(np.clip(x.bb + dx[15], -50.0, 50.0))

    # ------------------------------------------------------------------ sensors
    def _fuse_gps(self, g: GpsSample, live: bool):
        if not g.fix:
            return
        c = self.cfg
        sh = max(c.gps_pos_h, g.sigma_h or 0.0)
        sv = max(c.gps_pos_v, g.sigma_v or 0.0)
        x = self.x
        y = g.pos_ned - x.p
        ok_h = self._update([y[0], y[1]], [_e(0), _e(1)], [sh * sh] * 2, c.gate_gps_pos, "gps_pos_h", live)
        y = g.pos_ned - self.x.p
        ok_v = self._update([y[2]], [_e(2)], [sv * sv], c.gate_gps_pos, "gps_pos_v", live)
        y = g.vel_ned - self.x.v
        ok_vh = self._update([y[0], y[1]], [_e(3), _e(4)], [c.gps_vel_h ** 2] * 2, c.gate_gps_vel, "gps_vel_h", live)
        y = g.vel_ned - self.x.v
        ok_vv = self._update([y[2]], [_e(5)], [c.gps_vel_v ** 2], c.gate_gps_vel, "gps_vel_v", live)
        if not live:
            return
        self._gps_vel_accepted = ok_vh
        if ok_h:
            self.last_ok["gps_pos"] = self.t
        if ok_vh or ok_vv:
            self.last_ok["gps_vel"] = self.t
        if ok_v:
            self.last_ok["alt"] = self.t
        for key, ok in (("gps_pos_h", ok_h), ("gps_vel_h", ok_vh)):
            if ok:
                self._rej_since[key] = None
            elif self._rej_since[key] is None:
                self._rej_since[key] = self.t
        since = self._rej_since["gps_vel_h"]
        if since is not None and self.t - since > c.inflate_after_reject_s and self.t - self._last_inflate > 0.5:
            self._inflate()
        since = self._rej_since["gps_pos_h"]
        if since is not None and self.t - since > c.reset_after_reject_s:
            self._pending_reset = ("pos", g)
        since = self._rej_since["gps_vel_h"]
        if since is not None and self.t - since > c.reset_after_reject_s:
            self._pending_reset = ("vel" if self._pending_reset is None else "both", g)

    def _inflate(self):
        """Estimate and measurements disagree for a while: the filter is over-confident.  Add
        uncertainty to velocity, tilt and the biases so the next measurements can pull it back."""
        self._last_inflate = self.t
        d = np.zeros(N)
        d[3:6] = 0.5 ** 2
        d[6:8] = 0.03 ** 2
        d[9:12] = 0.1 ** 2
        d[12:15] = 0.01 ** 2
        self.P.flat[:: N + 1] += d
        self.stats.setdefault("inflate", [0, 0])[0] += 1

    def _apply_yaw_reset(self):
        y = self._pending_yaw_reset
        self._pending_yaw_reset = None
        x = self.x
        x.q = quat_normalize(quat_mul(quat_from_rotvec(np.array([0.0, 0.0, y])), x.q))
        P = self.P
        P[8, :] = 0.0
        P[:, 8] = 0.0
        P[8, 8] = self.cfg.init_sigma_yaw ** 2
        self.yaw_reset_delta = y
        self.yaw_reset_seq += 1
        self._mag_rej_since = None
        self.stats.setdefault("yaw_reset", [0, 0])[0] += 1
        self.slots.clear()
        self.log.clear()
        self._logkeys.clear()

    def _apply_reset(self):
        kind, g = self._pending_reset
        self._pending_reset = None
        c = self.cfg
        lag = max(self.t - g.t, 0.0)
        x = self.x
        P = self.P
        if kind in ("pos", "both"):
            old = x.p[0:2].copy()
            x.p[0:2] = g.pos_ned[0:2] + g.vel_ned[0:2] * lag
            self.reset_delta = x.p[0:2] - old
            self.reset_seq += 1
            P[0:2, :] = 0.0
            P[:, 0:2] = 0.0
            P[0, 0] = P[1, 1] = (c.gps_pos_h * 2.0) ** 2
            self._rej_since["gps_pos_h"] = None
        if kind in ("vel", "both"):
            x.v[0:2] = g.vel_ned[0:2]
            P[3:5, :] = 0.0
            P[:, 3:5] = 0.0
            P[3, 3] = P[4, 4] = (c.gps_vel_h * 3.0) ** 2
            self._rej_since["gps_vel_h"] = None
        self.P = P
        self.stats.setdefault("gps_reset", [0, 0])[0] += 1
        self.slots.clear()          # history before a reset is no longer consistent
        self.log.clear()
        self._logkeys.clear()

    def _fuse_baro(self, b: BaroSample, live: bool):
        x = self.x
        alt = b.alt - self.baro_ref
        y = alt - (-x.p[2] + x.bb)
        h = np.zeros(N)
        h[2] = -1.0
        h[15] = 1.0
        ok = self._update([y], [h], [self.cfg.baro ** 2], self.cfg.gate_baro, "baro", live)
        if ok and live:
            self.last_ok["alt"] = self.t

    def _fuse_range(self, r: RangeSample, live: bool):
        c = self.cfg
        if not c.use_range or not r.valid or not (c.range_min <= r.range_m <= c.range_max):
            return
        if self._range_holdoff[0] <= r.t < self._range_holdoff[1]:
            return
        x = self.x
        r33 = 1.0 - 2.0 * (x.q[1] ** 2 + x.q[2] ** 2)
        if r33 < math.cos(math.radians(c.range_max_tilt_deg)):
            return
        height = r.range_m * r33
        y = height + x.p[2]
        h = np.zeros(N)
        h[2] = -1.0
        sig = c.range + 0.01 * height
        ok = self._update([y], [h], [sig * sig], c.gate_range, "range", live)
        if not live:
            return
        if ok:
            self._range_rej = 0
            self.last_range_ok = self.t
            self.last_ok["alt"] = self.t
            self.range_height = height
        else:
            self._range_rej += 1
            if self._range_rej >= 10:
                self._range_holdoff = (r.t, r.t + c.range_reject_holdoff_s)
                self._range_rej = 0

    def _fuse_mag(self, m: MagSample, live: bool):
        c = self.cfg
        if not self.have_mag:
            return
        n = float(np.linalg.norm(m.field))
        if n < 1e-6:
            return
        if self.mag_norm_ref is not None and abs(n / self.mag_norm_ref - 1.0) > c.mag_norm_tol:
            if live:
                self.stats.setdefault("mag", [0, 0])[1] += 1
            return
        x = self.x
        R = quat_to_R(x.q)
        if R[2, 2] < math.cos(math.radians(70.0)):
            return
        mw = R @ m.field
        if math.hypot(mw[0], mw[1]) < 0.2 * n:
            return
        y = wrap_pi(c.mag_declination - math.atan2(mw[1], mw[0]))
        h = np.zeros(N)
        h[8] = 1.0
        ok = self._update([y], [h], [c.mag_yaw ** 2], c.gate_mag, "mag", live)
        if not live:
            return
        if ok:
            self.last_ok["mag"] = self.t
            self._mag_rej_since = None
        else:
            if self._mag_rej_since is None:
                self._mag_rej_since = self.t
            elif self.t - self._mag_rej_since > c.mag_reset_after_s:
                self._pending_yaw_reset = y

    def _mean_imu(self, j: int, n: int):
        lo = max(0, j - n + 1)
        acc = np.zeros(3)
        gyr = np.zeros(3)
        cnt = 0
        for i in range(lo, j + 1):
            acc += self.slots[i].imu.accel
            gyr += self.slots[i].imu.gyro
            cnt += 1
        return acc / cnt, gyr / cnt

    def _fuse_drag(self, j: int, live: bool):
        """Airborne: specific force xy ~ -D * body velocity xy (+ accel bias)."""
        c = self.cfg
        f, _ = self._mean_imu(j, self._drag_every)
        x = self.x
        R = quat_to_R(x.q)
        Rt = R.T
        vb = Rt @ x.v
        D = c.drag_coeff_xy
        Ht = -D * (Rt @ skew(x.v))
        ys, Hs = [], []
        for i in (0, 1):
            h = np.zeros(N)
            h[3:6] = -D * Rt[i]
            h[6:9] = Ht[i]
            h[9 + i] = 1.0
            Hs.append(h)
            ys.append(f[i] - (-D * vb[i] + x.ab[i]))
        ok = self._update(ys, Hs, [c.drag_accel ** 2] * 2, c.gate_drag, "drag", live)
        if ok and live:
            self.last_ok["drag"] = self.t

    def _fuse_static(self, j: int, live: bool):
        """On the ground: v = 0, specific force = -R^T g + ab, gyro = bias."""
        c = self.cfg
        f, w = self._mean_imu(j, self._static_every)
        x = self.x
        ys, Hs, Rs = [], [], []
        for i in range(3):
            h = np.zeros(N)
            h[3 + i] = 1.0
            ys.append(-x.v[i])
            Hs.append(h)
            Rs.append(c.zupt ** 2)
        ok = self._update(ys, Hs, Rs, 50.0, "zupt", live)
        x = self.x
        R = quat_to_R(x.q)
        fpred = -R.T @ GRAVITY_NED + x.ab
        Ht = -R.T @ skew(GRAVITY_NED)
        ys, Hs, Rs = [], [], []
        for i in range(3):
            h = np.zeros(N)
            h[6:9] = Ht[i]
            h[9 + i] = 1.0
            Hs.append(h)
            ys.append(f[i] - fpred[i])
            Rs.append(c.gravity_noise ** 2)
        self._update(ys, Hs, Rs, 6.0, "gravity", live)
        x = self.x
        ys, Hs, Rs = [], [], []
        for i in range(3):
            h = np.zeros(N)
            h[12 + i] = 1.0
            Hs.append(h)
            ys.append(w[i] - x.gb[i])
            Rs.append(c.static_gyro_noise ** 2)
        self._update(ys, Hs, Rs, 50.0, "static_gyro", live)
        if live:
            self.last_ok["static"] = self.t

    # ------------------------------------------------------------------ output
    def estimate(self) -> NavEstimate:
        est = NavEstimate()
        if not self.initialized:
            return est
        c = self.cfg
        x, P, t = self.x, self.P, self.t
        d = np.sqrt(np.maximum(np.diag(P), 0.0))
        est.t = t
        est.p = x.p.copy()
        est.v = x.v.copy()
        est.q = x.q.copy()
        gyro = self.last_imu.gyro if self.last_imu is not None else np.zeros(3)
        est.rate = gyro - x.gb
        est.sigma_pos_h = float(max(d[0], d[1]))
        est.sigma_vel_h = float(max(d[3], d[4]))
        est.sigma_alt = float(d[2])
        est.sigma_tilt = float(max(d[6], d[7]))
        est.sigma_yaw = float(d[8])
        est.diverged = self.diverged
        ok = not self.diverged
        est.att_valid = ok and est.sigma_tilt < c.max_sigma_tilt
        est.yaw_valid = ok and est.sigma_yaw < c.max_sigma_yaw
        vel_src = max(self.last_ok["gps_vel"], self.last_ok["drag"], self.last_ok["static"])
        est.vel_valid = (
            ok and est.sigma_vel_h < c.max_sigma_vel and (t - vel_src) < c.vel_valid_timeout_s
        )
        est.pos_valid = (
            ok
            and est.sigma_pos_h < c.max_sigma_pos_h
            and (t - self.last_ok["gps_pos"]) < c.pos_valid_timeout_s
            and est.vel_valid
        )
        est.alt_valid = ok and est.sigma_alt < c.max_sigma_alt and (t - self.last_ok["alt"]) < c.alt_valid_timeout_s
        est.range_height = self.range_height if (t - self.last_range_ok) < 0.5 else None
        return est

    def gps_validated(self, window: float) -> bool:
        """True when the latest GPS-velocity update was accepted and it happened within `window` s:
        i.e. an independent sensor currently agrees with the filter's velocity (and hence attitude)."""
        return self._gps_vel_accepted and (self.t - self.last_ok["gps_vel"]) < window

    # convenience for tests / logging
    @property
    def gyro_bias(self) -> np.ndarray:
        return self.x.gb.copy()

    @property
    def accel_bias(self) -> np.ndarray:
        return self.x.ab.copy()

    @property
    def baro_bias(self) -> float:
        return float(self.x.bb)
