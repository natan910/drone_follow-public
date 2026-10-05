"""Sensor models for the simulator: noise, bias random walk, sample rates, latency, dropouts."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..hal import BaroSample, BatterySample, GpsSample, ImuSample, MagSample, RangeSample, SensorFrame
from ..mathutil import quat_to_R


@dataclass
class SensorSpec:
    # IMU
    gyro_noise: float = 3e-4          # rad/s/sqrt(Hz)
    accel_noise: float = 2e-3         # m/s^2/sqrt(Hz)
    vibration_std: float = 0.15       # m/s^2, white, per sample (motor vibration)
    gyro_bias0: float = 0.008         # rad/s, 1-sigma initial turn-on bias
    accel_bias0: float = 0.08         # m/s^2
    gyro_bias_walk: float = 5e-5      # rad/s/sqrt(s)
    accel_bias_walk: float = 5e-4     # m/s^2/sqrt(s)
    # baro
    baro_hz: float = 25.0
    baro_noise: float = 0.25
    baro_walk: float = 0.01           # m/sqrt(s)
    baro_latency: float = 0.03
    # gps
    gps_hz: float = 10.0
    gps_pos_noise_h: float = 0.5
    gps_pos_noise_v: float = 0.9
    gps_vel_noise_h: float = 0.08
    gps_vel_noise_v: float = 0.12
    gps_pos_walk: float = 0.05        # m/sqrt(s) slow error (multipath / iono)
    gps_latency: float = 0.12
    gps_outages: list = field(default_factory=list)              # [(t0, t1)] no data
    gps_glitches: list = field(default_factory=list)             # [(t0, t1, [n, e, d])] position jump
    # mag
    mag_hz: float = 50.0
    mag_noise: float = 0.004
    mag_declination: float = 0.0
    mag_norm: float = 0.45
    mag_inclination: float = math.radians(64.0)
    mag_hard_iron: tuple = (0.0, 0.0, 0.0)
    mag_latency: float = 0.01
    mag_disturbances: list = field(default_factory=list)        # [(t0, t1, [x, y, z] body offset)]
    # range finder (TF-Luna-like)
    range_hz: float = 50.0
    range_noise: float = 0.02
    range_min: float = 0.2
    range_max: float = 8.0
    range_latency: float = 0.01
    terrain: Optional[Callable[[float, float], float]] = None    # ground height (m, up) under (n, e)
    # battery
    battery_hz: float = 10.0
    battery_drain_per_s: float = 0.0005     # fraction / s while motors spinning
    battery_start: float = 1.0


class SensorSuite:
    def __init__(self, spec: SensorSpec, dt: float, rng: np.random.Generator):
        self.s = spec
        self.dt = dt
        self.rng = rng
        self.gyro_bias = spec.gyro_bias0 * rng.standard_normal(3)
        self.accel_bias = spec.accel_bias0 * rng.standard_normal(3)
        self.baro_bias = 0.0
        self.gps_err = np.zeros(3)
        self._next = {"baro": 0.0, "gps": 0.0, "mag": 0.0, "range": 0.0, "batt": 0.0}
        self._q: dict[str, deque] = {k: deque() for k in ("baro", "gps", "mag", "range")}
        self.battery = spec.battery_start
        self._B = self._field_world()

    def _field_world(self):
        s = self.s
        h = s.mag_norm * math.cos(s.mag_inclination)
        dn = s.mag_norm * math.sin(s.mag_inclination)
        return np.array([h * math.cos(s.mag_declination), h * math.sin(s.mag_declination), dn])

    @staticmethod
    def _in(t, windows):
        return any(w[0] <= t < w[1] for w in windows)

    def _due(self, name: str, hz: float, t: float) -> bool:
        if t + 1e-9 >= self._next[name]:
            self._next[name] = max(self._next[name] + 1.0 / hz, t + 0.5 / hz)
            return True
        return False

    def sample(self, plant, t: float) -> SensorFrame:
        s, dt, rng = self.s, self.dt, self.rng
        # slow error processes
        self.gyro_bias += s.gyro_bias_walk * math.sqrt(dt) * rng.standard_normal(3)
        self.accel_bias += s.accel_bias_walk * math.sqrt(dt) * rng.standard_normal(3)
        self.baro_bias += s.baro_walk * math.sqrt(dt) * rng.standard_normal()
        self.gps_err += s.gps_pos_walk * math.sqrt(dt) * rng.standard_normal(3)

        gyro = plant.w + self.gyro_bias + s.gyro_noise / math.sqrt(dt) * rng.standard_normal(3)
        accel = (
            plant.f_body
            + self.accel_bias
            + s.accel_noise / math.sqrt(dt) * rng.standard_normal(3)
            + s.vibration_std * rng.standard_normal(3)
        )
        frame = SensorFrame(imu=ImuSample(t=t, dt=dt, gyro=gyro, accel=accel))
        R = quat_to_R(plant.q)

        if self._due("baro", s.baro_hz, t):
            alt = plant.altitude + self.baro_bias + s.baro_noise * rng.standard_normal()
            self._q["baro"].append((t + s.baro_latency, BaroSample(t=t, alt=alt)))

        if self._due("gps", s.gps_hz, t):
            if not self._in(t, s.gps_outages):
                pos = plant.p.copy() + self.gps_err
                pos[0:2] += s.gps_pos_noise_h * rng.standard_normal(2)
                pos[2] += s.gps_pos_noise_v * rng.standard_normal()
                for (t0, t1, off) in s.gps_glitches:
                    if t0 <= t < t1:
                        pos = pos + np.asarray(off, dtype=float)
                vel = plant.v.copy()
                vel[0:2] += s.gps_vel_noise_h * rng.standard_normal(2)
                vel[2] += s.gps_vel_noise_v * rng.standard_normal()
                self._q["gps"].append(
                    (t + s.gps_latency, GpsSample(t=t, pos_ned=pos, vel_ned=vel, fix=True,
                                                  sigma_h=s.gps_pos_noise_h, sigma_v=s.gps_pos_noise_v))
                )

        if self._due("mag", s.mag_hz, t):
            B = self._B.copy()
            m = R.T @ B + np.asarray(s.mag_hard_iron) + s.mag_noise * rng.standard_normal(3)
            for (t0, t1, off) in s.mag_disturbances:
                if t0 <= t < t1:
                    m = m + np.asarray(off, dtype=float)
            self._q["mag"].append((t + s.mag_latency, MagSample(t=t, field=m)))

        if self._due("range", s.range_hz, t):
            ground = s.terrain(plant.p[0], plant.p[1]) if s.terrain else 0.0
            height = plant.altitude - ground
            r33 = R[2, 2]
            if r33 > 0.2:
                r = max(height, 0.0) / r33 + s.range_noise * rng.standard_normal()
                valid = s.range_min <= r <= s.range_max     # below min: sensor reports a small, invalid value
            else:
                r, valid = 0.0, False
            self._q["range"].append((t + s.range_latency, RangeSample(t=t, range_m=max(r, 0.0), valid=valid)))

        if self._due("batt", s.battery_hz, t):
            spinning = float(np.sum(plant.m)) > 0.05
            self.battery = max(0.0, self.battery - (s.battery_drain_per_s * (1.0 / s.battery_hz) if spinning else 0.0))
            volts = 4 * 3.7 * (0.88 + 0.12 * self.battery)
            frame.battery.append(BatterySample(t=t, volts=volts, frac=self.battery))

        for name, dst in (("baro", frame.baro), ("gps", frame.gps), ("mag", frame.mag), ("range", frame.range)):
            q = self._q[name]
            while q and q[0][0] <= t + 1e-9:
                dst.append(q.popleft()[1])
        return frame
