"""Sensor sample types + tiny hardware-abstraction protocols.

All timestamps `t` are seconds on ONE monotonic clock (the IMU clock).  A sample's `t` is the
time the physical quantity was measured, not when it arrived -- the estimator handles latency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np


def _v3(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(3)


@dataclass
class ImuSample:
    t: float                 # end of the integration interval
    dt: float                # interval length
    gyro: np.ndarray         # rad/s, body FRD
    accel: np.ndarray        # m/s^2 specific force, body FRD (level & still -> [0, 0, -9.81])

    def __post_init__(self):
        self.gyro = _v3(self.gyro)
        self.accel = _v3(self.accel)


@dataclass
class BaroSample:
    t: float
    alt: float               # m, positive up, relative to any fixed reference


@dataclass
class GpsSample:
    t: float
    pos_ned: np.ndarray      # m from a fixed local origin
    vel_ned: np.ndarray      # m/s
    fix: bool = True         # 3D fix and enough satellites
    sigma_h: Optional[float] = None   # receiver-reported accuracy (m); None -> use config
    sigma_v: Optional[float] = None

    def __post_init__(self):
        self.pos_ned = _v3(self.pos_ned)
        self.vel_ned = _v3(self.vel_ned)


@dataclass
class MagSample:
    t: float
    field: np.ndarray        # body FRD, any consistent unit

    def __post_init__(self):
        self.field = _v3(self.field)


@dataclass
class RangeSample:
    t: float
    range_m: float           # along body -z (down-looking)
    valid: bool = True


@dataclass
class BatterySample:
    t: float
    volts: float
    frac: float              # remaining 0..1


@dataclass
class SensorFrame:
    """Everything that arrived during one loop tick."""
    imu: ImuSample
    baro: list = field(default_factory=list)
    gps: list = field(default_factory=list)
    mag: list = field(default_factory=list)
    range: list = field(default_factory=list)
    battery: list = field(default_factory=list)
    airborne: Optional[bool] = None    # only set by log replay (FlightCore ignores it): was the vehicle airborne?


@dataclass
class MotorOutput:
    """ESC commands 0..1.  `armed=False` means outputs must be zero (hardware layer enforces)."""
    t: float
    cmd: np.ndarray          # shape (4,)  ArduPilot-style order: FR, BL, FL, BR
    armed: bool


class MotorDriver(Protocol):
    def write(self, out: MotorOutput) -> None: ...


class Clock(Protocol):
    def now(self) -> float: ...
