"""Data passed between estimator, controller and supervisor."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .mathutil import quat_to_euler, quat_to_R, tilt_of, yaw_of


@dataclass
class NavEstimate:
    t: float = 0.0
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))      # NED, m
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))      # NED, m/s
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    rate: np.ndarray = field(default_factory=lambda: np.zeros(3))   # body rad/s, bias-corrected
    att_valid: bool = False      # roll/pitch/yaw usable for control
    yaw_valid: bool = False
    vel_valid: bool = False      # horizontal velocity usable
    pos_valid: bool = False      # horizontal position usable
    alt_valid: bool = False      # vertical position + velocity usable
    sigma_pos_h: float = math.inf
    sigma_vel_h: float = math.inf
    sigma_alt: float = math.inf
    sigma_tilt: float = math.inf
    sigma_yaw: float = math.inf
    diverged: bool = False
    range_height: Optional[float] = None    # last accepted rangefinder height, if fresh

    @property
    def altitude(self) -> float:
        return -float(self.p[2])

    @property
    def yaw(self) -> float:
        return yaw_of(self.q)

    @property
    def tilt(self) -> float:
        return tilt_of(self.q)

    @property
    def euler(self):
        return quat_to_euler(self.q)

    @property
    def R(self) -> np.ndarray:
        return quat_to_R(self.q)
