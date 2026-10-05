"""Attitude P controller on SO(3) (geometric error), separate per-axis gains.

Desired attitude is built from a desired thrust direction (body -z axis) and a heading:
    z_d = thrust direction expressed as body z-axis (unit, world frame)
    x_c = [cos psi, sin psi, 0]
    y_d = z_d x x_c / |.|,  x_d = y_d x z_d
Error  e_R = 1/2 vee(R_d^T R - R^T R_d)   (body frame);  rate setpoint = -Kp * e_R + yaw-rate feed-forward.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import ControlConfig
from ..mathutil import cross3, vee


def desired_rotation(z_d: np.ndarray, yaw: float) -> np.ndarray:
    z_d = np.asarray(z_d, dtype=float)
    x_c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    y_d = cross3(z_d, x_c)
    n = np.linalg.norm(y_d)
    if n < 1e-6:                      # thrust axis horizontal & along heading: pick any consistent y
        y_d = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    else:
        y_d = y_d / n
    x_d = cross3(y_d, z_d)
    return np.column_stack([x_d, y_d, z_d])


class AttitudeController:
    def __init__(self, cfg: ControlConfig):
        self.kp = np.array(cfg.att_kp)
        self.max_rate = np.array(cfg.max_rate)

    def update(self, R: np.ndarray, z_d: np.ndarray, yaw_sp: float, yaw_rate_ff: float = 0.0):
        """Return (rate_setpoint_body, R_desired)."""
        Rd = desired_rotation(z_d, yaw_sp)
        e = 0.5 * vee(Rd.T @ R - R.T @ Rd)
        rate = -self.kp * e + R.T @ np.array([0.0, 0.0, yaw_rate_ff])
        rate = np.clip(rate, -self.max_rate, self.max_rate)
        return rate, Rd
