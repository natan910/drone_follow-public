"""Independent backup attitude filter (Mahony complementary filter, gyro + accelerometer).

Not used for control while the ESKF is healthy.  It is a cross-check and the fallback source when
the ESKF diverges.  Limitation: for a multirotor the accelerometer reads mostly thrust (body -z), so
it pulls tilt toward 0 during sustained horizontal acceleration.  To stop that producing false alarms
the filter is *slaved* to the ESKF while the two agree (`nudge_toward`, slow, only when they are within
a few degrees): it then behaves as a gyro-integrating shadow that keeps the last good attitude if the
ESKF suddenly fails.  Good for level-descent recovery, not for aggressive flight on its own.
Yaw is dead-reckoned (gyro only).
"""
from __future__ import annotations

import numpy as np

from ..mathutil import G, quat_conj, quat_from_rotvec, quat_mul, quat_normalize, quat_to_R, quat_to_rotvec


class Mahony:
    def __init__(self, kp: float = 0.3, ki: float = 0.01):
        self.kp = kp
        self.ki = ki
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.bias = np.zeros(3)
        self.ready = False

    def init_from(self, q, gyro_bias=None):
        self.q = np.array(q, dtype=float)
        self.bias = np.zeros(3) if gyro_bias is None else np.array(gyro_bias, dtype=float)
        self.ready = True

    def nudge_toward(self, q_ref, rate: float, dt: float) -> None:
        """Rotate the estimate toward q_ref at `rate` (1/s) -- world-frame error, includes yaw."""
        e = quat_to_rotvec(quat_mul(np.asarray(q_ref, dtype=float), quat_conj(self.q)))
        self.q = quat_normalize(quat_mul(quat_from_rotvec(min(rate * dt, 1.0) * e), self.q))

    def update(self, gyro, accel, dt: float) -> np.ndarray:
        if not self.ready:
            return self.q
        gyro = np.asarray(gyro, dtype=float)
        accel = np.asarray(accel, dtype=float)
        R = quat_to_R(self.q)
        w = gyro - self.bias
        n = float(np.linalg.norm(accel))
        if 0.7 * G < n < 1.3 * G:
            f_meas = accel / n
            f_pred = -R[2, :]                 # -R^T e3 : expected specific-force direction in body
            e = np.cross(f_meas, f_pred)
            self.bias = self.bias - self.ki * e * dt
            self.bias = np.clip(self.bias, -0.35, 0.35)
            w = w + self.kp * e
        self.q = quat_normalize(quat_mul(self.q, quat_from_rotvec(w * dt)))
        return self.q
