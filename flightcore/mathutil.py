"""Small rotation / quaternion helpers.

Conventions (whole package):
  world frame  NED  (x north, y east, z down)
  body frame   FRD  (x forward, y right, z down)
  quaternion   Hamilton, scalar first [w, x, y, z], rotates body -> world
  yaw          positive clockwise seen from above (NED z is down)
"""
from __future__ import annotations

import math

import numpy as np

G = 9.80665  # m/s^2
GRAVITY_NED = np.array([0.0, 0.0, G])
E3 = np.array([0.0, 0.0, 1.0])


def wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def cross3(a, b) -> np.ndarray:
    """3-vector cross product (np.cross is ~10x slower for single vectors)."""
    return np.array(
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    )


def skew(v) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q) -> np.ndarray:
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.asarray(q, dtype=float) / n


def quat_from_rotvec(v) -> np.ndarray:
    """Exponential map: rotation vector (rad) -> quaternion."""
    x, y, z = v
    ang = math.sqrt(x * x + y * y + z * z)
    if ang < 1e-9:
        # first-order, already unit to 1e-18
        return np.array([1.0, 0.5 * x, 0.5 * y, 0.5 * z])
    s = math.sin(0.5 * ang) / ang
    return np.array([math.cos(0.5 * ang), x * s, y * s, z * s])


def quat_to_rotvec(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q[0] < 0.0:
        q = -q
    s = math.sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if s < 1e-12:
        return 2.0 * q[1:4]
    ang = 2.0 * math.atan2(s, q[0])
    return q[1:4] * (ang / s)


def quat_to_R(q) -> np.ndarray:
    """Rotation matrix body -> world."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def R_to_quat(R) -> np.ndarray:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = quat_normalize(q)
    return -q if q[0] < 0.0 else q


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX (yaw, pitch, roll) Euler angles -> quaternion."""
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_to_euler(q) -> tuple[float, float, float]:
    """quaternion -> (roll, pitch, yaw), ZYX."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = clamp(2 * (w * y - z * x), -1.0, 1.0)
    pitch = math.asin(sp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_of(q) -> float:
    """Angle (rad) between body z and world down."""
    w, x, y, z = q
    c = 1 - 2 * (x * x + y * y)  # R[2,2]
    return math.acos(clamp(c, -1.0, 1.0))


def rot_z(psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def vee(M) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]])
