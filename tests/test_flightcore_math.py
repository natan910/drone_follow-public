import math
import unittest

import numpy as np

from flightcore.mathutil import (
    R_to_quat, cross3, quat_conj, quat_from_euler, quat_from_rotvec, quat_mul, quat_to_R,
    quat_to_euler, quat_to_rotvec, skew, tilt_of, wrap_pi, yaw_of,
)


class MathTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def rand_q(self):
        q = self.rng.standard_normal(4)
        return q / np.linalg.norm(q)

    def test_rotation_matrix_is_orthonormal_and_roundtrips(self):
        for _ in range(50):
            q = self.rand_q()
            R = quat_to_R(q)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)
            q2 = R_to_quat(R)
            self.assertAlmostEqual(abs(float(q @ q2)), 1.0, places=12)

    def test_quat_mul_composes_rotations(self):
        for _ in range(20):
            a, b = self.rand_q(), self.rand_q()
            np.testing.assert_allclose(quat_to_R(quat_mul(a, b)), quat_to_R(a) @ quat_to_R(b), atol=1e-12)
            np.testing.assert_allclose(quat_to_R(quat_conj(a)), quat_to_R(a).T, atol=1e-12)

    def test_euler_roundtrip_and_convention(self):
        for _ in range(50):
            r, p, y = self.rng.uniform(-1.4, 1.4), self.rng.uniform(-1.4, 1.4), self.rng.uniform(-3, 3)
            r2, p2, y2 = quat_to_euler(quat_from_euler(r, p, y))
            self.assertAlmostEqual(r, r2, places=9)
            self.assertAlmostEqual(p, p2, places=9)
            self.assertAlmostEqual(wrap_pi(y - y2), 0.0, places=9)
        # yaw +90 deg (clockwise from above): body x (forward) points EAST
        R = quat_to_R(quat_from_euler(0, 0, math.pi / 2))
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        # roll +90 deg: body y (right) points DOWN
        R = quat_to_R(quat_from_euler(math.pi / 2, 0, 0))
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, 1], atol=1e-12)
        # pitch +90 deg (nose up): body x points UP (-z world)
        R = quat_to_R(quat_from_euler(0, math.pi / 2, 0))
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 0, -1], atol=1e-12)

    def test_rotvec_exp_log(self):
        for _ in range(50):
            v = self.rng.uniform(-1.7, 1.7, 3)   # |v| < pi so the log is unique
            np.testing.assert_allclose(quat_to_rotvec(quat_from_rotvec(v)), v, atol=1e-10)
        # tiny rotations stay unit and accurate
        v = np.array([1e-10, -2e-10, 3e-10])
        q = quat_from_rotvec(v)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=14)

    def test_wrap_pi(self):
        for a, w in [(0, 0), (math.pi * 1.5, -math.pi / 2), (-math.pi * 1.5, math.pi / 2), (7.0, 7.0 - 2 * math.pi)]:
            self.assertAlmostEqual(wrap_pi(a), w, places=12)
        self.assertAlmostEqual(abs(wrap_pi(math.pi)), math.pi, places=12)

    def test_cross_and_skew(self):
        for _ in range(20):
            a, b = self.rng.standard_normal(3), self.rng.standard_normal(3)
            np.testing.assert_allclose(cross3(a, b), np.cross(a, b), atol=1e-14)
            np.testing.assert_allclose(skew(a) @ b, np.cross(a, b), atol=1e-14)

    def test_tilt_and_yaw_helpers(self):
        q = quat_from_euler(0.3, -0.2, 1.1)
        self.assertAlmostEqual(yaw_of(q), 1.1, places=9)
        R = quat_to_R(q)
        self.assertAlmostEqual(tilt_of(q), math.acos(R[2, 2]), places=9)


if __name__ == "__main__":
    unittest.main()
