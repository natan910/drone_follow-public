"""Fault injection: sensor faults and estimator failures must end safely.  FLIGHTCORE_SLOW=1 to run."""
import math
import os
import unittest

import numpy as np

from flightcore.mathutil import wrap_pi
from flightcore.sim.harness import ClosedLoop

SLOW = unittest.skipUnless(os.environ.get("FLIGHTCORE_SLOW"), "set FLIGHTCORE_SLOW=1 for fault injection")


def hovering(seed, alt=3.0, **kw) -> ClosedLoop:
    s = ClosedLoop(seed=seed, **kw)
    assert s.arm_and_takeoff(alt)
    s.run(3.0)
    return s


def yaw_err_deg(s):
    return math.degrees(abs(wrap_pi(s.core.nav.yaw - s.truth.euler()[2])))


@SLOW
class SensorFaults(unittest.TestCase):
    def watch(self, s, secs):
        worst_dist, worst_tilt = 0.0, 0.0
        end = s.t + secs
        while s.t < end:
            s.run(0.2)
            r, p, _ = s.truth.euler()
            worst_dist = max(worst_dist, float(np.linalg.norm(s.truth.p[:2])))
            worst_tilt = max(worst_tilt, math.degrees(max(abs(r), abs(p))))
        return worst_dist, worst_tilt

    def test_gyro_bias_step(self):
        """8 deg/s of sudden gyro bias used to make the filter reject good GPS and the drone fly away."""
        s = hovering(21)
        s.world.sensors.gyro_bias += np.array([0.12, -0.08, 0.05])
        dist, tilt = self.watch(s, 14.0)
        self.assertLess(dist, 4.0)
        self.assertLess(tilt, 15.0)
        self.assertEqual(s.core.mode.value, "hold")
        self.assertLess(yaw_err_deg(s), 6.0)
        err = np.abs(s.core.est.x.gb - s.world.sensors.gyro_bias)
        self.assertLess(float(err[:2].max()), 0.01)     # roll/pitch bias: seen through GPS velocity
        self.assertLess(float(err[2]), 0.02)            # yaw bias: only the (slow) compass sees it

    def test_accelerometer_bias_step(self):
        s = hovering(21)
        s.world.sensors.accel_bias += np.array([0.9, -0.7, 0.4])
        dist, tilt = self.watch(s, 14.0)
        self.assertLess(dist, 2.0)
        self.assertLess(tilt, 10.0)
        self.assertAlmostEqual(s.altitude, 3.0, delta=0.3)

    def test_vibration_burst(self):
        s = hovering(22)
        s.world.sensors.s.vibration_std = 2.5           # 17x normal accelerometer noise
        dist, tilt = self.watch(s, 3.0)
        s.world.sensors.s.vibration_std = 0.15
        d2, t2 = self.watch(s, 4.0)
        self.assertLess(max(dist, d2), 1.5)
        self.assertLess(max(tilt, t2), 10.0)
        self.assertEqual(s.core.mode.value, "hold")

    def test_barometer_drift_is_absorbed_by_the_bias_state(self):
        s = hovering(23, alt=2.0)
        lo = hi = s.altitude
        for _ in range(60):                              # +0.3 m/s of baro drift for 6 s: 1.8 m
            s.world.sensors.baro_bias += 0.03
            s.run(0.1)
            lo, hi = min(lo, s.altitude), max(hi, s.altitude)
        s.run(4.0)
        self.assertGreater(lo, 1.6)
        self.assertLess(hi, 2.4)
        self.assertGreater(s.core.est.baro_bias, 1.0)    # the filter noticed

    def test_compass_yaw_jump_is_recovered_by_a_yaw_reset(self):
        s = hovering(24)
        s.core.est.perturb_attitude([0.0, 0.0, math.radians(60)])
        s.run(9.0)
        self.assertEqual(s.core.est.yaw_reset_seq, 1)
        self.assertIn("yaw_reset", [e[1] for e in s.core.sup.events])
        self.assertLess(yaw_err_deg(s), 3.0)
        self.assertLess(float(np.linalg.norm(s.truth.p[:2])), 2.0)
        self.assertEqual(s.core.mode.value, "hold")


@SLOW
class EstimatorFailure(unittest.TestCase):
    def test_grossly_wrong_attitude_recovers_or_lands_but_never_crashes(self):
        for mag in (40, 60):
            with self.subTest(deg=mag):
                s = hovering(13, alt=5.0)
                s.core.est.perturb_attitude([math.radians(mag), 0.0, 0.0])
                s.run_until(lambda c: not c.core.armed, 40.0)
                self.assertFalse(s.truth.crashed)
                self.assertLess(s.truth.max_impact, 3.5)
                self.assertLess(s.max_tilt, math.radians(100.0))

    def test_backup_attitude_takes_over_when_the_eskf_fails(self):
        s = hovering(13, alt=4.0)
        s.core.est.perturb_attitude([math.radians(60), 0.0, 0.0])
        s.run_until(lambda c: c.core.sup.reason == "estimator_failure", 5.0)
        self.assertEqual(s.core.sup.reason, "estimator_failure")
        self.assertIn("emergency_descent", [e[1] for e in s.core.sup.events])
        self.assertTrue(s.run_until(lambda c: not c.core.armed, 30.0))
        self.assertFalse(s.truth.crashed)

    def test_nan_in_the_filter_gives_a_safe_level_descent(self):
        s = hovering(31, alt=4.0, wind=(1.5, 0.5, 0.0), gust_std=0.5)
        s.core.est.x.q[:] = np.nan
        s.run(0.2)
        self.assertTrue(s.core.est.diverged)
        self.assertEqual(s.core.sup.reason, "estimator_failure")
        self.assertTrue(s.run_until(lambda c: not c.core.armed, 30.0))
        self.assertFalse(s.truth.crashed)
        self.assertLess(s.truth.max_impact, 2.0)
        self.assertLess(math.degrees(s.max_tilt), 25.0)
        self.assertLess(float(np.linalg.norm(s.truth.p[:2])), 15.0)

    def test_backup_filter_stays_close_to_the_eskf_in_aggressive_flight(self):
        s = ClosedLoop(seed=13, wind=(3, 0, 0), gust_std=1.0)
        self.assertTrue(s.arm_and_takeoff(2.0))
        worst = 0.0
        for _ in range(3):
            for vx, wz in ((4.0, 0.0), (-4.0, math.radians(60))):
                for _ in range(25):
                    s.core.set_velocity_body(vx, 0, 0, wz)
                    s.run(0.1)
                    worst = max(worst, s.core.att_disagreement_deg)
        self.assertLess(worst, 12.0)                    # alarm thresholds are 20 deg (1 s) / 30 deg (0.2 s)
        self.assertNotIn("emergency_descent", [e[1] for e in s.core.sup.events])


if __name__ == "__main__":
    unittest.main()
