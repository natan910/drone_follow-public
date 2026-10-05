import math
import unittest

import numpy as np

from flightcore.config import EstimatorConfig
from flightcore.estimation import Eskf, Mahony
from flightcore.hal import BaroSample, GpsSample, ImuSample, MagSample, RangeSample
from flightcore.mathutil import G, quat_from_euler, quat_to_R, quat_to_euler, wrap_pi
from flightcore.sim import SimWorld

DT = 0.002
B_WORLD = 0.45 * np.array([math.cos(math.radians(64)), 0.0, math.sin(math.radians(64))])


def imu_at(k: int, wobble: bool = True) -> ImuSample:
    """Deterministic 'hovering with a little motion' IMU stream (level, forward-facing)."""
    t = k * DT
    gyro = np.array([0.02 * math.sin(3 * t), 0.015 * math.cos(2 * t), 0.01 * math.sin(t)]) if wobble else np.zeros(3)
    accel = np.array([0.05 * math.sin(2 * t), 0.04 * math.cos(3 * t), -G])
    return ImuSample(t=t, dt=DT, gyro=gyro, accel=accel)


def fresh(cfg=None, yaw=0.0) -> Eskf:
    e = Eskf(cfg or EstimatorConfig())
    e.initialize(quat_from_euler(0, 0, yaw), p=np.zeros(3), t=0.0, dt=DT, have_mag=True, have_gps=True,
                 mag_norm=0.45)
    e.set_static(False)
    return e


def gps(t, pos=(0, 0, 0), vel=(0, 0, 0)) -> GpsSample:
    return GpsSample(t=t, pos_ned=np.array(pos, float), vel_ned=np.array(vel, float), fix=True)


def mag_for_yaw(t, yaw) -> MagSample:
    return MagSample(t=t, field=quat_to_R(quat_from_euler(0, 0, yaw)).T @ B_WORLD)


class InitTests(unittest.TestCase):
    def test_static_alignment_from_a_real_sensor_stream(self):
        w = SimWorld(seed=3, yaw0=math.radians(40))
        e = Eskf()
        n = 0
        while not e.feed_init(*self._unpack(w.step(np.zeros(4)))):
            n += 1
            self.assertLess(n, 1200)
        self.assertGreaterEqual(n * DT, 0.9)                       # needs ~1 s of stillness
        roll, pitch, yaw = quat_to_euler(e.x.q)
        self.assertLess(abs(math.degrees(roll)), 1.0)
        self.assertLess(abs(math.degrees(pitch)), 1.0)
        self.assertLess(abs(math.degrees(wrap_pi(yaw - math.radians(40)))), 2.0)
        np.testing.assert_allclose(e.x.gb, w.sensors.gyro_bias, atol=0.004)

    @staticmethod
    def _unpack(f):
        return f.imu, f.baro, f.mag, f.gps

    def test_no_init_while_moving(self):
        e = Eskf()
        for k in range(1500):
            imu = ImuSample(t=k * DT, dt=DT, gyro=[0.6 * math.sin(20 * k * DT), 0.4, 0.0], accel=[0, 0, -G])
            e.feed_init(imu, [BaroSample(t=k * DT, alt=0.0)] if k % 20 == 0 else [], [], [])
        self.assertFalse(e.initialized)

    def test_no_init_without_baro(self):
        e = Eskf()
        for k in range(1000):
            e.feed_init(imu_at(k, wobble=False))
        self.assertFalse(e.initialized)

    def test_estimate_is_invalid_before_init_and_valid_after(self):
        e = Eskf()
        n = e.estimate()
        self.assertFalse(any([n.att_valid, n.yaw_valid, n.vel_valid, n.pos_valid, n.alt_valid]))
        w = SimWorld(seed=3)
        while not e.feed_init(*self._unpack(w.step(np.zeros(4)))):
            pass
        e.set_static(True)
        for _ in range(500):
            f = w.step(np.zeros(4))
            e.step(f.imu, list(f.baro) + list(f.gps) + list(f.mag) + list(f.range))
        n = e.estimate()
        self.assertTrue(n.att_valid and n.yaw_valid and n.alt_valid and n.vel_valid and n.pos_valid)
        self.assertLess(float(np.linalg.norm(n.v)), 0.02)          # ZUPT keeps it still
        self.assertLess(math.degrees(n.tilt), 0.5)


class DelayedFusionTests(unittest.TestCase):
    """The replay must be exact: same result whether a measurement arrives on time or late."""

    def run_filter(self, deliveries, ticks):
        e = fresh()
        for k in range(1, ticks + 1):
            e.step(imu_at(k), deliveries.get(k, []))
        return e

    def assert_same(self, a: Eskf, b: Eskf):
        np.testing.assert_allclose(a.x.p, b.x.p, atol=1e-9)
        np.testing.assert_allclose(a.x.v, b.x.v, atol=1e-9)
        np.testing.assert_allclose(a.x.q, b.x.q, atol=1e-9)
        np.testing.assert_allclose(a.x.ab, b.x.ab, atol=1e-9)
        np.testing.assert_allclose(a.x.gb, b.x.gb, atol=1e-9)
        np.testing.assert_allclose(a.P, b.P, atol=1e-9)

    def test_late_gps_gives_the_same_state_as_on_time_gps(self):
        g = gps(200 * DT, pos=(0.9, -0.6, -0.3), vel=(0.15, 0.0, 0.05))
        on_time = self.run_filter({200: [g]}, 300)
        late = self.run_filter({260: [g]}, 300)
        self.assert_same(on_time, late)
        self.assertGreater(abs(on_time.x.p[0]), 0.1)                # it did correct something

    def test_out_of_order_arrival_gives_the_same_state(self):
        baro = BaroSample(t=150 * DT, alt=0.4)
        g = gps(200 * DT, pos=(0.5, 0.2, -0.2), vel=(0.0, 0.1, 0.0))
        in_order = self.run_filter({150: [baro], 200: [g]}, 300)
        swapped = self.run_filter({200: [g], 230: [baro]}, 300)
        self.assert_same(in_order, swapped)

    def test_stale_measurement_is_dropped(self):
        e, twin = fresh(), fresh()
        for k in range(1, 400):
            e.step(imu_at(k))
            twin.step(imu_at(k))
        e.step(imu_at(400), [gps(0.05, pos=(30, 30, 0))])           # 0.75 s old, window is 0.3 s
        twin.step(imu_at(400))
        np.testing.assert_allclose(e.x.p, twin.x.p, atol=1e-12)
        np.testing.assert_allclose(e.P, twin.P, atol=1e-12)
        self.assertEqual(e.stats["gps_stale"][1], 1)

    def test_replay_cost_is_bounded_by_the_window(self):
        e = fresh()
        for k in range(1, 1000):
            e.step(imu_at(k))
        self.assertLessEqual(len(e.slots), int(e.cfg.delay_window_s / DT) + 3)


class GatingAndResetTests(unittest.TestCase):
    def settle(self, e, ticks=500):
        for k in range(1, ticks + 1):
            e.step(imu_at(k), [gps(k * DT)] if k % 50 == 0 else [])
        return ticks

    def test_gps_outlier_is_rejected(self):
        e = fresh()
        k0 = self.settle(e)
        before = e.x.p.copy()
        e.step(imu_at(k0 + 1), [gps((k0 + 1) * DT, pos=(60.0, -40.0, 0.0))])
        np.testing.assert_allclose(e.x.p, before, atol=1e-3)
        self.assertEqual(e.stats["gps_pos_h"][1], 1)
        self.assertEqual(e.reset_seq, 0)

    def test_gps_accepts_small_consistent_corrections(self):
        e = fresh()
        k0 = self.settle(e)
        e.step(imu_at(k0 + 1), [gps((k0 + 1) * DT, pos=(0.8, 0.0, 0.0))])
        self.assertGreater(float(e.x.p[0]), 0.05)

    def test_persistent_gps_offset_resets_position_after_timeout(self):
        e = fresh()
        k = self.settle(e)
        deltas = []
        for i in range(1, 3200):                                   # 6.4 s of a 50 m offset
            k += 1
            meas = [gps(k * DT, pos=(50.0, 0.0, 0.0))] if i % 50 == 0 else []
            e.step(imu_at(k), meas)
        self.assertEqual(e.reset_seq, 1)
        self.assertAlmostEqual(float(e.reset_delta[0]), 50.0, delta=2.0)
        self.assertAlmostEqual(float(e.x.p[0]), 50.0, delta=2.0)

    def test_persistent_disagreement_inflates_covariance_to_unstick_the_filter(self):
        e = fresh()
        k = self.settle(e)
        sig0 = math.sqrt(e.P[3, 3])
        for i in range(1, 1100):                                    # > inflate_after_reject_s of disagreement
            k += 1
            meas = [gps(k * DT, pos=(0, 0, 0), vel=(6.0, 0.0, 0.0))] if i % 50 == 0 else []
            e.step(imu_at(k), meas)
        self.assertGreaterEqual(e.stats.get("inflate", [0, 0])[0], 1)

    def test_mag_with_wrong_field_strength_is_ignored(self):
        e, twin = fresh(), fresh()
        for k in range(1, 800):
            m = MagSample(t=k * DT, field=np.array([1.5, 0.5, 0.2]))    # norm far from 0.45
            e.step(imu_at(k), [m] if k % 20 == 0 else [])
            twin.step(imu_at(k))
        np.testing.assert_allclose(e.x.q, twin.x.q, atol=1e-12)         # mag had no effect at all
        self.assertEqual(e.yaw_reset_seq, 0)

    def test_persistently_rejected_compass_resets_yaw(self):
        e = fresh()
        k = 0
        for i in range(1, 3300):                                   # 6.6 s; compass says 60 deg, filter says 0
            k += 1
            e.step(imu_at(k), [mag_for_yaw(k * DT, math.radians(60))] if k % 20 == 0 else [])
        self.assertEqual(e.yaw_reset_seq, 1)
        self.assertAlmostEqual(e.yaw_reset_delta, math.radians(60), delta=math.radians(3))
        self.assertLess(abs(wrap_pi(quat_to_euler(e.x.q)[2] - math.radians(60))), math.radians(3))

    def test_small_compass_error_is_corrected_smoothly_not_reset(self):
        e = fresh()
        for k in range(1, 3000):
            e.step(imu_at(k), [mag_for_yaw(k * DT, math.radians(5))] if k % 20 == 0 else [])
        self.assertEqual(e.yaw_reset_seq, 0)
        self.assertLess(abs(wrap_pi(quat_to_euler(e.x.q)[2] - math.radians(5))), math.radians(2))


class RangeAndDivergenceTests(unittest.TestCase):
    def feed(self, e, samples, ticks=400):
        for k in range(1, ticks + 1):
            e.step(imu_at(k), [s(k * DT) for s in samples] if k % 25 == 0 else [])

    def test_range_pulls_altitude_toward_the_measurement(self):
        e = fresh()
        e.x.p[2] = -1.2                                             # believes 1.2 m up
        self.feed(e, [lambda t: RangeSample(t=t, range_m=2.0, valid=True)], ticks=600)
        self.assertAlmostEqual(-float(e.x.p[2]), 2.0, delta=0.15)

    def test_range_ignored_when_invalid_out_of_range_or_tilted(self):
        for rs in (RangeSample(t=0, range_m=2.0, valid=False), RangeSample(t=0, range_m=9.0, valid=True),
                   RangeSample(t=0, range_m=0.05, valid=True)):
            e = fresh()
            self.feed(e, [lambda t, rs=rs: RangeSample(t=t, range_m=rs.range_m, valid=rs.valid)])
            self.assertNotIn("range", e.stats)
        e = fresh(yaw=0.0)
        e.x.q = quat_from_euler(math.radians(40), 0, 0)             # too tilted for a down-looking range
        self.feed(e, [lambda t: RangeSample(t=t, range_m=2.0, valid=True)])
        self.assertNotIn("range", e.stats)

    def test_range_step_from_a_person_underneath_is_rejected(self):
        e = fresh()
        e.x.p[2] = -2.0
        e.P[2, 2] = 0.01 ** 2
        self.feed(e, [lambda t: RangeSample(t=t, range_m=0.3, valid=True)], ticks=400)   # sudden 1.7 m step
        self.assertGreater(-float(e.x.p[2]), 1.8)
        self.assertGreaterEqual(e.stats["range"][1], 1)

    def test_nan_marks_the_filter_diverged(self):
        e = fresh()
        for k in range(1, 100):
            e.step(imu_at(k))
        self.assertFalse(e.diverged)
        e.x.v[0] = float("nan")
        e.step(imu_at(100))
        self.assertTrue(e.diverged)
        self.assertFalse(e.estimate().att_valid)

    def test_perturb_attitude_helper_survives_replay(self):
        e = fresh()
        for k in range(1, 300):
            e.step(imu_at(k))
        e.perturb_attitude([0.3, 0.0, 0.0])
        for k in range(300, 400):
            e.step(imu_at(k), [gps(k * DT - 0.1)] if k % 50 == 0 else [])
        self.assertGreater(abs(quat_to_euler(e.x.q)[0]), 0.2)


class MahonyTests(unittest.TestCase):
    def test_converges_to_gravity_from_a_tilted_start(self):
        m = Mahony(kp=1.5, ki=0.0)
        m.init_from(quat_from_euler(math.radians(25), math.radians(-15), 0.0))
        for _ in range(5000):
            m.update(np.zeros(3), np.array([0.0, 0.0, -G]), 0.002)
        r, p, _ = quat_to_euler(m.q)
        self.assertLess(abs(math.degrees(r)), 0.5)
        self.assertLess(abs(math.degrees(p)), 0.5)

    def test_tracks_a_rotation_from_the_gyro(self):
        m = Mahony(kp=0.5, ki=0.0)
        m.init_from(np.array([1.0, 0, 0, 0]))
        for _ in range(500):                                         # 1 s at 0.5 rad/s yaw
            m.update(np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, -G]), 0.002)
        self.assertAlmostEqual(quat_to_euler(m.q)[2], 0.5, delta=0.01)

    def test_ignores_accelerometer_when_it_is_not_measuring_gravity(self):
        m = Mahony(kp=2.0, ki=0.0)
        m.init_from(np.array([1.0, 0, 0, 0]))
        for _ in range(500):
            m.update(np.zeros(3), np.array([0.0, 0.0, -3 * G]), 0.002)     # 3 g: crash / vibration
        self.assertLess(abs(quat_to_euler(m.q)[0]), 1e-9)


if __name__ == "__main__":
    unittest.main()
