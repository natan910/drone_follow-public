import math
import unittest
from dataclasses import replace

import numpy as np

from flightcore.config import VehicleParams
from flightcore.mathutil import G, quat_from_euler
from flightcore.sim import Plant, SensorSpec, SimWorld
from flightcore.sim.dynamics import WindModel

CLEAN = SensorSpec(accel_noise=0, gyro_noise=0, vibration_std=0, gyro_bias0=0, accel_bias0=0,
                   gyro_bias_walk=0, accel_bias_walk=0, baro_noise=0, baro_walk=0, gps_pos_noise_h=0,
                   gps_pos_noise_v=0, gps_vel_noise_h=0, gps_vel_noise_v=0, gps_pos_walk=0, mag_noise=0,
                   range_noise=0)


def hover_cmd(veh: VehicleParams) -> float:
    e, x = veh.thrust_expo, veh.hover_frac
    return (-(1 - e) + math.sqrt((1 - e) ** 2 + 4 * e * x)) / (2 * e)


class PlantTests(unittest.TestCase):
    def setUp(self):
        self.veh = VehicleParams()

    def airborne_plant(self, alt=50.0, **kw):
        pl = Plant(self.veh, **kw)
        pl.on_ground = False
        pl.p[:] = [0, 0, -alt]
        pl.m[:] = self.veh.hover_frac
        return pl

    def test_sits_on_the_ground_below_liftoff_thrust(self):
        pl = Plant(self.veh)
        for _ in range(500):
            pl.step(np.full(4, 0.9 * hover_cmd(self.veh)), 0.002)
        self.assertTrue(pl.on_ground)
        self.assertEqual(float(pl.p[2]), 0.0)
        np.testing.assert_allclose(pl.f_body, [0, 0, -G], atol=1e-9)

    def test_lifts_off_above_hover_thrust(self):
        pl = Plant(self.veh)
        for _ in range(1000):
            pl.step(np.full(4, 1.1 * hover_cmd(self.veh)), 0.002)
        self.assertFalse(pl.on_ground)
        self.assertGreater(pl.altitude, 0.2)

    def test_hover_thrust_balances_weight(self):
        pl = self.airborne_plant()
        for _ in range(1000):
            pl.step(np.full(4, hover_cmd(self.veh)), 0.002)
        self.assertAlmostEqual(pl.altitude, 50.0, delta=0.01)
        self.assertLess(float(np.linalg.norm(pl.w)), 1e-9)

    def test_free_fall_with_drag(self):
        pl = self.airborne_plant()
        pl.m[:] = 0.0
        for _ in range(500):
            pl.step(np.zeros(4), 0.002)
        k = self.veh.drag_z
        expected = G / k * (1 - math.exp(-k * 1.0))
        self.assertAlmostEqual(float(pl.v[2]), expected, delta=0.05)

    def test_touchdown_records_impact_and_clamps(self):
        pl = self.airborne_plant(alt=0.5)
        pl.m[:] = 0.0
        for _ in range(400):
            pl.step(np.zeros(4), 0.002)
        self.assertTrue(pl.on_ground)
        self.assertAlmostEqual(pl.max_impact, math.sqrt(2 * G * 0.5), delta=0.3)
        self.assertFalse(pl.crashed)
        self.assertEqual(float(pl.p[2]), 0.0)

    def test_hard_landing_is_flagged_as_crash(self):
        pl = self.airborne_plant(alt=3.0)
        pl.m[:] = 0.0
        for _ in range(600):
            pl.step(np.zeros(4), 0.002)
        self.assertTrue(pl.crashed)

    def test_wind_pushes_an_uncontrolled_hover_downwind(self):
        pl = self.airborne_plant(wind=WindModel(mean=np.array([4.0, 0.0, 0.0])))
        for _ in range(500):
            pl.step(np.full(4, hover_cmd(self.veh)), 0.002)
        self.assertGreater(float(pl.v[0]), 0.5)
        self.assertAlmostEqual(float(pl.v[1]), 0.0, places=6)

    def test_motor_lag_time_constant(self):
        pl = self.airborne_plant()
        pl.m[:] = 0.0
        for _ in range(int(self.veh.motor_tau / 0.002)):
            pl.step(np.ones(4), 0.002)
        self.assertAlmostEqual(float(pl.m[0]), 1 - math.exp(-1), delta=0.02)

    def test_tilted_thrust_accelerates_sideways(self):
        pl = self.airborne_plant()
        pl.q = quat_from_euler(0.0, -0.2, 0.0)               # nose down -> forward (north) acceleration
        pl.step(np.full(4, hover_cmd(self.veh)), 0.002)
        self.assertGreater(float(pl.v[0]), 0.0)
        self.assertGreater(float(pl.v[2]), 0.0)              # and thrust no longer balances weight


class SensorTests(unittest.TestCase):
    def test_imu_reads_minus_g_on_the_ground(self):
        w = SimWorld(spec=CLEAN, seed=1)
        f = None
        for _ in range(50):
            f = w.step(np.zeros(4))
        np.testing.assert_allclose(f.imu.accel, [0, 0, -G], atol=1e-9)
        np.testing.assert_allclose(f.imu.gyro, 0.0, atol=1e-12)

    def test_gps_latency_and_rate(self):
        spec = replace(CLEAN, gps_latency=0.12, gps_hz=10.0)
        w = SimWorld(spec=spec, seed=1)
        got = []
        for _ in range(1000):
            f = w.step(np.zeros(4))
            got += [(w.t, g.t) for g in f.gps]
        self.assertTrue(9 <= len(got) <= 19)
        for now, stamp in got:
            self.assertAlmostEqual(now - stamp, 0.12, delta=0.003)
        stamps = [g[1] for g in got]
        self.assertAlmostEqual(np.diff(stamps).mean(), 0.1, delta=0.005)

    def test_gps_outage_and_glitch_windows(self):
        spec = replace(CLEAN, gps_outages=[(0.5, 1.5)], gps_glitches=[(1.5, 2.0, [30.0, 0.0, 0.0])])
        w = SimWorld(spec=spec, seed=1)
        pos, stamps = [], []
        for _ in range(1500):
            f = w.step(np.zeros(4))
            for g in f.gps:
                pos.append(g.pos_ned.copy()); stamps.append(g.t)
        stamps = np.array(stamps); pos = np.array(pos)
        self.assertFalse(np.any((stamps >= 0.5) & (stamps < 1.5)))
        glitched = pos[(stamps >= 1.5) & (stamps < 2.0)]
        self.assertTrue(len(glitched) > 0 and np.all(glitched[:, 0] > 25.0))
        clean = pos[stamps < 0.5]
        self.assertTrue(np.all(np.abs(clean[:, 0]) < 1e-9))

    def test_range_is_height_over_cos_tilt_and_invalid_out_of_range(self):
        w = SimWorld(spec=CLEAN, seed=1)
        pl = w.plant
        pl.on_ground = False
        pl.p[:] = [0, 0, -3.0]
        pl.q = quat_from_euler(0.0, math.radians(30), 0.0)
        pl.m[:] = 0.0
        f = None
        for _ in range(60):
            f = w.step(np.zeros(4))
            if f.range:
                break
        # range sample is delayed by 10 ms; run a little more to collect one
        rs = []
        for _ in range(20):
            f = w.step(np.zeros(4))
            rs += f.range
        self.assertTrue(rs)
        r = rs[-1]
        self.assertTrue(r.valid)
        self.assertGreater(r.range_m, 2.9)                   # ~ alt / cos(30)
        pl.p[:] = [0, 0, -50.0]
        rs = []
        for _ in range(40):
            rs += w.step(np.zeros(4)).range
        self.assertTrue(rs and not rs[-1].valid)

    def test_mag_field_follows_yaw(self):
        w = SimWorld(spec=CLEAN, seed=1, yaw0=math.pi / 2)
        ms = []
        for _ in range(100):
            ms += w.step(np.zeros(4)).mag
        m = ms[-1].field
        # facing east: north field points to the LEFT of the body (-y)
        self.assertLess(m[1], -0.05)
        self.assertGreater(m[2], 0.3)                        # downward component

    def test_same_seed_is_reproducible(self):
        def run(seed):
            w = SimWorld(seed=seed)
            out = []
            for _ in range(200):
                out.append(w.step(np.full(4, 0.5)).imu.accel.copy())
            return np.array(out)
        np.testing.assert_array_equal(run(5), run(5))
        self.assertFalse(np.array_equal(run(5), run(6)))

    def test_battery_drains_only_while_spinning(self):
        spec = replace(CLEAN, battery_drain_per_s=0.1)
        w = SimWorld(spec=spec, seed=1)
        fr = None
        for _ in range(1000):
            f = w.step(np.zeros(4))
            fr = f.battery[-1].frac if f.battery else fr
        self.assertAlmostEqual(fr, 1.0, places=6)
        for _ in range(1000):
            f = w.step(np.full(4, 0.5))
            fr = f.battery[-1].frac if f.battery else fr
        self.assertLess(fr, 0.95)


if __name__ == "__main__":
    unittest.main()
