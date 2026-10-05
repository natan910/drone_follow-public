import math
import unittest

import numpy as np

from flightcore.config import ControlConfig, VehicleParams
from flightcore.control import AttitudeController, Mixer, PositionController, RateController, Setpoint
from flightcore.mathutil import G, quat_from_euler, quat_to_R
from flightcore.sim.dynamics import Plant


class MixerTests(unittest.TestCase):
    def setUp(self):
        self.veh = VehicleParams()
        self.mx = Mixer(self.veh, ControlConfig())
        self.rng = np.random.default_rng(1)

    def test_feasible_demand_is_realised_exactly(self):
        for _ in range(100):
            c = self.rng.uniform(0.3, 0.6)
            tau = self.rng.uniform(-0.15, 0.15, 3) * np.array([1, 1, 0.3])
            r = self.mx.mix(c, tau)
            self.assertFalse(r.sat_rp or r.sat_yaw or r.sat_thrust)
            T, tx, ty, tz = self.mx.realised(r.thrust)
            self.assertAlmostEqual(T, 4 * c * self.veh.max_thrust, places=9)
            np.testing.assert_allclose([tx, ty, tz], tau, atol=1e-9)

    def test_outputs_always_within_limits(self):
        for _ in range(300):
            c = self.rng.uniform(0.0, 1.0)
            tau = self.rng.uniform(-3, 3, 3)
            r = self.mx.mix(c, tau)
            self.assertTrue(np.all(r.cmd >= self.veh.idle - 1e-9) and np.all(r.cmd <= 1.0 + 1e-9))

    def test_yaw_is_sacrificed_before_roll_and_pitch(self):
        tau = np.array([0.35, -0.2, 0.0])
        base = self.mx.mix(0.5, tau)
        self.assertFalse(base.sat_rp)
        r = self.mx.mix(0.5, tau + np.array([0.0, 0.0, 5.0]))
        self.assertTrue(r.sat_yaw)
        self.assertFalse(r.sat_rp)
        _, tx, ty, tz = self.mx.realised(r.thrust)
        np.testing.assert_allclose([tx, ty], tau[:2], atol=1e-6)
        self.assertLess(abs(tz), 5.0)
        self.assertGreater(tz, 0.0)

    def test_collective_is_shifted_to_keep_attitude_authority(self):
        tau = np.array([0.3, 0.0, 0.0])
        r = self.mx.mix(0.98, tau)                     # collective far too high for that torque
        self.assertTrue(r.sat_thrust)
        _, tx, _, _ = self.mx.realised(r.thrust)
        self.assertAlmostEqual(tx, 0.3, places=6)
        T = self.mx.realised(r.thrust)[0]
        self.assertLess(T, 4 * 0.98 * self.veh.max_thrust)

    def test_impossible_roll_is_scaled_not_clipped_asymmetrically(self):
        r = self.mx.mix(0.5, np.array([50.0, 0.0, 0.0]))
        self.assertTrue(r.sat_rp)
        _, tx, ty, tz = self.mx.realised(r.thrust)
        self.assertGreater(tx, 0.0)
        self.assertAlmostEqual(ty, 0.0, places=6)     # direction preserved

    def test_thrust_curve_inverse(self):
        for u in np.linspace(0, 1, 11):
            x = (1 - self.veh.thrust_expo) * u + self.veh.thrust_expo * u * u
            self.assertAlmostEqual(float(self.mx.inv_curve(x)), u, places=9)

    def test_torque_signs_match_the_independent_plant(self):
        """Mixer conventions (motor order, spin directions) vs the physics model."""
        for axis in range(3):
            plant = Plant(self.veh)
            plant.on_ground = False
            plant.p[:] = [0, 0, -100.0]
            plant.m[:] = self.veh.hover_frac
            tau = np.zeros(3)
            tau[axis] = 0.05
            r = self.mx.mix(self.veh.hover_frac, tau)
            for _ in range(10):                        # 20 ms
                plant.step(r.cmd, 0.002)
            w = plant.w
            self.assertGreater(w[axis], 0.0, f"axis {axis}")
            others = [abs(w[i]) for i in range(3) if i != axis]
            self.assertLess(max(others), 0.05 * abs(w[axis]) + 1e-9, f"axis {axis} coupling")


class RateTests(unittest.TestCase):
    def test_step_response_on_a_rigid_body(self):
        veh, cfg = VehicleParams(), ControlConfig()
        rc = RateController(cfg, veh)
        I = np.array(veh.inertia)
        w = np.zeros(3)
        sp = np.array([2.0, -1.0, 0.5])
        dt = 0.002
        peak = np.zeros(3)
        for _ in range(900):                       # 1.8 s
            tau = rc.update(sp, w, dt)
            w = w + tau / I * dt
            peak = np.maximum(peak, np.abs(w))
        np.testing.assert_allclose(w, sp, atol=0.05)
        self.assertTrue(np.all(peak <= 1.35 * np.abs(sp)))          # overshoot < 35 %

    def test_integrator_freezes_and_is_clamped(self):
        veh, cfg = VehicleParams(), ControlConfig()
        rc = RateController(cfg, veh)
        for _ in range(2000):
            rc.update([5.0, 0, 0], [0, 0, 0], 0.002)
        self.assertLessEqual(rc.integ[0], cfg.rate_i_max[0] + 1e-9)
        rc.reset()
        for _ in range(500):
            rc.update([5.0, 0, 0], [0, 0, 0], 0.002, freeze_i=(True, True, True))
        np.testing.assert_allclose(rc.integ, 0.0)

    def test_gyro_noise_does_not_produce_huge_torque(self):
        veh, cfg = VehicleParams(), ControlConfig()
        rc = RateController(cfg, veh)
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(2000):
            tau = rc.update([0, 0, 0], 0.007 * rng.standard_normal(3), 0.002)
            worst = max(worst, float(np.max(np.abs(tau))))
        self.assertLess(worst, 0.05)


class AttitudeTests(unittest.TestCase):
    def setUp(self):
        self.att = AttitudeController(ControlConfig())
        self.level = np.array([0.0, 0.0, 1.0])

    def rate(self, r=0.0, p=0.0, y=0.0, yaw_sp=0.0, z_d=None, ff=0.0):
        R = quat_to_R(quat_from_euler(r, p, y))
        return self.att.update(R, self.level if z_d is None else z_d, yaw_sp, ff)[0]

    def test_zero_error_gives_zero_rate(self):
        np.testing.assert_allclose(self.rate(), 0.0, atol=1e-12)

    def test_rolled_right_commands_roll_left_and_so_on(self):
        self.assertLess(self.rate(r=0.2)[0], 0.0)
        self.assertLess(self.rate(p=0.2)[1], 0.0)          # nose up -> pitch down
        self.assertLess(self.rate(y=0.3)[2], 0.0)          # heading ahead of setpoint -> turn back
        self.assertGreater(self.rate(y=-0.3)[2], 0.0)

    def test_rate_is_proportional_for_small_errors(self):
        a, b = self.rate(r=0.02)[0], self.rate(r=0.04)[0]
        self.assertAlmostEqual(b / a, 2.0, places=2)

    def test_yaw_rate_feed_forward_is_added(self):
        self.assertAlmostEqual(self.rate(ff=0.7)[2], 0.7, places=9)

    def test_output_is_clamped(self):
        r = self.rate(r=1.3)
        self.assertLessEqual(abs(r[0]), ControlConfig().max_rate[0] + 1e-9)


class PositionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ControlConfig()
        self.pc = PositionController(self.cfg)
        self.pc.reset(np.zeros(3), 0.0)
        self.dt = 0.002

    def run_for(self, sp, secs, p=None, v=None):
        p = np.zeros(3) if p is None else p
        v = np.zeros(3) if v is None else v
        out = None
        for _ in range(int(secs / self.dt)):
            out = self.pc.update(p, v, 0.0, sp, self.dt)
        return out

    def test_velocity_setpoint_is_slew_limited(self):
        sp = Setpoint(kind="fly", vel=np.array([5.0, 0, 0]))
        out = self.run_for(sp, 0.5)
        self.assertLessEqual(out.v_sp[0], self.cfg.max_accel_xy * 0.5 + 1e-6)
        self.assertGreater(out.v_sp[0], 0.9 * self.cfg.max_accel_xy * 0.5)

    def test_speed_limit(self):
        sp = Setpoint(kind="fly", vel=np.array([50.0, 0, 0]))
        out = self.run_for(sp, 5.0, v=np.array([self.cfg.max_speed_xy, 0, 0]))
        self.assertLessEqual(out.v_sp[0], self.cfg.max_speed_xy + 1e-9)

    def test_tilt_limit_caps_horizontal_acceleration(self):
        sp = Setpoint(kind="fly", vel=np.array([6.0, 0, 0]))
        out = self.run_for(sp, 3.0, v=np.zeros(3))
        a_lim = (G - out.a_sp[2]) * math.tan(math.radians(self.cfg.max_tilt_deg))
        self.assertLessEqual(math.hypot(*out.a_sp[:2]), a_lim + 1e-9)
        self.assertTrue(out.sat_xy)

    def test_no_horizontal_control_when_degraded(self):
        sp = Setpoint(kind="fly", vel=np.array([3.0, 3.0, 0]), horizontal=False)
        out = self.run_for(sp, 1.0)
        np.testing.assert_allclose(out.a_sp[:2], 0.0)

    def test_nan_position_axes_fall_back_to_velocity(self):
        sp = Setpoint(kind="fly", pos=np.array([np.nan, np.nan, -3.0]), vel=np.array([1.0, 0, 0]))
        out = self.run_for(sp, 3.0, p=np.array([10.0, 10.0, 0.0]))
        self.assertAlmostEqual(out.v_sp[0], 1.0, places=6)      # x follows vel, not the x position
        self.assertAlmostEqual(out.v_sp[1], 0.0, places=6)
        self.assertLess(out.v_sp[2], -0.5)                      # z climbs toward -3 (up)

    def test_position_error_produces_velocity_toward_target(self):
        sp = Setpoint(kind="fly", pos=np.array([4.0, -4.0, 0.0]))
        out = self.run_for(sp, 3.0)
        self.assertGreater(out.v_sp[0], 0.5)
        self.assertLess(out.v_sp[1], -0.5)

    def test_heading_integrates_yaw_rate_and_lag_is_bounded(self):
        sp = Setpoint(kind="fly", yaw=None, yaw_rate=1.0)
        out = self.run_for(sp, 0.5)
        self.assertAlmostEqual(out.yaw_sp, 0.5, places=2)
        out = self.run_for(sp, 5.0)                             # vehicle never turns: setpoint must not run away
        self.assertLessEqual(abs(out.yaw_sp), math.radians(self.cfg.yaw_lag_max_deg) + 1e-6)

    def test_absolute_heading_is_rate_limited(self):
        sp = Setpoint(kind="fly", yaw=math.pi / 2)
        out = self.run_for(sp, 0.1)
        self.assertLess(out.yaw_sp, 0.8 * self.cfg.max_rate[2] * 0.1 + 1e-6)


if __name__ == "__main__":
    unittest.main()
