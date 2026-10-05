"""Closed-loop flights: FlightCore (estimator + supervisor + controller) against the simulated plant.

Default run: the essentials (~1 min).  FLIGHTCORE_SLOW=1 adds every scenario (~4 min).
"""
import math
import os
import unittest
from dataclasses import replace

import numpy as np

from flightcore.config import ControlConfig, FlightConfig, SupervisorConfig
from flightcore.mathutil import wrap_pi
from flightcore.sim import SensorSpec
from flightcore.sim.harness import ClosedLoop

SLOW = unittest.skipUnless(os.environ.get("FLIGHTCORE_SLOW"), "set FLIGHTCORE_SLOW=1 for the full scenario set")


def est_errors(s: ClosedLoop):
    n, pl = s.core.nav, s.truth
    return (float(np.linalg.norm(n.p - pl.p)), float(np.linalg.norm(n.v - pl.v)),
            math.degrees(abs(wrap_pi(n.yaw - pl.euler()[2]))))


class Mission(unittest.TestCase):
    """One shared flight: align, arm, takeoff, hover, fly forward, stop, turn, land."""

    @classmethod
    def setUpClass(cls):
        s = cls.s = ClosedLoop(seed=2)
        r = cls.r = {}
        r["ready"] = s.wait_ready()
        r["t_ready"] = s.t
        r["arm"] = s.core.arm()
        r["takeoff"] = s.core.takeoff(2.0)
        r["takeoff_done"] = s.run_until(lambda c: c.core.mode.value != "takeoff", 10.0)
        r["t_takeoff"] = s.t
        r["max_alt"] = s.altitude
        s.run(1.0)
        pos, alts, errs = [], [], []
        for _ in range(30):
            s.run(0.1)
            pos.append(s.truth.p[:2].copy()); alts.append(s.altitude); errs.append(est_errors(s))
        r["hover_drift"] = float(np.ptp(np.array(pos), axis=0).max())
        r["hover_alt"] = (min(alts), max(alts))
        yaw0 = s.truth.euler()[2]
        p0 = s.truth.p.copy()
        speeds = []
        for _ in range(40):
            s.core.set_velocity_body(2.0, 0.0, 0.0, 0.0)
            s.run(0.1)
            speeds.append(float(s.truth.v[0])); errs.append(est_errors(s))
        r["cruise_speed"] = float(np.mean(speeds[20:]))
        r["distance"] = float(s.truth.p[0] - p0[0])
        r["heading_change"] = math.degrees(abs(wrap_pi(s.truth.euler()[2] - yaw0)))
        r["alt_in_cruise"] = s.altitude
        s.fly_velocity(0, 0, 0, 0, 3.0)
        r["stopped_speed"] = float(np.linalg.norm(s.truth.v))
        s.fly_velocity(0, 0, 0, math.radians(45), 2.0)
        s.fly_velocity(0, 0, 0, 0, 1.5)
        r["yaw_after_turn"] = math.degrees(s.truth.euler()[2])
        errs.append(est_errors(s))
        r["est_err"] = np.array(errs)
        s.core.land()
        r["landed"] = s.run_until(lambda c: not c.core.armed, 30.0)
        r["t_landed"] = s.t

    def test_estimator_aligns_and_arms_quickly(self):
        self.assertTrue(self.r["ready"])
        self.assertLess(self.r["t_ready"], 2.0)
        self.assertEqual(self.r["arm"], (True, []))

    def test_takeoff_reaches_altitude_without_big_overshoot(self):
        self.assertTrue(self.r["takeoff"] and self.r["takeoff_done"])
        self.assertLess(self.r["t_takeoff"] - self.r["t_ready"], 5.0)
        self.assertLess(self.r["max_alt"], 2.4)

    def test_hover_holds_position_and_altitude(self):
        self.assertLess(self.r["hover_drift"], 0.4)
        lo, hi = self.r["hover_alt"]
        self.assertGreater(lo, 1.85)
        self.assertLess(hi, 2.15)

    def test_velocity_command_is_followed(self):
        self.assertAlmostEqual(self.r["cruise_speed"], 2.0, delta=0.25)
        self.assertGreater(self.r["distance"], 5.0)
        self.assertLess(self.r["heading_change"], 5.0)
        self.assertAlmostEqual(self.r["alt_in_cruise"], 2.0, delta=0.15)     # altitude held while moving

    def test_zero_command_stops_the_vehicle(self):
        self.assertLess(self.r["stopped_speed"], 0.25)

    def test_yaw_rate_command_turns_by_the_right_amount(self):
        self.assertAlmostEqual(self.r["yaw_after_turn"], 90.0, delta=8.0)

    def test_vehicle_stays_within_the_tilt_limit(self):
        self.assertLess(math.degrees(self.s.max_tilt), self.s.cfg.control.max_tilt_deg + 6.0)

    def test_estimate_tracks_truth_throughout(self):
        e = self.r["est_err"]
        self.assertLess(e[:, 0].max(), 0.8)             # position, m
        self.assertLess(e[:, 1].max(), 0.4)             # velocity, m/s
        self.assertLess(e[:, 2].max(), 3.0)             # heading, deg

    def test_lands_gently_and_disarms_itself(self):
        self.assertTrue(self.r["landed"])
        self.assertFalse(self.s.truth.crashed)
        self.assertLess(self.s.truth.max_impact, 1.0)
        self.assertLess(self.r["t_landed"], 45.0)
        seq = [e[1] for e in self.s.core.sup.events]
        for a, b in zip(["disarmed->armed", "armed->takeoff", "takeoff->hold", "hold->guided", "guided->land", "landed"],
                        seq):
            self.assertEqual(a, b)


class PreArm(unittest.TestCase):
    def test_refuses_before_the_estimator_is_aligned(self):
        s = ClosedLoop(seed=1)
        s.run(0.2)
        ok, why = s.core.arm()
        self.assertFalse(ok)
        self.assertEqual(why, ["estimator_not_initialised"])
        self.assertEqual(float(np.max(s.cmd)), 0.0)

    def test_refuses_and_explains_each_problem(self):
        s = ClosedLoop(seed=1, spec=SensorSpec(battery_start=0.2, gps_outages=[(0.0, 99.0)]))
        s.run(2.5)
        ok, why = s.core.arm()
        self.assertFalse(ok)
        for reason in ("battery_low", "position_invalid", "gps_stale"):
            self.assertIn(reason, why)
        self.assertFalse(s.core.armed)

    def test_refuses_when_not_still_and_when_already_armed(self):
        s = ClosedLoop(seed=1)
        self.assertTrue(s.wait_ready())
        s.core.nav.v[:] = [1.0, 0, 0]
        self.assertIn("not_still", s.core.prearm_check())
        s.core.nav.v[:] = 0.0
        self.assertEqual(s.core.prearm_check(), [])
        self.assertTrue(s.core.arm()[0])
        self.assertIn("already_armed", s.core.prearm_check())

    def test_armed_vehicle_idles_on_the_ground(self):
        s = ClosedLoop(seed=1)
        self.assertTrue(s.wait_ready())
        s.core.arm()
        s.run(1.0)
        self.assertTrue(s.truth.on_ground)
        self.assertAlmostEqual(float(np.mean(s.cmd)), s.cfg.vehicle.idle, places=6)

    def test_no_commands_before_takeoff(self):
        s = ClosedLoop(seed=1)
        s.wait_ready()
        self.assertFalse(s.core.takeoff(2.0))                   # not armed
        self.assertFalse(s.core.set_velocity_body(1, 0, 0, 0))


class Robustness(unittest.TestCase):
    def test_wind_and_gusts(self):
        s = ClosedLoop(seed=3, wind=(3.0, -2.0, 0.0), gust_std=1.0)
        self.assertTrue(s.arm_and_takeoff(2.0))
        s.run(3.0)
        ps = []
        for _ in range(50):
            s.run(0.1)
            ps.append(s.truth.p.copy())
        ps = np.array(ps)
        self.assertLess(float(np.abs(ps - ps.mean(0)).max()), 0.5)
        self.assertLess(est_errors(s)[0], 0.8)

    def test_unknown_mass_is_learned_as_hover_thrust(self):
        cfg = FlightConfig(control=ControlConfig(hover_lpf_tau=2.0))
        heavy = replace(cfg.vehicle, mass=cfg.vehicle.mass * 1.25)
        s = ClosedLoop(cfg, true_vehicle=heavy, seed=4)
        self.assertTrue(s.arm_and_takeoff(2.0))
        s.run(10.0)
        self.assertAlmostEqual(s.altitude, 2.0, delta=0.1)
        self.assertAlmostEqual(s.core.ctl.hover_frac, heavy.hover_frac, delta=0.03 * heavy.hover_frac)
        self.assertLess(abs(s.core.ctl.pos.i_z), 0.5)            # integrator handed its load to hover_frac

    def test_a_weak_motor_is_compensated(self):
        s = ClosedLoop(seed=5, motor_scale=(1.0, 1.0, 0.88, 1.0))
        self.assertTrue(s.arm_and_takeoff(2.0))
        s.run(5.0)
        self.assertAlmostEqual(s.altitude, 2.0, delta=0.15)
        self.assertLess(math.degrees(s.max_tilt), 12.0)
        self.assertGreater(s.cmd[2], s.cmd[[0, 1, 3]].max())     # the weak motor is driven harder

    def test_command_silence_ends_in_an_automatic_landing(self):
        s = ClosedLoop(seed=8)
        self.assertTrue(s.arm_and_takeoff(2.0))
        s.fly_velocity(1.0, 0, 0, 0, 2.0)
        self.assertTrue(s.run_until(lambda c: not c.core.armed, 40.0))          # brain goes silent
        names = [e[1] for e in s.core.sup.events]
        self.assertIn("command_timeout", names)
        self.assertIn("failsafe:command_lost->land", names)
        self.assertFalse(s.truth.crashed)
        self.assertLess(s.truth.max_impact, 1.0)

    def test_operator_kill_cuts_the_motors(self):
        s = ClosedLoop(seed=14)
        self.assertTrue(s.arm_and_takeoff(2.0))
        s.core.kill()
        s.run(0.05)
        self.assertEqual(float(np.max(s.cmd)), 0.0)
        self.assertFalse(s.core.armed)

    def test_gps_outage_is_bridged_by_the_drag_model(self):
        spec = SensorSpec(gps_outages=[(6.0, 12.0)])
        s = ClosedLoop(seed=6, spec=spec)
        self.assertTrue(s.arm_and_takeoff(2.0))
        worst_v = worst_p = 0.0
        while s.t < 11.5:
            s.core.set_velocity_body(1.5, 0, 0, 0)
            s.run(0.1)
            if s.t > 6.5:
                ep, ev, _ = est_errors(s)
                worst_p, worst_v = max(worst_p, ep), max(worst_v, ev)
        self.assertLess(worst_v, 0.5)
        self.assertLess(worst_p, 1.5)
        s.fly_velocity(0, 0, 0, 0, 5.0)                                          # GPS back at t=12
        self.assertLess(est_errors(s)[0], 0.6)
        self.assertEqual(s.core.mode.value, "guided")
        self.assertGreater(s.core.est.stats["gps_pos_h"][0], 20)


@SLOW
class SlowScenarios(unittest.TestCase):
    def test_low_battery_returns_home_and_lands(self):
        cfg = FlightConfig(supervisor=SupervisorConfig(arm_min_battery=0.1))
        s = ClosedLoop(cfg, seed=9, spec=SensorSpec(battery_start=0.31, battery_drain_per_s=0.01))
        self.assertTrue(s.arm_and_takeoff(2.0))
        s.fly_velocity(2.0, 0, 0, 0, 3.0)
        s.run_until(lambda c: c.core.sup.latched is not None, 10.0)
        self.assertEqual(s.core.sup.reason, "battery_low")
        s.core.set_velocity_body(0, 0, 0, 0)
        self.assertTrue(s.run_until(lambda c: not c.core.armed, 40.0))
        self.assertLess(float(np.linalg.norm(s.truth.p[:2])), 2.5)               # landed near home
        self.assertFalse(s.truth.crashed)

    def test_fence_forces_a_return(self):
        s = ClosedLoop(seed=15)
        self.assertTrue(s.arm_and_takeoff(2.0))
        while s.core.sup.latched is None and s.t < 40:
            s.core.set_velocity_body(5.0, 0, 0, 0)
            s.run(0.1)
        self.assertEqual(s.core.sup.reason, "fence")
        self.assertTrue(s.run_until(lambda c: not c.core.armed, 60.0))
        self.assertLess(float(np.linalg.norm(s.truth.p[:2])), 3.0)

    def test_short_gps_glitch_is_rejected(self):
        spec = SensorSpec(gps_glitches=[(8.0, 10.0, [30.0, -20.0, 0.0])])
        s = ClosedLoop(seed=7, spec=spec)
        self.assertTrue(s.arm_and_takeoff(2.0))
        worst = 0.0
        while s.t < 14.0:
            s.run(0.1)
            worst = max(worst, est_errors(s)[0])
        self.assertLess(worst, 1.0)
        self.assertGreater(s.core.est.stats["gps_pos_h"][1], 10)

    def test_persistent_gps_offset_does_not_fling_the_vehicle(self):
        spec = SensorSpec(gps_glitches=[(8.0, 60.0, [30.0, -20.0, 0.0])])
        s = ClosedLoop(seed=7, spec=spec)
        self.assertTrue(s.arm_and_takeoff(2.0))
        p0 = s.truth.p.copy()
        s.run(14.0)
        self.assertEqual(s.core.est.reset_seq, 1)
        self.assertIn("position_reset", [e[1] for e in s.core.sup.events])
        self.assertLess(float(np.linalg.norm(s.truth.p[:2] - p0[:2])), 3.0)     # stayed put physically

    def test_compass_disturbance_is_ignored(self):
        spec = SensorSpec(mag_disturbances=[(6.0, 11.0, [0.30, -0.2, 0.1])])
        s = ClosedLoop(seed=10, spec=spec)
        self.assertTrue(s.arm_and_takeoff(2.0))
        worst = 0.0
        while s.t < 12.0:
            s.run(0.1)
            worst = max(worst, est_errors(s)[2])
        self.assertLess(worst, 3.0)

    def test_person_walking_under_the_rangefinder_does_not_move_the_drone(self):
        holder = {}
        spec = SensorSpec(terrain=lambda n, e: 1.7 if 8.0 <= holder["s"].t < 12.0 else 0.0)
        s = holder["s"] = ClosedLoop(seed=11, spec=spec)
        self.assertTrue(s.arm_and_takeoff(2.0))
        s.run(3.0)                                             # settle at the takeoff height first
        lo = hi = s.altitude
        while s.t < 14.0:
            s.run(0.1)
            lo, hi = min(lo, s.altitude), max(hi, s.altitude)
        self.assertGreater(lo, 1.8)
        self.assertLess(hi, 2.2)

    def test_same_seed_same_flight(self):
        def go():
            s = ClosedLoop(seed=12)
            s.arm_and_takeoff(2.0)
            s.fly_velocity(1.0, 0, 0, 0, 1.0)
            return s.truth.p.copy(), s.cmd.copy()
        a, b = go(), go()
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


if __name__ == "__main__":
    unittest.main()
