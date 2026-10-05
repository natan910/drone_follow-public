import math
import unittest

import numpy as np

from flightcore.config import SupervisorConfig
from flightcore.mathutil import quat_from_euler
from flightcore.state import NavEstimate
from flightcore.supervisor import Context, Decision, Mode, Supervisor

DT = 0.02


def nav(p=(0.0, 0.0, 0.0), v=(0.0, 0.0, 0.0), yaw=0.0, pos=True, vel=True, alt=True, range_height=None) -> NavEstimate:
    return NavEstimate(
        p=np.array(p, float), v=np.array(v, float), q=quat_from_euler(0, 0, yaw), att_valid=True, yaw_valid=True,
        vel_valid=vel, pos_valid=pos, alt_valid=alt, range_height=range_height,
    )


class Harness:
    """Drives a Supervisor with hand-made estimates (no simulation)."""

    def __init__(self, cfg: SupervisorConfig | None = None):
        self.cfg = cfg or SupervisorConfig()
        self.sup = Supervisor(self.cfg)
        self.t = 0.0
        self.est = nav()
        self.last: Decision | None = None

    def run(self, secs, est=None, **ctx) -> Decision:
        if est is not None:
            self.est = est
        for _ in range(max(1, int(round(secs / DT)))):
            self.t += DT
            tilt = ctx.pop("tilt", None) if False else ctx.get("tilt", self.est.tilt)
            kw = {k: v for k, v in ctx.items() if k != "tilt"}
            self.last = self.sup.update(Context(t=self.t, dt=DT, est=self.est, tilt=tilt, **kw))
        return self.last

    def armed_and_hovering(self, alt=2.0) -> "Harness":
        self.sup.arm(nav(), self.t)
        self.sup.takeoff(alt, self.est)
        self.run(0.1, est=nav(p=(0, 0, -alt)))
        self.assert_mode(Mode.HOLD)
        return self

    def assert_mode(self, mode):
        assert self.sup.mode == mode, f"{self.sup.mode} != {mode}: {self.sup.events}"


class ArmingAndTakeoffTests(unittest.TestCase):
    def test_disarmed_means_motors_off(self):
        h = Harness()
        self.assertEqual(h.run(0.1).power, "off")
        self.assertFalse(h.sup.armed)

    def test_armed_idles_then_times_out(self):
        h = Harness()
        h.sup.arm(h.est, h.t)
        self.assertEqual(h.run(1.0).power, "idle")
        d = h.run(h.cfg.arm_timeout_s + 1.0)
        self.assertEqual(d.power, "off")
        self.assertFalse(h.sup.armed)

    def test_takeoff_climbs_then_holds_at_altitude(self):
        h = Harness()
        h.sup.arm(h.est, h.t)
        self.assertTrue(h.sup.takeoff(3.0, h.est))
        d = h.run(0.1)
        self.assertEqual(d.power, "fly")
        self.assertLess(d.sp.vel[2], 0.0)                          # NED: negative = up
        self.assertTrue(math.isnan(d.sp.pos[2]))                   # vertical speed, not position
        h.run(0.1, est=nav(p=(0, 0, -2.9)))
        h.assert_mode(Mode.HOLD)
        d = h.run(0.1)
        self.assertAlmostEqual(d.sp.pos[2], -3.0)

    def test_takeoff_only_from_armed(self):
        h = Harness()
        self.assertFalse(h.sup.takeoff(2.0, h.est))
        h.armed_and_hovering()
        self.assertFalse(h.sup.takeoff(2.0, h.est))

    def test_takeoff_speed_tapers_near_the_target(self):
        h = Harness()
        h.sup.arm(h.est, h.t)
        h.sup.takeoff(3.0, h.est)
        far = h.run(0.1, est=nav(p=(0, 0, -0.5))).sp.vel[2]
        near = h.run(0.1, est=nav(p=(0, 0, -2.6))).sp.vel[2]
        self.assertLess(far, near)                                  # faster (more negative) when far
        self.assertLessEqual(abs(far), h.cfg.takeoff_speed + 1e-9)

    def test_takeoff_timeout_lands(self):
        h = Harness()
        h.sup.arm(h.est, h.t)
        h.sup.takeoff(3.0, h.est)
        h.run(h.cfg.takeoff_timeout_s + 1.0, est=nav(p=(0, 0, -0.5)))
        h.assert_mode(Mode.LAND)
        self.assertEqual(h.sup.reason, "takeoff_timeout")

    def test_disarm_is_refused_in_flight_unless_forced(self):
        h = Harness().armed_and_hovering()
        self.assertFalse(h.sup.disarm())
        self.assertTrue(h.sup.disarm(force=True))
        self.assertFalse(h.sup.armed)

    def test_arm_clears_a_previous_failsafe(self):
        h = Harness().armed_and_hovering()
        h.run(0.1, batt_frac=0.05)
        self.assertEqual(h.sup.latched, "land")
        h.sup.disarm(force=True)
        h.sup.arm(nav(), h.t)
        self.assertIsNone(h.sup.latched)


class GuidedCommandTests(unittest.TestCase):
    def test_commands_refused_unless_flying(self):
        h = Harness()
        self.assertFalse(h.sup.command_velocity_body(1, 0, 0, 0, h.t, h.est))
        h.sup.arm(h.est, h.t)
        self.assertFalse(h.sup.command_velocity_body(1, 0, 0, 0, h.t, h.est))
        self.assertFalse(h.sup.command_hold(h.est))

    def test_body_frame_velocity_is_rotated_by_heading(self):
        h = Harness().armed_and_hovering()
        h.est = nav(p=(0, 0, -2), yaw=math.pi / 2)                   # facing east
        h.sup.command_velocity_body(2.0, 0.0, 0.0, 0.0, h.t, h.est)
        d = h.run(DT)
        np.testing.assert_allclose(d.sp.vel[:2], [0.0, 2.0], atol=1e-9)   # forward = east
        h.sup.command_velocity_body(0.0, 1.0, 0.0, 0.0, h.t, h.est)
        d = h.run(DT)
        np.testing.assert_allclose(d.sp.vel[:2], [-1.0, 0.0], atol=1e-9)  # right of east = south
        self.assertEqual(h.sup.mode, Mode.GUIDED)

    def test_yaw_rate_is_passed_through_and_heading_not_held(self):
        h = Harness().armed_and_hovering()
        h.sup.command_velocity_body(0, 0, 0, 0.5, h.t, h.est)
        d = h.run(DT)
        self.assertIsNone(d.sp.yaw)
        self.assertAlmostEqual(d.sp.yaw_rate, 0.5)

    def test_zero_vertical_command_holds_altitude_nonzero_does_not(self):
        h = Harness().armed_and_hovering()
        h.sup.command_velocity_body(1, 0, 0, 0, h.t, h.est)
        d = h.run(DT)
        self.assertTrue(np.isnan(d.sp.pos[0]) and np.isnan(d.sp.pos[1]))
        self.assertAlmostEqual(d.sp.pos[2], -2.0)
        h.sup.command_velocity_body(1, 0, -0.5, 0, h.t, h.est)
        d = h.run(DT)
        self.assertIsNone(d.sp.pos)
        self.assertAlmostEqual(d.sp.vel[2], -0.5)

    def test_altitude_ceiling_blocks_further_climb(self):
        h = Harness().armed_and_hovering()
        est = nav(p=(0, 0, -(h.cfg.max_altitude + 0.1)))
        h.sup.command_velocity_body(0, 0, -1.0, 0, h.t, est)
        d = h.run(DT, est=est)
        self.assertGreaterEqual(d.sp.vel[2], 0.0)

    def test_command_silence_holds_then_lands_and_stays_landing(self):
        h = Harness().armed_and_hovering()
        h.sup.command_velocity_body(2, 0, 0, 0, h.t, h.est)
        h.run(0.3)
        h.assert_mode(Mode.GUIDED)
        d = h.run(h.cfg.setpoint_timeout_s)                          # stale now
        np.testing.assert_allclose(d.sp.vel, 0.0)
        self.assertFalse(np.isnan(d.sp.pos[0]))                      # position hold, not "keep flying"
        self.assertIsNone(h.sup.latched)
        self.assertTrue(h.sup.command_velocity_body(1, 0, 0, 0, h.t, h.est))   # link back in time: resumes
        h.run(0.1)
        h.run(h.cfg.setpoint_timeout_s + h.cfg.hold_before_land_s + 0.5)
        h.assert_mode(Mode.LAND)
        self.assertEqual(h.sup.latched, "land")
        self.assertEqual(h.sup.reason, "command_lost")
        self.assertFalse(h.sup.command_velocity_body(1, 0, 0, 0, h.t, h.est))  # too late: stays landing

    def test_goto_flies_to_position_and_hold_stops(self):
        h = Harness().armed_and_hovering()
        self.assertTrue(h.sup.command_position([5, 3, -4], math.pi / 4, h.est))
        d = h.run(DT)
        np.testing.assert_allclose(d.sp.pos, [5, 3, -4])
        self.assertAlmostEqual(d.sp.yaw, math.pi / 4)
        h.est = nav(p=(2, 1, -3), v=(1.0, 0.5, 0))
        self.assertTrue(h.sup.command_hold(h.est))
        d = h.run(DT)
        self.assertEqual(h.sup.mode, Mode.HOLD)
        self.assertGreater(d.sp.pos[0], 2.0)                         # stopping point is ahead of us


class FailsafeTests(unittest.TestCase):
    def test_low_battery_returns_home(self):
        h = Harness().armed_and_hovering()
        h.run(DT, est=nav(p=(10, 5, -2)), batt_frac=h.cfg.battery_warn - 0.01)
        h.assert_mode(Mode.RTL)
        self.assertEqual(h.sup.latched, "rtl")
        self.assertFalse(h.sup.command_velocity_body(1, 0, 0, 0, h.t, h.est))   # brain cannot override

    def test_critical_battery_lands_immediately_and_overrides_rtl(self):
        h = Harness().armed_and_hovering()
        h.run(DT, batt_frac=h.cfg.battery_warn - 0.01)
        h.assert_mode(Mode.RTL)
        h.run(DT, batt_frac=h.cfg.battery_crit - 0.01)
        h.assert_mode(Mode.LAND)
        self.assertEqual(h.sup.latched, "land")
        h.run(DT, batt_frac=h.cfg.battery_warn - 0.01)               # warn again: must not downgrade to RTL
        h.assert_mode(Mode.LAND)

    def test_unknown_battery_triggers_nothing(self):
        h = Harness().armed_and_hovering()
        h.run(1.0, batt_frac=None)
        h.assert_mode(Mode.HOLD)

    def test_rtl_climbs_then_cruises_home_then_lands(self):
        h = Harness().armed_and_hovering(alt=2.0)
        far = nav(p=(20, 10, -2))
        h.sup.command_rtl(far)
        d = h.run(DT, est=far)
        self.assertEqual(h.sup.mode, Mode.RTL)
        np.testing.assert_allclose(d.sp.pos, [20, 10, -h.cfg.rtl_altitude])   # climb in place first
        at_alt = nav(p=(20, 10, -(h.cfg.rtl_altitude - 0.2)))
        h.run(DT, est=at_alt)                                                  # phase switches on this tick
        d = h.run(DT, est=at_alt)
        np.testing.assert_allclose(d.sp.pos, [0, 0, -h.cfg.rtl_altitude])     # then head home
        self.assertEqual(d.sp.max_speed_xy, h.cfg.rtl_speed)
        h.run(DT, est=nav(p=(0.3, 0.2, -h.cfg.rtl_altitude)))
        h.assert_mode(Mode.LAND)

    def test_rtl_never_descends_below_current_altitude(self):
        h = Harness().armed_and_hovering()
        high = nav(p=(5, 0, -20))
        h.sup.command_rtl(high)
        d = h.run(DT, est=high)
        self.assertAlmostEqual(d.sp.pos[2], -20.0)

    def test_fence_radius_and_ceiling(self):
        for est in (nav(p=(SupervisorConfig().max_radius + 5, 0, -2)), nav(p=(0, 0, -(SupervisorConfig().max_altitude + 3)))):
            h = Harness().armed_and_hovering()
            h.run(DT, est=est)
            h.assert_mode(Mode.RTL)
            self.assertEqual(h.sup.reason, "fence")

    def test_fence_can_be_configured_to_land_or_be_off(self):
        far = nav(p=(500, 0, -2))
        h = Harness(SupervisorConfig(fence_action="land")).armed_and_hovering()
        h.run(DT, est=far)
        h.assert_mode(Mode.LAND)
        h = Harness(SupervisorConfig(fence_action="none")).armed_and_hovering()
        h.run(DT, est=far)
        h.assert_mode(Mode.HOLD)

    def test_fence_needs_a_valid_position(self):
        h = Harness().armed_and_hovering()
        h.run(DT, est=nav(p=(500, 0, -2), pos=False))
        self.assertNotEqual(h.sup.reason, "fence")

    def test_flip_kills_but_a_brief_tilt_does_not(self):
        h = Harness().armed_and_hovering()
        h.run(0.1, tilt=math.radians(85))
        h.assert_mode(Mode.HOLD)
        h.run(0.5, tilt=math.radians(10))
        h.run(h.cfg.kill_tilt_time_s + 0.2, tilt=math.radians(85))
        h.assert_mode(Mode.KILLED)
        self.assertEqual(h.last.power, "off")
        self.assertFalse(h.sup.armed)
        self.assertTrue(h.sup.disarm())

    def test_operator_kill(self):
        h = Harness().armed_and_hovering()
        h.sup.kill("operator_kill")
        self.assertEqual(h.run(DT).power, "off")
        self.assertEqual(h.sup.reason, "operator_kill")

    def test_hold_degrades_with_the_estimator(self):
        h = Harness().armed_and_hovering()
        d = h.run(DT, est=nav(p=(0, 0, -2)))
        self.assertFalse(np.isnan(d.sp.pos[0]))                      # full position hold
        d = h.run(DT, est=nav(p=(0, 0, -2), pos=False))
        self.assertTrue(np.isnan(d.sp.pos[0]))                       # velocity hold
        self.assertTrue(d.sp.horizontal)
        d = h.run(DT, est=nav(p=(0, 0, -2), pos=False, vel=False))
        self.assertFalse(d.sp.horizontal)                            # level attitude only

    def test_goto_without_position_falls_back_then_lands(self):
        h = Harness().armed_and_hovering()
        h.sup.command_position([5, 0, -2], None, h.est)
        h.run(DT, est=nav(p=(0, 0, -2), pos=False))
        h.assert_mode(Mode.GOTO)
        h.run(h.cfg.est_pos_lost_hold_s + 0.2)
        h.assert_mode(Mode.LAND)
        self.assertEqual(h.sup.reason, "position_lost")

    def test_lost_attitude_or_altitude_means_emergency_descent(self):
        for kw in ({"att_ok": False}, {}):
            h = Harness().armed_and_hovering()
            est = nav(p=(0, 0, -2), alt=("att_ok" in kw))
            d = h.run(DT, est=est, **kw)
            self.assertEqual(d.power, "emergency")
            self.assertEqual(h.sup.latched, "land")
            self.assertEqual(h.sup.reason, "estimator_failure")

    def test_emergency_ends_on_impact_or_ground_hint_not_on_a_short_timer(self):
        h = Harness().armed_and_hovering()
        h.run(DT, att_ok=False)
        self.assertEqual(h.run(20.0, att_ok=False).power, "emergency")   # no ground evidence: keep descending
        h.run(0.2, att_ok=False, ground_hint=True)
        self.assertEqual(h.last.power, "emergency")                       # hint must persist
        h.run(h.cfg.ground_hint_time_s + 0.1, att_ok=False, ground_hint=True)
        self.assertEqual(h.last.power, "off")
        self.assertFalse(h.sup.armed)
        h = Harness().armed_and_hovering()
        h.run(DT, att_ok=False)
        h.run(DT, att_ok=False, impact=True)
        self.assertEqual(h.last.power, "off")

    def test_position_and_yaw_resets_shift_stored_targets(self):
        h = Harness().armed_and_hovering()
        h.sup.command_position([5, 3, -2], 0.5, h.est)
        h.sup.hold_pos = np.array([1.0, 2.0, -2.0])
        home = h.sup.home.copy()
        h.sup.shift_position([10.0, -4.0])
        np.testing.assert_allclose(h.sup.hold_pos, [11.0, -2.0, -2.0])
        np.testing.assert_allclose(h.sup.goto_pos, [15.0, -1.0, -2.0])
        np.testing.assert_allclose(h.sup.home, home)                  # home is absolute: untouched
        h.sup.hold_yaw = 3.0
        h.sup.shift_yaw(0.5)
        self.assertAlmostEqual(h.sup.hold_yaw, 3.5 - 2 * math.pi)     # wrapped
        self.assertAlmostEqual(h.sup.goto_yaw, 1.0)


class LandingTests(unittest.TestCase):
    def test_descent_rate_schedule(self):
        h = Harness().armed_and_hovering()
        h.sup.command_land(h.est)
        c = h.cfg
        fast = h.run(DT, est=nav(p=(0, 0, -(c.land_fast_above + 1)))).sp.vel[2]
        slow = h.run(DT, est=nav(p=(0, 0, -(c.land_slow_below - 0.5)))).sp.vel[2]
        mid = h.run(DT, est=nav(p=(0, 0, -(0.5 * (c.land_fast_above + c.land_slow_below))))).sp.vel[2]
        self.assertAlmostEqual(fast, c.land_speed_fast)
        self.assertAlmostEqual(slow, c.land_speed)
        self.assertTrue(slow < mid < fast)

    def test_range_height_is_preferred_over_the_estimated_altitude(self):
        h = Harness().armed_and_hovering()
        h.sup.command_land(h.est)
        d = h.run(DT, est=nav(p=(0, 0, -10), range_height=0.8))
        self.assertAlmostEqual(d.sp.vel[2], h.cfg.land_speed)         # 0.8 m above ground, not 10

    def test_touchdown_disarms_after_the_dwell_time(self):
        h = Harness().armed_and_hovering()
        h.sup.command_land(h.est)
        ground = nav(p=(0, 0, -0.1), v=(0, 0, 0.0))
        h.run(h.cfg.touchdown_time_s * 0.6, est=ground)
        self.assertTrue(h.sup.armed)
        h.run(h.cfg.touchdown_time_s * 0.6, est=ground)
        self.assertFalse(h.sup.armed)
        self.assertEqual(h.last.power, "off")
        self.assertIn("landed", [e[1] for e in h.sup.events])

    def test_still_descending_is_not_touchdown(self):
        h = Harness().armed_and_hovering()
        h.sup.command_land(h.est)
        h.run(3.0, est=nav(p=(0, 0, -0.1), v=(0, 0, 0.6)))
        self.assertTrue(h.sup.armed)

    def test_pushing_against_the_ground_counts_as_touchdown(self):
        h = Harness().armed_and_hovering()
        h.sup.command_land(h.est)
        h.run(1.5 * h.cfg.touchdown_time_s + 0.2, est=nav(p=(0, 0, -0.6), v=(0, 0, 0.0)))   # commanded descent, not moving
        self.assertFalse(h.sup.armed)


if __name__ == "__main__":
    unittest.main()
