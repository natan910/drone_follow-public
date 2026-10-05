"""End-to-end: the real autopilot code flying the toy world."""

import math
import unittest

import numpy as np

from autonomy.autopilot import Autopilot
from autonomy.runner import run
from config import AppConfig
from datatypes import Mode, Observation
from platforms.sim import SimPlatform
from sim.scenarios import demo_world


def fly(world, cfg=None, seconds=120, until=None, platform=None):
    """Run the autopilot; returns (autopilot, platform, list of (t, mode))."""
    cfg = cfg or AppConfig()
    ap, plat = Autopilot(cfg), platform or SimPlatform(world)
    modes = []

    def hook(obs, decision):
        modes.append((obs.now, decision.mode))
        if until is not None and until(world, ap):
            return False

    run(plat, ap, seconds=seconds, on_step=hook)
    return ap, plat, modes


def seen(modes):
    return {m for _, m in modes}


class PatrolTests(unittest.TestCase):
    def test_explores_and_maps_the_room_without_hitting_anything(self):
        world = demo_world(person=None)
        ap, _, modes = fly(world, seconds=300)
        g = ap.grid
        interior = (np.abs(g.cx) < 9) & (np.abs(g.cy) < 6)
        self.assertEqual(world.collisions, 0)
        self.assertGreater(float(np.isfinite(g.viewed_t)[interior].mean()), 0.9)
        self.assertIn(Mode.EXPLORE, seen(modes))
        self.assertNotIn(Mode.TRACK, seen(modes))   # nobody to follow

    def test_learned_map_matches_the_real_walls_and_obstacles(self):
        world = demo_world(person=None)
        ap, _, _ = fly(world, seconds=300)
        g = ap.grid
        truth = np.zeros((g.h, g.w), bool)
        for b in world.boxes:
            dx = np.maximum.reduce([b.x0 - g.cx, g.cx - b.x1, np.zeros_like(g.cx)])
            dy = np.maximum.reduce([b.y0 - g.cy, g.cy - b.y1, np.zeros_like(g.cy)])
            truth |= (dx < g.res / 2) & (dy < g.res / 2)

        def grown(m):
            out = m.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    out |= np.roll(np.roll(m, dy, 0), dx, 1)
            return out

        self.assertGreater(float(grown(g.occupied)[truth].mean()), 0.9)   # found what is there
        self.assertGreater(float(grown(truth)[g.occupied].mean()), 0.95)  # invented almost nothing

    def test_switches_from_exploring_to_patrolling_once_everything_is_seen(self):
        world = demo_world(person=None)
        _, _, modes = fly(world, seconds=450)
        self.assertIn(Mode.PATROL, seen(modes))
        first_patrol = next(t for t, m in modes if m == Mode.PATROL)
        self.assertTrue(all(m != Mode.EXPLORE for t, m in modes if t > first_patrol + 30))


class TargetTests(unittest.TestCase):
    def test_finds_the_person_on_patrol_and_hovers_above_them(self):
        world = demo_world(person=(-7.0, 4.0))
        cfg = AppConfig()
        _, _, modes = fly(world, cfg, seconds=150)
        self.assertIn(Mode.TRACK, seen(modes))
        self.assertEqual(world.collisions, 0)
        # Task.FOLLOW never "parks" (that's Task.HOVER), but once settled it should
        # sit almost directly above them at the configured hover height.
        self.assertLess(world.person_distance(), cfg.control.arrive_radius_m + 0.3)
        self.assertAlmostEqual(world.height_above_person(), cfg.control.hover_height_above_target_m,
                               delta=0.3)

    def test_follows_a_walking_person_and_recovers_from_brief_visual_loss(self):
        # Hovering this close (hover_height_above_target_m) and almost directly
        # overhead, a single downward-ish camera has very little margin: a person
        # walking away from directly underneath can briefly leave the frame before
        # the drone catches up (see FollowController's docstring on yaw release).
        # The bar here is not "never blinks" but "reliably reacquires, stays close,
        # and never runs into anything" -- exactly the LOST -> SEARCH -> TRACK
        # recovery path exercised in test_when_the_target_disappears... below.
        world = demo_world(person=(-3.0, -4.0))
        ap, plat, _ = fly(world, seconds=40)         # settle above them first
        world.pvx = 0.3                              # they stroll east along the open south side
        modes = []
        run(plat, ap, seconds=25, on_step=lambda o, d: modes.append(d.mode))
        self.assertGreater(modes.count(Mode.TRACK) / len(modes), 0.4)
        self.assertEqual(modes[-1], Mode.TRACK)       # back on them by the end
        self.assertEqual(world.collisions, 0)
        self.assertLess(world.person_distance(), 4.5)  # never wandered far away while searching

    def test_when_the_target_disappears_it_searches_where_it_was_then_resumes_patrol(self):
        world = demo_world(person=(-7.0, 4.0))
        ap, plat, modes = fly(world, seconds=60)
        self.assertIn(Mode.TRACK, seen(modes))
        world.person = (8.0, -5.0)                   # target relocates out of sight
        modes = []
        run(plat, ap, seconds=200, on_step=lambda o, d: modes.append(d.mode))
        self.assertIn(Mode.LOST, modes)
        self.assertIn(Mode.SEARCH, modes)
        self.assertEqual(modes[-1], Mode.TRACK)      # patrol eventually found them again
        self.assertEqual(world.collisions, 0)

    def test_does_not_follow_anyone_if_no_target_is_enrolled(self):
        # nobody enrolled = the matcher never reports a detection
        world = demo_world(person=(-7.0, 4.0))
        world.miss_prob = 1.0
        _, _, modes = fly(world, seconds=100)
        self.assertNotIn(Mode.TRACK, seen(modes))


class SafetyTests(unittest.TestCase):
    def test_pilot_takeover_silences_the_autopilot_and_it_resumes_cleanly(self):
        world = demo_world(person=None)
        ap, plat, _ = fly(world, seconds=20)
        world.pilot_override = True
        sent = []
        before = (world.x, world.y)
        original = plat.driver.send
        plat.driver.send = lambda cmd: (sent.append(cmd), original(cmd))
        modes = []
        run(plat, ap, seconds=5, on_step=lambda o, d: modes.append(d.mode))
        self.assertEqual(set(modes), {Mode.IDLE})
        self.assertEqual(sent, [])                                   # not a single command sent
        self.assertAlmostEqual(math.dist(before, (world.x, world.y)), 0.0, places=2)
        world.pilot_override = False
        modes = []
        run(plat, ap, seconds=5, on_step=lambda o, d: modes.append(d.mode))
        self.assertNotIn(Mode.IDLE, modes)

    def test_flight_time_limit_brings_it_home_and_lands(self):
        world = demo_world(person=None)
        cfg = AppConfig()
        cfg.safety.max_flight_s = 60
        _, _, modes = fly(world, cfg, seconds=300)
        self.assertIn(Mode.RETURN, seen(modes))
        self.assertEqual(modes[-1][1], Mode.LAND)
        self.assertLess(math.hypot(world.x, world.y), 1.2)
        self.assertTrue(world.landed)
        self.assertEqual(world.collisions, 0)

    def test_leaving_the_geofence_sends_it_home(self):
        world = demo_world(person=None)
        cfg = AppConfig()
        cfg.safety.geofence_radius_m = 4.0
        _, _, modes = fly(world, cfg, seconds=300)
        self.assertIn(Mode.RETURN, seen(modes))
        self.assertLess(math.hypot(world.x, world.y), 1.2)
        self.assertTrue(world.landed)

    def test_low_battery_returns_then_lands(self):
        world = demo_world(person=None, battery_drain_pct_per_min=30.0)
        _, _, modes = fly(world, seconds=300)
        self.assertIn(Mode.RETURN, seen(modes))
        self.assertTrue(world.landed)

    def test_losing_the_range_sensors_holds_still_then_lands(self):
        world = demo_world(person=None)

        class BlindAfter20s(SimPlatform):
            def observe(self):
                obs = super().observe()
                if obs.now < 20:
                    return obs
                return Observation(now=obs.now, pose=obs.pose, detection=obs.detection, scan=None,
                                   battery_pct=obs.battery_pct, scan_age=obs.now - 20.0)

        plat = BlindAfter20s(world)
        ap, _, modes = fly(world, seconds=60, platform=plat)
        after = [m for t, m in modes if t > 20]
        self.assertEqual(after[10], Mode.HOLD)
        self.assertEqual(after[-1], Mode.LAND)
        self.assertNotIn(Mode.EXPLORE, [m for t, m in modes if t > 21])
        self.assertEqual(world.collisions, 0)


class CeilingTests(unittest.TestCase):
    """HANDOFF must-fix 1: the altitude ceiling holds in closed loop. Patrol wants
    PatrolConfig.altitude_m (2 m); a lower ceiling must win."""

    def heights(self, cfg, seconds=45):
        world = demo_world(person=None)
        trace = []
        run(SimPlatform(world), Autopilot(cfg), seconds=seconds,
            on_step=lambda o, d: trace.append((o.now, o.pose.z, d.mode)))
        return world, trace

    def test_without_a_low_ceiling_it_patrols_at_cruise_height(self):
        _, trace = self.heights(AppConfig())
        self.assertGreater(max(z for t, z, m in trace if t > 15), 1.6)

    def test_a_low_ceiling_keeps_it_down_without_sending_it_home(self):
        cfg = AppConfig()
        cfg.safety.max_altitude_m = 1.2
        world, trace = self.heights(cfg)
        self.assertLessEqual(max(z for t, z, m in trace if t > 15), 1.2 + 0.15)
        self.assertNotIn(Mode.RETURN, {m for t, z, m in trace})
        self.assertEqual(world.collisions, 0)


if __name__ == "__main__":
    unittest.main()
