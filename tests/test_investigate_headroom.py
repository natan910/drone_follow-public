"""R1 investigate (autonomy/investigate.py + Autopilot.investigate) and the headroom map
(mapping/headroom.py + Autopilot._ceiling). Pure parts first, then the real autopilot
flying the toy world (sim/scenarios.py), like tests/test_autopilot_sim.py."""

import math
import unittest
from dataclasses import replace

from autonomy.autopilot import Autopilot
from autonomy.investigate import face_rate, standoff_point
from autonomy.runner import run
from config import AppConfig
from datatypes import Mode, Pose, Task
from mapping.headroom import HeadroomMap
from platforms.sim import SimPlatform
from sim.scenarios import demo_world


class TestGeometry(unittest.TestCase):
    def test_standoff_point_is_on_the_line_toward_the_drone(self):
        self.assertEqual(standoff_point((10.0, 0.0), (0.0, 0.0), 3.0), (3.0, 0.0))
        self.assertEqual(standoff_point((1.0, 0.0), (0.0, 0.0), 3.0), (3.0, 0.0))    # too close: back off
        self.assertEqual(standoff_point((0.0, 0.0), (0.0, 0.0), 3.0), (0.0, -3.0))   # overhead: step south

    def test_face_rate_turns_the_short_way(self):
        self.assertGreater(face_rate(0, 0, 0.0, (5, 5), 1.5, 40), 0)                # spot to the NE: turn right
        self.assertLess(face_rate(0, 0, 0.0, (-5, 5), 1.5, 40), 0)
        self.assertEqual(face_rate(0, 0, 0.0, (0, 5), 1.5, 40), 0.0)                # dead ahead
        self.assertEqual(abs(face_rate(0, 0, 0.0, (0, -5), 1.5, 40)), 40)            # behind: full rate


class TestHeadroomMap(unittest.TestCase):
    def test_lowest_reading_wins_and_a_moved_thing_is_forgotten(self):
        m = HeadroomMap(20, 0.5)
        p = Pose(1.0, 1.0, 0.0, 1.5)
        self.assertIsNone(m.ceiling_near(p, 1.0))
        m.observe(p, 1.2)                                   # ceiling 2.7
        m.observe(replace(p, x=1.6), 0.6)                   # a lamp 0.6 m further east: 2.1
        self.assertAlmostEqual(m.ceiling_near(p, 1.0), 2.1)
        self.assertAlmostEqual(m.ceiling_near(p, 0.0), 2.7)
        m.observe(p, None)
        m.observe(p, 20.0)                                  # out of range: ignored
        self.assertAlmostEqual(m.ceiling_near(p, 0.0), 2.7)
        for _ in range(3):
            m.observe(replace(p, x=1.6), 1.2)               # the lamp was taken down
        self.assertAlmostEqual(m.ceiling_near(p, 1.0), 2.7)


class TestAutopilotInterface(unittest.TestCase):
    def test_investigate_updates_extends_and_stops(self):
        ap = Autopilot(AppConfig())
        ap.investigate(1, 2, now=10, timeout_s=30, reason="T1")
        ap.investigate(3, 4, now=20, timeout_s=999)          # running: the spot moves, deadline kept
        inv = ap._investigation
        self.assertEqual((inv.x, inv.y, inv.until, inv.reason), (3, 4, 40, "T1"))
        ap.investigate(3, 4, now=35, extend_s=20)            # still in view: keep going
        self.assertEqual(inv.until, 55)
        ap.set_task(Task.PATROL)
        self.assertTrue(ap.investigating)                    # PATROL keeps it
        ap.set_task(Task.HOLD)
        self.assertFalse(ap.investigating)
        self.assertIn("HOLD", ap.last_investigation)
        self.assertIsNone(ap.status()["investigating"])

    def test_goals_stay_inside_the_geofence(self):
        cfg = AppConfig()
        cfg.safety.geofence_radius_m = 10.0
        ap = Autopilot(cfg)
        self.assertEqual(ap._inside_fence((3.0, 4.0)), (3.0, 4.0))
        x, y = ap._inside_fence((12.0, 0.0))
        self.assertAlmostEqual(x, 8.0)
        cfg.safety.geofence_radius_m = float("inf")                   # --no-geofence dry runs
        self.assertEqual(ap._inside_fence((120.0, 0.0)), (120.0, 0.0))

    def test_ceiling_limit(self):
        ap = Autopilot(AppConfig())
        p = Pose(0, 0, 0, 2.0)
        self.assertEqual(ap._ceiling(p), ap.cfg.safety.max_altitude_m)
        ap.headroom.observe(p, 0.3)                          # ceiling 2.3 m above launch
        self.assertAlmostEqual(ap._ceiling(p), 2.3 - ap.cfg.avoid.ceiling_clearance_m)


def fly(ap, plat, seconds, trace):
    run(plat, ap, seconds=seconds, on_step=lambda o, d: trace.append((o.now, d.mode, o.pose, d.note)))


class TestClosedLoop(unittest.TestCase):
    def test_goes_to_the_standoff_point_watches_then_resumes_patrol(self):
        world = demo_world(person=None)
        plat, ap = SimPlatform(world), Autopilot(AppConfig())
        ap.set_task(Task.PATROL)
        fly(ap, plat, 30, [])
        spot, standoff = (-6.0, 3.0), 2.5
        ap.investigate(*spot, now=plat.now(), standoff_m=standoff, timeout_s=60, reason="T1")
        trace = []
        fly(ap, plat, 70, trace)
        looking = [(t, p) for t, m, p, _ in trace if m == Mode.INVESTIGATE]
        self.assertGreater(len(looking), 100)
        best = min(abs(math.hypot(p.x - spot[0], p.y - spot[1]) - standoff) for _, p in looking)
        self.assertLess(best, 0.8)                                       # reached the standoff ring
        t, p = looking[-1]
        err = math.degrees((math.atan2(spot[0] - p.x, spot[1] - p.y) - p.yaw + math.pi) % (2 * math.pi) - math.pi)
        self.assertLess(abs(err), 25)                                    # facing the spot
        self.assertGreater(math.hypot(p.x - spot[0], p.y - spot[1]), 1.5)  # never overhead
        self.assertNotIn(Mode.INVESTIGATE, {m for tt, m, _, _ in trace if tt > trace[0][0] + 62})
        self.assertEqual(ap.last_investigation, "time up")
        self.assertEqual(world.collisions, 0)

    def test_stays_under_a_lower_ceiling_once_seen(self):
        class LowCeiling(SimPlatform):
            """An upward range sensor: 2.7 m room, 2.1 m over the south part (a beam, stairs)."""
            def observe(self):
                obs = super().observe()
                if obs.scan is None:
                    return obs
                ceil = 2.1 if obs.pose.y < -2.0 else 2.7
                return replace(obs, scan=replace(obs.scan, up=max(0.0, ceil - obs.pose.z)))

        world = demo_world(person=None)
        plat, ap = LowCeiling(world), Autopilot(AppConfig())
        ap.set_task(Task.PATROL)
        trace = []
        fly(ap, plat, 240, trace)
        under = [2.1 - p.z for t, m, p, _ in trace if p.y < -2.5 and t > 5]
        self.assertGreater(len(under), 50, "the patrol never went under the low part")
        settled = sorted(under[20:])                  # after the first 2 s under it
        self.assertGreaterEqual(settled[len(settled) // 10], 0.3)       # 90 % of the time >= 0.3 m clearance
        self.assertGreater(ap.headroom.known(), 20)
        self.assertEqual(world.collisions, 0)


if __name__ == "__main__":
    unittest.main()
