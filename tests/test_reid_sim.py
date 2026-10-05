"""Closed loop, toy world, realistic faces: the face is only recognisable from in
front and not from steeply above. Body re-ID is what keeps the lock while
hovering over someone and while following them from behind."""

import math
import unittest

from autonomy.autopilot import Autopilot
from autonomy.runner import run
from config import AppConfig
from datatypes import Mode
from platforms.sim import SimPlatform
from sim.scenarios import demo_world


def setup(body_reid, cfg=None):
    # Standing at (-3, -4), facing the drone's start point: face visible on approach.
    world = demo_world(person=(-3.0, -4.0), realistic_face=True, body_reid=body_reid,
                       person_heading_deg=math.degrees(math.atan2(3, 4)))
    return world, Autopilot(cfg or AppConfig()), SimPlatform(world)


def fly(plat, ap, seconds):
    modes, sources = [], []
    run(plat, ap, seconds=seconds,
        on_step=lambda o, d: (modes.append(d.mode),
                              sources.append(None if o.detection is None else o.detection.source)))
    return modes, sources


def share(modes, mode):
    return modes.count(mode) / len(modes)


class OverheadTests(unittest.TestCase):
    def test_body_reid_keeps_the_lock_while_hovering_over_them(self):
        world, ap, plat = setup(body_reid=True)
        fly(plat, ap, 30)                                  # find, approach, settle overhead
        modes, sources = fly(plat, ap, 10)
        self.assertGreater(share(modes, Mode.TRACK), 0.95)
        self.assertNotIn("face", sources)                  # from up here it is the body doing it
        self.assertLess(world.person_distance(), 0.4)
        self.assertAlmostEqual(world.height_above_person(), 0.30, delta=0.15)

    def test_face_alone_loses_them_once_overhead(self):
        world, ap, plat = setup(body_reid=False)
        fly(plat, ap, 30)
        modes, _ = fly(plat, ap, 10)
        self.assertLess(share(modes, Mode.TRACK), 0.2)


class FromBehindTests(unittest.TestCase):
    def walk_away(self, body_reid):
        world, ap, plat = setup(body_reid)
        fly(plat, ap, 30)
        world.pvx = 0.3                                    # turns east and walks: back to the drone
        modes, _ = fly(plat, ap, 15)
        return world, modes

    def test_body_reid_follows_them_walking_away(self):
        world, modes = self.walk_away(body_reid=True)
        self.assertGreater(share(modes, Mode.TRACK), 0.9)
        self.assertLess(world.person_distance(), 1.0)
        self.assertEqual(world.collisions, 0)

    def test_keeps_up_with_a_normal_walking_pace(self):
        world, ap, plat = setup(body_reid=True)
        fly(plat, ap, 30)
        world.pvx = 0.8
        modes, _ = fly(plat, ap, 10)
        self.assertGreater(share(modes, Mode.TRACK), 0.75)  # a blink as they step off from under us
        self.assertLess(world.person_distance(), 1.0)       # ...but never falls behind

    def test_face_alone_cannot(self):
        world, modes = self.walk_away(body_reid=False)
        self.assertLess(share(modes, Mode.TRACK), 0.2)


class SafetyTests(unittest.TestCase):
    def test_body_only_tracking_expires_without_a_fresh_look_at_the_face(self):
        cfg = AppConfig()
        cfg.tracker.track_only_timeout_s = 10.0
        world, ap, plat = setup(body_reid=True, cfg=cfg)
        modes, sources = fly(plat, ap, 15)                 # overhead, on the body alone, by the end
        self.assertEqual(modes[-1], Mode.TRACK)
        self.assertEqual(sources[-30:].count("face"), 0)
        modes, _ = fly(plat, ap, 10)                       # last face now well over 10 s ago
        self.assertNotIn(Mode.TRACK, modes[-60:])          # the body alone is no longer trusted


if __name__ == "__main__":
    unittest.main()
