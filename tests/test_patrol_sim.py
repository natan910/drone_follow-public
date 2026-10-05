"""tools/patrol_sim.py runs (the toy world + the real autopilot + the real perimeter mission),
and missions/simsource.py sees what a camera would. The long demo is run by hand."""

import importlib.util
import io
import math
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np

from datatypes import Observation, Pose
from missions.simsource import SimObject, SimSceneSource, Walker, render_topdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tool():
    spec = importlib.util.spec_from_file_location("patrol_sim", os.path.join(ROOT, "tools", "patrol_sim.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Box = lambda x0, y0, x1, y1: SimpleNamespace(x0=x0, y0=y0, x1=x1, y1=y1)   # noqa: E731


class TestSimSource(unittest.TestCase):
    def test_sees_only_in_view_in_range_and_not_behind_walls(self):
        world = SimpleNamespace(boxes=[Box(-1, 4, 1, 4.2)])                    # a wall 4 m north
        src = SimSceneSource(world, [Walker([(0, 3)], wait_s=100), Walker([(0, 6)], wait_s=100),
                                     Walker([(5, 0)], wait_s=100)], [SimObject("laptop", -1.5, 3)],
                             hfov_deg=66, view_range_m=8, noise_m=0.0)
        s = src.step(None, Observation(now=0.0, pose=Pose(0, 0, 0.0, 2.0)))    # facing north
        self.assertEqual([(p.x, p.y) for p in s.people], [(0, 3)])            # (0,6) behind the wall, (5,0) aside
        self.assertEqual([o.label for o in s.objects], ["laptop"])
        self.assertIsNone(src.step(None, Observation(now=0.1, pose=Pose(0, 0, 0.0, 2.0))))   # rate limit
        s = src.step(None, Observation(now=1.0, pose=Pose(0, 0, math.pi / 2, 2.0)))           # facing east
        self.assertEqual([(p.x, p.y) for p in s.people], [(5, 0)])

    def test_walker_timeline(self):
        w = Walker([(0, 0), (10, 0)], start_t=5, speed_mps=2, wait_s=3)
        self.assertIsNone(w.pos(4))
        self.assertEqual(w.pos(7.5), (5.0, 0.0))
        self.assertEqual(w.pos(12), (10, 0))
        self.assertIsNone(w.pos(20))

    def test_picture(self):
        world = SimpleNamespace(boxes=[Box(-10, -7, 10, -6.8), Box(-10, 6.8, 10, 7)])
        img = render_topdown(world, Pose(0, 0, 0.3, 2.0), None, [Walker([(1, 1)])], [SimObject("tv", 2, 2)],
                             t=1.0, investigating={"x": 1, "y": 1}, lines=["hello"])
        self.assertEqual(img.dtype, np.uint8)
        self.assertGreater(img.shape[0], 100)


class TestPatrolSimTool(unittest.TestCase):
    def test_runs(self):
        tool = load_tool()
        out = io.StringIO()
        with redirect_stdout(out):
            code = tool.main(["--seconds", "60", "--intruder-at", "5", "--trigger-at", "2", "--quiet"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("Done at t=", text)
        self.assertNotIn("switched off after an error", text)
        self.assertIn("collisions 0", text)


if __name__ == "__main__":
    unittest.main()
