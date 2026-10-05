"""R6 tracks (missions/tracks.py), R7 baseline (missions/baseline.py), the event log. Pure logic, no I/O."""

import json
import math
import os
import shutil
import tempfile
import unittest

from datatypes import Pose
from missions.baseline import SceneBaseline
from missions.entities import Alert, EventLog, ObjectSighting, Scene
from missions.mission_config import BaselineConfig, TrackConfig
from missions.tracks import GroundTracker

GATE = [{"name": "gate", "polygon": [(-5, 5), (5, 5), (5, 15), (-5, 15)]}]


class TestTracker(unittest.TestCase):
    def tracker(self, **kw):
        return GroundTracker(TrackConfig(**kw), GATE, confirm_s=1.0, zone_cooldown_s=10.0)

    def test_a_walker_keeps_one_id(self):
        t = self.tracker()
        for k in range(20):                               # 1 m/s east, a sighting every 0.5 s
            t.update(k * 0.5, [(k * 0.5, 0.0)])
        (tr,) = t.confirmed()
        self.assertEqual(tr.id, "T1")
        self.assertAlmostEqual(tr.vx, 1.0, delta=0.1)
        self.assertAlmostEqual(tr.walked_m(), 9.5, delta=0.1)

    def test_two_people_crossing_paths_stay_apart(self):
        t = self.tracker()
        for k in range(21):
            s = k * 0.5
            t.update(s, [(-5 + s, 0.0), (5 - s, 0.6)])     # pass each other 0.6 m apart at t=5
        a, b = sorted(t.confirmed(), key=lambda tr: tr.id)
        self.assertGreater(a.x, 4.0)                       # T1 walked east all the way
        self.assertLess(b.x, -4.0)                         # T2 walked west

    def test_one_false_detection_is_not_a_person(self):
        t = self.tracker()
        ev, _ = t.update(0.0, [(3.0, 3.0)])
        self.assertEqual(ev[0]["type"], "track.new")
        t.update(10.0, [])
        self.assertEqual(t.tracks, {})                    # never confirmed: dropped silently

    def test_zone_alert_once_per_track_then_a_second_person_alerts_too(self):
        t = self.tracker()
        hits = []
        for k in range(8):
            _, h = t.update(k * 0.5, [(0.0, 10.0)])
            hits += h
        self.assertEqual([(tr.id, z) for tr, z in hits], [("T1", "gate")])   # dwell 1 s, once
        self.assertEqual(t.people_in("gate", 3.5), 1)
        for k in range(8, 40):                                               # someone else walks in
            _, h = t.update(k * 0.5, [(0.0, 10.0), (4.0, 12.0)])
            hits += h
        self.assertEqual([tr.id for tr, _ in hits], ["T1", "T2"])
        self.assertEqual(t.occupied(19.5), ["gate"])

    def test_leaving_and_forgetting(self):
        t = self.tracker(forget_s=5.0)
        for k in range(4):
            t.update(k * 0.5, [(0.0, 10.0)])
        ev, _ = t.update(2.0, [(0.0, 20.0)])              # 10 m jump: someone new, not T1 teleporting
        self.assertIn("track.new", [e["type"] for e in ev])
        ev, _ = t.update(9.0, [])
        ends = [e for e in ev if e["type"] == "track.end"]
        self.assertEqual(ends[0]["track"], "T1")
        self.assertEqual(ends[0]["zones"], ["gate"])
        json.dumps(ends)


class TestEventLog(unittest.TestCase):
    def test_file_and_memory(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "sub", "events.jsonl")
            log = EventLog(p, keep=2, clock=lambda: 5.0)
            log.write("track.new", 1.0, track="T1")
            a = Alert("gate", 2.0, 1.0, 2.0, 1, kind="intrusion", actions=[{"url": "secret-token"}])
            log.write("alert", 2.0, **a.to_dict())         # the alert's own "t" must not collide
            log.write("trigger", None, sensor="pir")
            log.close()
            with open(p) as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual([r["type"] for r in rows], ["track.new", "alert", "trigger"])
            self.assertIsNone(rows[2]["t"])
            self.assertNotIn("secret-token", json.dumps(rows))    # push buttons never reach disk
            self.assertEqual(len(log.recent), 2)
        finally:
            shutil.rmtree(d)


def scene(t, x, y, yaw_deg, labels):
    return Scene(t, Pose(x, y, math.radians(yaw_deg), 2.0), 30.0,
                 objects=[ObjectSighting(l, 0.9, (1, 1, 5, 5)) for l in labels])


class TestBaseline(unittest.TestCase):
    CFG = dict(min_visits=3, learn_visits=3, confirm_visits=2, revisit_gap_s=5.0, cooldown_s=0.0,
               label_cooldown_s=0.0)

    def visit(self, b, t, labels, frames=3, x=0.5, y=0.5, yaw=0.0):
        out = []
        for k in range(frames):
            out += b.observe(scene(t + k * 0.5, x, y, yaw, labels))
        return out + b.flush(t + 100)                        # away long enough: the visit closes

    def test_learns_then_reports_missing_after_two_visits(self):
        b = SceneBaseline(BaselineConfig(**self.CFG))
        t = 0.0
        for _ in range(4):
            self.assertEqual(self.visit(b, t, ["laptop", "chair"]), [])
            t += 200
        self.assertEqual(self.visit(b, t, ["chair"]), [])      # first visit without it: not yet
        t += 200
        (ch,) = self.visit(b, t, ["chair"])
        self.assertEqual((ch.label, ch.kind), ("laptop", "missing"))
        self.assertEqual((ch.x, ch.y), (1.0, 1.0))               # the viewpoint's cell centre (2 m cells)

    def test_new_object_appears(self):
        b = SceneBaseline(BaselineConfig(**self.CFG))
        t = 0.0
        for _ in range(4):
            self.visit(b, t, ["chair"])
            t += 200
        got = self.visit(b, t, ["chair", "backpack"]) + self.visit(b, t + 200, ["chair", "backpack"])
        self.assertEqual([(c.label, c.kind) for c in got], [("backpack", "appeared")])
        self.assertEqual(got[0].box, (1, 1, 5, 5))

    def test_one_missed_frame_is_not_a_change_and_others_are_ignored(self):
        b = SceneBaseline(BaselineConfig(**self.CFG))
        t = 0.0
        for _ in range(4):
            self.visit(b, t, ["laptop"])
            t += 200
        out = []
        for _ in range(3):   # laptop in 2 of 3 frames each visit (a person walked in front once)
            out += b.observe(scene(t, 0.5, 0.5, 0, ["laptop", "cup"]))
            out += b.observe(scene(t + 0.5, 0.5, 0.5, 0, ["person"]))
            out += b.observe(scene(t + 1.0, 0.5, 0.5, 0, ["laptop"]))
            out += b.flush(t + 100)
            t += 200
        self.assertEqual(out, [])
        self.assertNotIn("cup", b.places[b.key(Pose(0.5, 0.5, 0, 2), 30)].p)   # not a watched label

    def test_viewpoints_are_separate_and_a_new_normal_is_learned(self):
        b = SceneBaseline(BaselineConfig(**self.CFG))
        t = 0.0
        for _ in range(4):
            self.visit(b, t, ["tv"], yaw=0)
            self.visit(b, t + 50, [], yaw=180)                  # looking the other way: no tv, fine
            t += 200
        self.assertEqual(b.status()["places"], 2)
        reported = []
        for _ in range(12):                                     # the tv is gone for good
            reported += self.visit(b, t, [], yaw=0)
            t += 200
        self.assertEqual(len(reported), 1)                      # reported once, then it is the new normal

    def test_same_label_from_two_viewpoints_reported_once(self):
        b = SceneBaseline(BaselineConfig(**{**self.CFG, "label_cooldown_s": 300.0}))
        t = 0.0
        for _ in range(4):
            self.visit(b, t, ["laptop"], x=0.5)
            self.visit(b, t + 20, ["laptop"], x=2.5)
            t += 200
        got = []
        for _ in range(2):
            got += self.visit(b, t, [], x=0.5) + self.visit(b, t + 20, [], x=2.5)
            t += 100
        self.assertEqual(len(got), 1)

    def test_save_load_and_mismatched_settings(self):
        d = tempfile.mkdtemp()
        try:
            b = SceneBaseline(BaselineConfig(**self.CFG))
            for k in range(4):
                self.visit(b, k * 200.0, ["laptop"])
            p = os.path.join(d, "baseline.json")
            b.save(p)
            b2 = SceneBaseline(BaselineConfig(**self.CFG))
            self.assertEqual(b2.load(p), 1)
            (place,) = b2.places.values()
            self.assertEqual(place.n, 4)
            self.assertAlmostEqual(place.p["laptop"], 1.0)
            with self.assertRaises(ValueError):
                SceneBaseline(BaselineConfig(cell_m=1.0)).load(p)
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
