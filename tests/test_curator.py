"""dataset/curator.py: which frames are worth keeping, tagging, face blurring. Temp dirs, fake clocks."""

import json
import os
import shutil
import tempfile
import unittest
from argparse import Namespace

import numpy as np

from dataset.curator import Curator, CuratedRecorder, FaceBlur, build_recorder, pixelate
from dataset.layout import SessionInfo
from dataset.recorder import Recorder
from datatypes import Decision, Detection, DriveCommand, Mode, Observation, Pose, RangeBeam, RangeScan
from missions.mission_config import CuratorConfig


def obs(t=0.0, x=0.0, y=0.0, det=None, scan=None):
    return Observation(now=t, pose=Pose(x, y, 0.0, 2.0), detection=det, scan=scan)


def dec(mode):
    return Decision(DriveCommand(), mode)


FACE = Detection((10, 60, 60, 10), 0.0, 0.0, 0.2, "face")
BODY = Detection((10, 60, 200, 10), 0.0, 0.0, 0.2, "track")
TINY = Detection((10, 12, 12, 10), 0.0, 0.0, 0.01, "face")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class TestCuratorTriggers(unittest.TestCase):
    def setUp(self):
        self.c = Curator(CuratorConfig(cooldown_s=5.0, burst_s=2.0))

    def test_ground_frames_are_never_events(self):
        self.assertEqual(self.c.check(obs(), None, 0.0), [])

    def test_lost_and_reacquired(self):
        self.c.check(obs(0), dec(Mode.PATROL), 0)
        self.assertIn("reacquired", self.c.check(obs(1, det=FACE), dec(Mode.TRACK), 1))
        self.assertIn("lost", self.c.check(obs(2), dec(Mode.LOST), 2))

    def test_face_to_body_and_the_free_label_back(self):
        self.c.check(obs(0, det=FACE), dec(Mode.TRACK), 0)
        self.assertIn("face_to_body", self.c.check(obs(1, det=BODY), dec(Mode.TRACK), 1))
        self.assertIn("body_confirmed_by_face", self.c.check(obs(2, det=FACE), dec(Mode.TRACK), 2))

    def test_far_target_obstacle_and_borderline(self):
        self.assertIn("far_target", self.c.check(obs(0, det=TINY), dec(Mode.TRACK), 0))
        scan = RangeScan((RangeBeam(0.0, 1.2),), 8.0)
        self.assertIn("close_obstacle", self.c.check(obs(10, scan=scan), dec(Mode.PATROL), 10))
        c = Curator(stats_fn=lambda: {"best_sim": 0.61, "threshold": 0.6})
        self.assertIn("reid_borderline", c.check(obs(0, det=BODY), dec(Mode.TRACK), 0))
        c = Curator(stats_fn=lambda: {"best_sim": 0.9, "threshold": 0.6})
        self.assertNotIn("reid_borderline", c.check(obs(0, det=BODY), dec(Mode.TRACK), 0))

    def test_new_place_but_not_the_launch_spot(self):
        self.assertNotIn("new_place", self.c.check(obs(0), dec(Mode.PATROL), 0))
        self.assertIn("new_place", self.c.check(obs(1, x=12.0), dec(Mode.PATROL), 1))
        self.assertEqual(self.c.check(obs(20, x=12.5), dec(Mode.PATROL), 20), [])   # same cell, burst over

    def test_burst_then_quiet_and_cooldown(self):
        self.c.check(obs(0, det=FACE), dec(Mode.TRACK), 0)
        self.assertIn("face_to_body", self.c.check(obs(1, det=BODY), dec(Mode.TRACK), 1))
        self.assertEqual(self.c.check(obs(2, det=BODY), dec(Mode.TRACK), 2), ["burst:face_to_body"])
        self.assertEqual(self.c.check(obs(3.5, det=BODY), dec(Mode.TRACK), 3.5), [])
        self.c.check(obs(3.6, det=FACE), dec(Mode.TRACK), 3.6)
        self.assertNotIn("face_to_body", self.c.check(obs(3.7, det=BODY), dec(Mode.TRACK), 3.7))  # cooldown
        self.c.check(obs(7, det=FACE), dec(Mode.TRACK), 7)
        self.assertIn("face_to_body", self.c.check(obs(7.1, det=BODY), dec(Mode.TRACK), 7.1))

    def test_budget_caps_events_per_minute(self):
        c = Curator(CuratorConfig(max_events_per_min=3, cooldown_s=0.0, burst_s=0.0, new_place_cell_m=1.0))
        hits = sum(bool(c.check(obs(k, x=float(k)), dec(Mode.PATROL), k)) for k in range(20))
        self.assertEqual(hits, 3)


class TestCuratedRecorder(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="df_cur_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def rows(self):
        with open(os.path.join(self.root, "s1", "frames.jsonl")) as f:
            return [json.loads(line) for line in f]

    def test_events_are_saved_and_tagged_on_top_of_the_background(self):
        clock = Clock()
        rec = CuratedRecorder(self.root, SessionInfo("s1"), curator=Curator(), threaded=False,
                              clock=clock, free_bytes=lambda p: 10 ** 12)
        frame = np.full((48, 64, 3), 90, np.uint8)
        saved = 0
        for k in range(100):                          # 10 s at 10 Hz, face lost at 5 s
            clock.t = k * 0.1
            det = FACE if k < 50 else None
            mode = Mode.TRACK if k < 50 else Mode.LOST
            saved += rec.maybe_record(frame, obs(clock.t, det=det), dec(mode))
        rec.close()
        rows = self.rows()
        self.assertEqual(len(rows), saved)
        whys = [r["why"] for r in rows]
        self.assertIn(["lost"], whys)
        self.assertTrue(any(w[0].startswith("burst:") for w in whys))
        background = [w for w in whys if w == ["background"]]
        self.assertTrue(1 <= len(background) <= 3, whys)   # 0.2 fps over 10 s
        self.assertLess(saved, 12)                          # a plain 4 fps recorder: 40
        self.assertEqual(rec.status()["events"]["lost"], 1)

    def test_redaction_runs_before_saving(self):
        clock = Clock()
        seen = []

        def redact(frame, meta):
            seen.append(meta)
            frame[:] = 0
            return frame

        rec = CuratedRecorder(self.root, SessionInfo("s1"), redact=redact, fps=1.0, threaded=False,
                              clock=clock, free_bytes=lambda p: 10 ** 12)
        rec.maybe_record(np.full((48, 64, 3), 200, np.uint8), obs(det=FACE), dec(Mode.TRACK))
        rec.close()
        import cv2
        img = cv2.imread(os.path.join(self.root, "s1", "frames", "000000.jpg"))
        self.assertLess(float(img.mean()), 5)
        self.assertEqual(seen[0]["live_source"], "face")
        self.assertNotIn("why", self.rows()[0])          # no curator: rows look like a plain recording

    def test_failing_redaction_saves_nothing(self):
        def broken(frame, meta):
            raise RuntimeError("boom")

        rec = CuratedRecorder(self.root, SessionInfo("s1"), redact=broken, fps=1.0, threaded=False,
                              clock=Clock(), free_bytes=lambda p: 10 ** 12)
        rec.maybe_record(np.zeros((48, 64, 3), np.uint8), obs(), None)
        self.assertIn("redaction", rec.stopped)
        self.assertEqual(os.listdir(os.path.join(self.root, "s1", "frames")), [])

    def test_build_recorder_is_plain_without_the_new_flags(self):
        rec = build_recorder(Namespace(record=self.root, record_fps=2.0, record_max_gb=1.0), SessionInfo("s1"))
        try:
            self.assertIs(type(rec), Recorder)
            self.assertAlmostEqual(rec.interval, 0.5)
        finally:
            rec.close()
        rec = build_recorder(Namespace(record=self.root, curate=True), SessionInfo("s2"))
        try:
            self.assertIsInstance(rec, CuratedRecorder)
            self.assertAlmostEqual(rec.interval, 1 / CuratorConfig().background_fps)
        finally:
            rec.close()


class TestFaceBlur(unittest.TestCase):
    def test_blurs_bystanders_keeps_the_target(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)
        target, other = (10, 10, 50, 50), (120, 10, 160, 50)
        blur = FaceBlur(detect=lambda f: [target, other], pad=0.0)
        before_t = frame[10:50, 10:50].std()
        out = blur(frame, {"live_box": [5, 5, 60, 90], "live_source": "track"})
        self.assertAlmostEqual(float(out[10:50, 10:50].std()), float(before_t), places=3)
        self.assertLess(float(out[10:50, 120:160].std()), 0.5 * float(before_t))

    def test_no_target_blurs_everyone(self):
        frame = np.random.default_rng(1).integers(0, 255, (60, 60, 3), dtype=np.uint8)
        s = frame[5:45, 5:45].std()
        FaceBlur(detect=lambda f: [(5, 5, 45, 45)], pad=0.0)(frame, {})
        self.assertLess(float(frame[5:45, 5:45].std()), 0.5 * float(s))

    def test_pixelate_clips_to_the_frame(self):
        frame = np.zeros((20, 20, 3), np.uint8)
        pixelate(frame, (-10, -10, 5, 5))
        pixelate(frame, (18, 18, 40, 40))


if __name__ == "__main__":
    unittest.main()
