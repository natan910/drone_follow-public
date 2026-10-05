"""The live-enrolment path end to end, minus real hardware: a photo arrives while
'flying', the main loop enrols it between frames, and the autopilot starts following."""

import unittest

from autonomy.autopilot import Autopilot
from comms.phone_server import CommandBox, EnrollmentBox
from config import AppConfig
from control.print_driver import PrintDriver
from datatypes import Mode, Task
from perception.face_matcher import FaceMatcher
from perception.stream_input import FrameSource
from platforms.real import RealPlatform
from tests.test_perception import ALICE, BOB, FakeEmbedder, face, img


class FakeCamera(FrameSource):
    def __init__(self, frame):
        self.frame, self.released = frame, False

    def read(self):
        return self.frame

    def release(self):
        self.released = True


def make(frame_tag=2):
    matcher = FaceMatcher(FakeEmbedder({
        1: [face((100, 100, 300, 300), ALICE)],                            # the photo from the phone
        2: [face((280, 150, 360, 250), ALICE), face((400, 100, 480, 200), BOB)],  # what the camera sees
        4: [],
    }))
    box = EnrollmentBox()
    return RealPlatform(FakeCamera(img(frame_tag)), matcher, PrintDriver(), enrollment=box), matcher, box


class RealPlatformTests(unittest.TestCase):
    def test_nobody_is_followed_until_a_photo_arrives(self):
        platform, matcher, box = make()
        self.assertIsNone(platform.observe().detection)

    def test_photo_sent_mid_flight_is_enrolled_between_frames(self):
        platform, matcher, box = make()
        reply = box.submit(img(1))
        obs = platform.observe()
        self.assertEqual(reply.result(timeout=0), {"ok": True, "target": True, "faces_in_photo": 1})
        self.assertTrue(matcher.has_target)
        self.assertIsNotNone(obs.detection)          # already spotted in that very frame

    def test_a_bad_photo_fails_the_request_but_not_the_flight(self):
        platform, matcher, box = make()
        reply = box.submit(img(4))                   # no face in it
        platform.observe()
        with self.assertRaises(ValueError):
            reply.result(timeout=0)
        self.assertFalse(matcher.has_target)

    def test_clearing_the_target_stops_the_following(self):
        platform, matcher, box = make()
        box.submit(img(1))
        platform.observe()
        cleared = box.submit(None)
        obs = platform.observe()
        self.assertEqual(cleared.result(timeout=0), {"ok": True, "target": False})
        self.assertIsNone(obs.detection)

    def test_autopilot_starts_following_once_the_photo_arrives(self):
        cfg = AppConfig()
        cfg.safety.require_scan = False              # webcam dry run: no range sensors
        platform, matcher, box = make()
        ap = Autopilot(cfg)
        before = [ap.step(platform.observe()).mode for _ in range(10)]
        box.submit(img(1))
        after = [ap.step(platform.observe()).mode for _ in range(10)]
        self.assertNotIn(Mode.TRACK, before)
        self.assertEqual(after[-1], Mode.TRACK)

    def test_stalled_camera_makes_the_frame_age_grow(self):
        platform, matcher, box = make()
        platform.camera.frame = None
        first = platform.observe().frame_age
        second = platform.observe().frame_age
        self.assertGreater(second, first)


class RealPlatformCommandTests(unittest.TestCase):
    def make(self):
        matcher = FaceMatcher(FakeEmbedder({2: []}))
        commands = CommandBox()
        platform = RealPlatform(FakeCamera(img(2)), matcher, PrintDriver(), commands=commands)
        return platform, commands

    def test_no_pending_command_leaves_the_observation_untouched(self):
        platform, commands = self.make()
        obs = platform.observe()
        self.assertIsNone(obs.task)
        self.assertIsNone(obs.hover_height_m)

    def test_a_task_switch_reaches_the_next_observation_and_is_acknowledged(self):
        platform, commands = self.make()
        reply = commands.submit({"task": "HOVER"})
        obs = platform.observe()
        self.assertEqual(obs.task, Task.HOVER)
        self.assertIsNone(obs.hover_height_m)
        self.assertEqual(reply.result(timeout=0), {"ok": True, "task": "HOVER"})

    def test_a_hover_height_change_reaches_the_next_observation(self):
        platform, commands = self.make()
        reply = commands.submit({"hover_height_m": 0.5})
        obs = platform.observe()
        self.assertIsNone(obs.task)
        self.assertEqual(obs.hover_height_m, 0.5)
        self.assertEqual(reply.result(timeout=0), {"ok": True, "hover_height_m": 0.5})

    def test_a_command_can_carry_both_at_once(self):
        platform, commands = self.make()
        commands.submit({"task": "FOLLOW", "hover_height_m": 0.2})
        obs = platform.observe()
        self.assertEqual((obs.task, obs.hover_height_m), (Task.FOLLOW, 0.2))

    def test_the_autopilot_actually_changes_task_from_a_phone_command(self):
        platform, commands = self.make()
        ap = Autopilot(AppConfig())
        self.assertEqual(ap.task, Task.FOLLOW)
        commands.submit({"task": "PATROL"})
        ap.step(platform.observe())
        self.assertEqual(ap.task, Task.PATROL)

    def test_no_command_box_means_no_command_support_but_still_works(self):
        matcher = FaceMatcher(FakeEmbedder({2: []}))
        platform = RealPlatform(FakeCamera(img(2)), matcher, PrintDriver())
        obs = platform.observe()
        self.assertIsNone(obs.task)
        self.assertIsNone(obs.hover_height_m)


class RealPlatformBodyReIDTests(unittest.TestCase):
    """The real perception chain end to end (minus the neural networks): the
    target is locked by face, then turns around; body re-ID keeps the lock."""

    def test_the_lock_survives_the_face_disappearing(self):
        import numpy as np
        from config import BodyReIDConfig
        from perception.body_reid import ColorHistogramEmbedder
        from perception.face_matcher import FaceObservation, unit
        from perception.target_finder import TargetFinder
        from tests.test_body_reid import RED, JEANS, FaceTable, RoiListDetector, face_of, scene

        faces, detector = FaceTable(), RoiListDetector()
        finder = TargetFinder(FaceMatcher(faces), detector, ColorHistogramEmbedder(), BodyReIDConfig())
        frame, boxes = scene((280, 60, 360, 420, RED, JEANS))
        detector.boxes, detector.frame = boxes, frame
        alice = [FaceObservation(face_of(boxes[0]), unit(np.array(ALICE, np.float32)))]

        faces.current = alice
        box = EnrollmentBox()
        platform = RealPlatform(FakeCamera(frame), finder, PrintDriver(), enrollment=box)
        reply = box.submit(frame)                         # the "photo" is this full-length frame
        cfg = AppConfig()
        cfg.safety.require_scan = False
        ap = Autopilot(cfg)
        for _ in range(10):
            ap.step(platform.observe())
        self.assertEqual(reply.result(timeout=0)["body_looks"], 1)
        self.assertEqual(ap.mode, Mode.TRACK)

        faces.current = []                                # turned around
        modes, sources = [], []
        for _ in range(20):
            obs = platform.observe()
            modes.append(ap.step(obs).mode)
            sources.append(obs.detection.source if obs.detection else None)
        self.assertEqual(set(sources), {"track"})
        self.assertEqual(set(modes), {Mode.TRACK})


if __name__ == "__main__":
    unittest.main()
