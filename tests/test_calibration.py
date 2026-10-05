import json
import math
import os
import tempfile
import unittest

from autonomy.autopilot import Autopilot
from comms.phone_server import CommandBox, CommandError, EnrollmentBox, parse_command
from config import AppConfig, CameraConfig
from control.print_driver import PrintDriver
from datatypes import Detection
from perception.calibration import FovCalibrator, load_calibration, save_calibration
from perception.face_matcher import FaceMatcher
from perception.geometry import CameraModel, fov_from_measurement
from platforms.real import RealPlatform
from tests.test_perception import ALICE, FakeEmbedder, face, img
from tests.test_real_platform import FakeCamera

FRAME = (480, 640)


def seen(size, source="face"):
    return Detection(bbox=(0, 0, 0, 0), offset_x=0.0, offset_y=0.0, size=size, source=source)


def size_for(hfov_deg, aspect, face_height_m, distance_m):
    """What a camera with this FOV would report for a face at this distance --
    straight from CameraModel, so tests check against the real forward model."""
    model = CameraModel(CameraConfig(hfov_deg=hfov_deg, aspect=aspect, face_height_m=face_height_m))
    return model.size_at_1m / distance_m


class CalibratorTests(unittest.TestCase):
    def cal(self, **kw):
        return FovCalibrator(face_height_m=0.22, samples_needed=5, timeout_s=10.0, **kw)

    def test_idle_calibrator_ignores_frames(self):
        c = self.cal()
        self.assertIsNone(c.feed(seen(0.2), FRAME, 0.0))
        self.assertEqual(c.status(), {"state": "idle"})

    def test_recovers_the_true_fov_from_the_forward_model(self):
        c = self.cal()
        c.start(2.0, now=0.0)
        s = size_for(71.0, 0.75, 0.22, 2.0)
        results = [c.feed(seen(s), FRAME, 0.1 * i) for i in range(5)]
        self.assertEqual(results[:4], [None] * 4)
        hfov, aspect = results[4]
        self.assertAlmostEqual(hfov, 71.0, places=6)
        self.assertAlmostEqual(aspect, 0.75)
        self.assertEqual(c.status()["state"], "done")

    def test_aspect_comes_from_the_actual_frame_not_the_config(self):
        c = self.cal()
        c.start(2.0, now=0.0)
        s = size_for(80.0, 720 / 1280, 0.22, 2.0)
        result = None
        for i in range(5):
            result = c.feed(seen(s), (720, 1280), 0.1 * i) or result
        self.assertAlmostEqual(result[1], 720 / 1280)
        self.assertAlmostEqual(result[0], 80.0, places=6)

    def test_result_is_reported_exactly_once(self):
        c = self.cal()
        c.start(2.0, now=0.0)
        s = size_for(66.0, 0.75, 0.22, 2.0)
        out = [c.feed(seen(s), FRAME, 0.1 * i) for i in range(8)]
        self.assertEqual(sum(r is not None for r in out), 1)

    def test_body_only_sightings_and_empty_frames_do_not_count(self):
        c = self.cal()
        c.start(2.0, now=0.0)
        for i in range(10):
            c.feed(seen(0.5, source="track"), FRAME, 0.1 * i)
            c.feed(None, FRAME, 0.1 * i)
        self.assertEqual(c.status()["samples"], 0)

    def test_median_shrugs_off_a_bad_frame(self):
        c = self.cal()
        c.start(2.0, now=0.0)
        s = size_for(66.0, 0.75, 0.22, 2.0)
        sizes = [s, s, s * 3, s, s]              # one wildly wrong box (someone leaned in)
        result = None
        for i, x in enumerate(sizes):
            result = c.feed(seen(x), FRAME, 0.1 * i) or result
        self.assertAlmostEqual(result[0], 66.0, places=6)

    def test_times_out_with_a_useful_reason(self):
        c = self.cal()
        c.start(2.0, now=0.0)
        c.feed(seen(0.2), FRAME, 1.0)
        self.assertIsNone(c.feed(seen(0.2), FRAME, 11.0))
        status = c.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("photo", status["reason"])

    def test_implausible_result_fails_instead_of_being_applied(self):
        c = self.cal()
        c.start(10.0, now=0.0)                   # said 10 m, but the face fills most of the frame
        out = [c.feed(seen(0.9), FRAME, 0.1 * i) for i in range(5)]
        self.assertEqual(out, [None] * 5)
        self.assertEqual(c.status()["state"], "failed")

    def test_distance_out_of_range_is_refused(self):
        with self.assertRaises(ValueError):
            self.cal().start(0.1, now=0.0)
        with self.assertRaises(ValueError):
            self.cal().start(50.0, now=0.0)

    def test_can_recalibrate_after_finishing(self):
        c = self.cal()
        for hfov in (60.0, 75.0):
            c.start(2.0, now=0.0)
            s = size_for(hfov, 0.75, 0.22, 2.0)
            result = None
            for i in range(5):
                result = c.feed(seen(s), FRAME, 0.1 * i) or result
            self.assertAlmostEqual(result[0], hfov, places=6)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "camera_calibration.json")

    def tearDown(self):
        self.dir.cleanup()

    def test_missing_file_means_never_calibrated(self):
        self.assertIsNone(load_calibration(self.path))

    def test_round_trip(self):
        save_calibration(self.path, 71.234, 0.5625, distance_m=2.0)
        data = load_calibration(self.path)
        self.assertAlmostEqual(data["hfov_deg"], 71.23)
        self.assertAlmostEqual(data["aspect"], 0.5625)
        self.assertEqual(data["distance_m"], 2.0)
        self.assertIn("measured_at", data)
        self.assertFalse(os.path.exists(self.path + ".tmp"))   # atomic write cleaned up

    def test_corrupt_file_is_an_error_not_silently_ignored(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        with self.assertRaises(ValueError):
            load_calibration(self.path)

    def test_implausible_values_are_rejected(self):
        with open(self.path, "w") as f:
            json.dump({"hfov_deg": 400, "aspect": 0.75}, f)
        with self.assertRaises(ValueError):
            load_calibration(self.path)


class CommandValidationTests(unittest.TestCase):
    def test_calibrate_command_is_accepted(self):
        self.assertEqual(parse_command({"calibrate_distance_m": "2"}), {"calibrate_distance_m": 2.0})

    def test_calibrate_distance_bounds(self):
        for bad in (0.1, 25, "far"):
            with self.assertRaises(CommandError):
                parse_command({"calibrate_distance_m": bad})

    def test_existing_commands_still_validate(self):
        self.assertEqual(parse_command({"task": "hover"}), {"task": "HOVER"})
        with self.assertRaises(CommandError) as e:
            parse_command({"hover_height_m": -1})
        self.assertEqual(e.exception.code, 422)


class EndToEndTests(unittest.TestCase):
    """Operator taps calibrate -> the platform measures the enrolled face ->
    the autopilot's camera model changes -> the file is written for next boot."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "camera_calibration.json")
        # frame 2: Alice's face is 100 px tall in a 480 x 640 frame
        matcher = FaceMatcher(FakeEmbedder({1: [face((100, 100, 300, 300), ALICE)],
                                            2: [face((280, 150, 360, 250), ALICE)]}))
        self.box, self.commands = EnrollmentBox(), CommandBox()
        self.platform = RealPlatform(FakeCamera(img(2)), matcher, PrintDriver(),
                                     enrollment=self.box, commands=self.commands,
                                     calibrator=FovCalibrator(0.22, samples_needed=5),
                                     calibration_path=self.path)
        cfg = AppConfig()
        cfg.safety.require_scan = False
        self.ap = Autopilot(cfg)

    def tearDown(self):
        self.dir.cleanup()

    def test_one_tap_calibration_updates_the_autopilot_and_disk(self):
        self.box.submit(img(1))                                    # enrol first
        reply = self.commands.submit({"calibrate_distance_m": 2.0})
        fovs = []
        for _ in range(8):
            obs = self.platform.observe()
            fovs.append(obs.camera_fov)
            self.ap.step(obs)
        self.assertEqual(reply.result(timeout=0), {"ok": True, "calibrate_distance_m": 2.0})
        applied = [f for f in fovs if f is not None]
        self.assertEqual(len(applied), 1)

        expected, _ = fov_from_measurement(2.0, 0.22, 100 / 480, 0.75)
        self.assertAlmostEqual(self.ap.cfg.camera.hfov_deg, expected, places=6)
        self.assertAlmostEqual(self.ap.follow.model.tan_h, math.tan(math.radians(expected) / 2))
        self.assertAlmostEqual(load_calibration(self.path)["hfov_deg"], round(expected, 2))
        self.assertEqual(self.platform.calibration_status()["state"], "done")

    def test_refused_while_really_airborne(self):
        class Flying(PrintDriver):
            def airborne(self):
                return True

        self.platform.driver = Flying()
        reply = self.commands.submit({"calibrate_distance_m": 2.0})
        self.platform.observe()
        with self.assertRaises(ValueError) as e:
            reply.result(timeout=0)
        self.assertIn("ground", str(e.exception))
        self.assertEqual(self.platform.calibration_status()["state"], "idle")

    def test_dry_run_driver_counts_as_on_the_ground(self):
        self.assertFalse(PrintDriver().airborne())

    def test_without_an_enrolled_target_it_fails_rather_than_guessing(self):
        self.platform.calibrator.timeout_s = 0.0
        self.commands.submit({"calibrate_distance_m": 2.0})
        for _ in range(3):
            self.assertIsNone(self.platform.observe().camera_fov)
        self.assertEqual(self.platform.calibration_status()["state"], "failed")
        self.assertFalse(os.path.exists(self.path))


if __name__ == "__main__":
    unittest.main()
