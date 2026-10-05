"""
main.py flags from the own-detector session, re-added 2026-09-25:
--person-detector own / --own-model / --own-size, --yolo-size, --record / --record-subject.
Plus the preflight check for our own detector's model file. No hardware, no mocks.
"""

import os
import tempfile
import unittest

import numpy as np

import main
from config import AppConfig
from tools import preflight_check


def configured(*argv):
    args = main.parse_args(list(argv))
    cfg = AppConfig()
    main.configure_reid(args, cfg)
    return args, cfg


class TestDetectorFlags(unittest.TestCase):
    def test_own_detector(self):
        _, cfg = configured("--person-detector", "own", "--own-model", "models/x.onnx", "--own-size", "256")
        self.assertEqual(cfg.reid.detector, "own")
        self.assertEqual(cfg.reid.own_model, "models/x.onnx")
        self.assertEqual(cfg.reid.own_input_size, 256)

    def test_own_defaults_come_from_config(self):
        _, cfg = configured("--person-detector", "own")
        self.assertEqual(cfg.reid.own_model, AppConfig().reid.own_model)
        self.assertEqual(cfg.reid.own_input_size, AppConfig().reid.own_input_size)

    def test_yolox_defaults_to_640(self):
        _, cfg = configured()
        self.assertEqual(cfg.reid.yolo_input_size, 640)

    def test_yolo_size_overrides(self):
        _, cfg = configured("--yolo-size", "416")
        self.assertEqual(cfg.reid.yolo_input_size, 416)
        _, cfg = configured("--yolo-format", "v8", "--yolo-size", "320")
        self.assertEqual(cfg.reid.yolo_input_size, 320)

    def test_own_detector_is_built_from_these_settings(self):
        from perception.target_finder import make_person_detector
        _, cfg = configured("--person-detector", "own", "--own-model", "does/not/exist.onnx")
        with self.assertRaises(FileNotFoundError) as cm:        # proves the "own" branch ran with our path
            make_person_detector(cfg.reid)
        self.assertIn("does/not/exist.onnx", str(cm.exception))


class TestRecordFlags(unittest.TestCase):
    def test_off_by_default(self):
        self.assertIsNone(main.make_recorder(main.parse_args([])))

    def test_records_a_frame_under_the_folder(self):
        with tempfile.TemporaryDirectory() as root:
            args = main.parse_args(["--record", root, "--record-subject", "tester"])
            rec = main.make_recorder(args)
            try:
                self.assertTrue(os.path.abspath(rec.dir).startswith(os.path.abspath(root)))
                self.assertTrue(main.safe_record(rec, np.zeros((48, 64, 3), np.uint8), None, None))
            finally:
                rec.close()
            self.assertEqual(rec.status()["frames"], 1)

    def test_safe_record_never_raises(self):
        class Broken:
            def maybe_record(self, *a):
                raise AttributeError("'Observation' object has no attribute 'camera_pitch_deg'")

        self.assertFalse(main.safe_record(Broken(), np.zeros((4, 4, 3), np.uint8), None, None))


class TestPreflightOwnModel(unittest.TestCase):
    def test_checks_own_model_file(self):
        cfg = AppConfig()
        checks = preflight_check.model_file_checks(cfg, "insightface", True, "own", "color",
                                                   exists_fn=lambda p: False)
        self.assertEqual([c.name for c in checks], ["own person detector model"])
        self.assertIn(cfg.reid.own_model, checks[0].detail)

    def test_cli_accepts_own(self):
        self.assertEqual(preflight_check.parse_args(["--person-detector", "own"]).person_detector, "own")


if __name__ == "__main__":
    unittest.main()
