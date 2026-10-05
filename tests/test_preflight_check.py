import io
import contextlib
import unittest
import os
import tempfile

from config import AppConfig
from tools.preflight_check import (
    check_camera,
    check_disk_space,
    check_mavlink,
    check_model_file,
    check_opencv,
    check_python_version,
    check_ranger,
    main,
    model_file_checks,
    parse_args,
    run_all,
)


class FakeCapture:
    def __init__(self, opened=True, frame=(48, 64, 3)):
        self._opened, self._frame_shape, self.released = opened, frame, False

    def isOpened(self):
        return self._opened

    def read(self):
        if self._frame_shape is None:
            return False, None
        import numpy as np
        return True, np.zeros(self._frame_shape, dtype="uint8")

    def release(self):
        self.released = True


class FakeSerial:
    def __init__(self, port, baud):
        if "bad" in port.lower():
            raise OSError(f"could not open {port}")
        self.port, self.closed = port, False

    def close(self):
        self.closed = True


class VersionChecks(unittest.TestCase):
    def test_known_good_version_passes(self):
        r = check_python_version((3, 11, 4))
        self.assertTrue(r.ok)

    def test_other_version_warns_not_fails(self):
        self.assertTrue(check_python_version((3, 13, 5)).ok)   # the Pi 5's system Python
        r = check_python_version((3, 14, 0))
        self.assertFalse(r.ok)
        self.assertTrue(r.warning)  # a warning, not a hard failure


class OpenCVCheck(unittest.TestCase):
    def test_opencv_importable_in_this_environment(self):
        r = check_opencv()
        self.assertTrue(r.ok)
        self.assertIn("cv2", r.detail)


class ModelFileChecks(unittest.TestCase):
    def test_present_file_passes(self):
        r = check_model_file("thing", "models/x.onnx", exists_fn=lambda p: True)
        self.assertTrue(r.ok)

    def test_missing_file_fails_hard(self):
        r = check_model_file("thing", "models/x.onnx", exists_fn=lambda p: False)
        self.assertFalse(r.ok)
        self.assertFalse(r.warning)
        self.assertIn("models/x.onnx", r.detail)

    def test_opencv_backend_needs_face_models_insightface_does_not(self):
        cfg = AppConfig()
        present = model_file_checks(cfg, backend="opencv", reid_enabled=False,
                                     reid_detector="yolo", reid_embedder="color",
                                     exists_fn=lambda p: True)
        absent = model_file_checks(cfg, backend="insightface", reid_enabled=False,
                                    reid_detector="yolo", reid_embedder="color",
                                    exists_fn=lambda p: True)
        self.assertEqual({c.name for c in present}, {"face detector model", "face recognition model"})
        self.assertEqual(absent, [])

    def test_reid_color_needs_no_model_but_yolo_detector_does(self):
        cfg = AppConfig()
        checks = model_file_checks(cfg, backend="insightface", reid_enabled=True,
                                    reid_detector="yolo", reid_embedder="color",
                                    exists_fn=lambda p: True)
        self.assertEqual({c.name for c in checks}, {"person detector model"})

    def test_reid_fused_needs_both_detector_and_reid_model(self):
        cfg = AppConfig()
        checks = model_file_checks(cfg, backend="insightface", reid_enabled=True,
                                    reid_detector="yolo", reid_embedder="fused",
                                    exists_fn=lambda p: True)
        self.assertEqual({c.name for c in checks}, {"person detector model", "body re-ID model"})

    def test_reid_disabled_needs_nothing(self):
        cfg = AppConfig()
        checks = model_file_checks(cfg, backend="insightface", reid_enabled=False,
                                    reid_detector="yolo", reid_embedder="fused",
                                    exists_fn=lambda p: True)
        self.assertEqual(checks, [])


class CameraChecks(unittest.TestCase):
    def test_camera_opens_and_reads_a_frame(self):
        r = check_camera(0, "webcam", opener=lambda i: FakeCapture(opened=True))
        self.assertTrue(r.ok)
        self.assertIn("64x48", r.detail)

    def test_camera_that_wont_open_fails(self):
        r = check_camera(0, "webcam", opener=lambda i: FakeCapture(opened=False))
        self.assertFalse(r.ok)

    def test_camera_that_opens_but_gives_no_frame_fails(self):
        r = check_camera(0, "webcam", opener=lambda i: FakeCapture(opened=True, frame=None))
        self.assertFalse(r.ok)

    def test_camera_is_released_even_on_failure(self):
        cap = FakeCapture(opened=False)
        check_camera(0, "webcam", opener=lambda i: cap)
        self.assertTrue(cap.released)

    def test_pi_camera_not_importable_fails(self):
        def boom(name):
            raise ImportError("No module named 'picamera2'")
        r = check_camera(kind="pi", import_fn=boom)
        self.assertFalse(r.ok)
        self.assertIn("picamera2", r.detail)

    def test_pi_camera_importable_passes (self):
        r = check_camera(kind="pi", import_fn=lambda name: object())
        self.assertTrue(r.ok)
        self.assertIn("picamera2", r.detail)


class RangerChecks(unittest.TestCase):
    def test_no_ports_configured_is_fine(self):
        r = check_ranger({})
        self.assertTrue(r.ok)

    def test_all_ports_open(self):
        r = check_ranger({0.0: "/dev/ttyUSB0", 45.0: "/dev/ttyUSB1"},
                          opener=lambda port, baud: FakeSerial(port, baud))
        self.assertTrue(r.ok)

    def test_one_bad_port_fails_and_names_it(self):
        r = check_ranger({0.0: "/dev/ttyUSB0", 45.0: "/dev/ttyBAD"},
                          opener=lambda port, baud: FakeSerial(port, baud))
        self.assertFalse(r.ok)
        self.assertIn("ttyBAD", r.detail)


class MavlinkChecks(unittest.TestCase):
    def test_heartbeat_seen_passes(self):
        r = check_mavlink("udpin:127.0.0.1:14550", connector=lambda d, b, t: True)
        self.assertTrue(r.ok)

    def test_no_heartbeat_fails(self):
        r = check_mavlink("udpin:127.0.0.1:14550", connector=lambda d, b, t: False)
        self.assertFalse(r.ok)

    def test_connector_exception_fails_cleanly(self):
        def boom(d, b, t):
            raise RuntimeError("port busy")
        r = check_mavlink("/dev/serial0", connector=boom)
        self.assertFalse(r.ok)
        self.assertIn("port busy", r.detail)


class DiskSpaceChecks(unittest.TestCase):
    def test_plenty_of_space_passes(self):
        class Usage:
            free = 5_000_000_000
        r = check_disk_space(min_mb=200, usage_fn=lambda p: Usage())
        self.assertTrue(r.ok)

    def test_low_space_warns_not_fails(self):
        class Usage:
            free = 10_000_000
        r = check_disk_space(min_mb=200, usage_fn=lambda p: Usage())
        self.assertFalse(r.ok)
        self.assertTrue(r.warning)


class RunAllAndMain(unittest.TestCase):
    def test_run_all_skips_hardware_checks_when_asked(self):
        args = parse_args(["--reid", "off", "--backend", "insightface", "--skip-hardware"])
        results = run_all(args)
        names = {r.name for r in results}
        self.assertNotIn("camera (webcam)", names)
        self.assertNotIn("mavlink link", names)

    def test_run_all_includes_ranger_check_when_ranger_given_and_not_skipped(self):
        args = parse_args(["--ranger", "0:/dev/ttyUSB0", "--skip-camera", "--skip-mavlink"])
        results = run_all(args)
        self.assertIn("range sensors", {r.name for r in results})

    def test_run_all_omits_ranger_check_when_no_ranger_given(self):
        args = parse_args(["--skip-camera", "--skip-mavlink"])
        results = run_all(args)
        self.assertNotIn("range sensors", {r.name for r in results})

    def test_main_returns_zero_when_nothing_hard_fails(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["--reid", "off", "--backend", "insightface", "--skip-hardware"])
        self.assertEqual(code, 0)
        self.assertIn("All checks passed", buf.getvalue())

    def test_main_returns_one_on_a_missing_model_file(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            cfg = AppConfig()
            cfg.matcher.yunet_model = os.path.join(empty_dir, "yunet.onnx")
            cfg.matcher.sface_model = os.path.join(empty_dir, "sface.onnx")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(["--backend", "opencv", "--reid", "off", "--skip-hardware"], cfg=cfg)
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] face detector model", buf.getvalue())

if __name__ == "__main__":
    unittest.main()
