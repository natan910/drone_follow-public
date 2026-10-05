"""R9 missions/detect.py (worker thread, one COCO pass, frame-time pose) and R4 missions/responder.py
(sensor HTTP endpoint, pending trigger, Launch button). Fakes only; the HTTP tests use a local port."""

import json
import math
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np

from datatypes import Detection, Observation, Pose
from missions.detect import CocoSceneDetector, DetectorWorker, ImageSceneSource
from missions.mission_config import COCO_NAMES, PerimeterConfig
from missions.notify import AlertNotifier, format_actions
from missions.responder import Responder, TriggerServer, launch_actions

ZONES = [{"name": "back gate", "polygon": [(-5, 5), (5, 5), (5, 15), (-5, 15)]},
         {"name": "hall", "polygon": [(0, 0), (2, 0), (2, 2), (0, 2)]}]


def pose(x=0.0, y=0.0, yaw_deg=0.0, z=10.0):
    return Pose(x, y, math.radians(yaw_deg), z)


class FakeYoloxNet:
    """(1, 8400, 85) YOLOX-640 output; objects = [(class, grid x, grid y, size in stride-32 cells)]."""

    def __init__(self, objects):
        self.raw = np.zeros((1, 8400, 85), np.float32)
        for cls, gx, gy, cells in objects:
            row = 80 * 80 + 40 * 40 + gy * 20 + gx
            self.raw[0, row, :4] = [0.5, 0.5, math.log(cells), math.log(cells)]
            self.raw[0, row, 4] = 0.9
            self.raw[0, row, 5 + cls] = 0.9

    def setInput(self, blob):
        pass

    def forward(self):
        return self.raw


REID = SimpleNamespace(yolo_model="unused", yolo_input_size=640, yolo_format="yolox", yolo_rgb=False,
                       yolo_scale_01=False)


class TestCocoDetector(unittest.TestCase):
    def test_person_sitting_on_a_chair_keeps_both(self):
        chair = COCO_NAMES.index("chair")
        net = FakeYoloxNet([(0, 10, 10, 4.0), (chair, 10, 11, 4.0), (COCO_NAMES.index("cup"), 3, 3, 1.0)])
        det = CocoSceneDetector(REID, ["chair", "laptop"], net=net)
        got = det.detect(np.zeros((640, 640, 3), np.uint8))
        self.assertEqual(sorted(l for _, _, l in got), ["chair", "person"])   # cup: not asked for
        with self.assertRaises(ValueError):
            CocoSceneDetector(REID, ["unicorn"], net=net)


class Slow:
    def __init__(self, s=0.05, found=()):
        self.s, self.found, self.calls = s, list(found), 0

    def detect(self, frame):
        self.calls += 1
        time.sleep(self.s)
        return list(self.found)


class TestWorker(unittest.TestCase):
    def test_threaded_never_blocks_and_skips_while_busy(self):
        w = DetectorWorker(Slow(0.2).detect)
        try:
            t0 = time.perf_counter()
            self.assertTrue(w.submit(np.zeros((4, 4, 3)), {"n": 1}))
            self.assertFalse(w.submit(np.zeros((4, 4, 3)), {"n": 2}))   # busy: skipped
            self.assertLess(time.perf_counter() - t0, 0.05)              # the caller did not wait
            self.assertIsNone(w.poll())
            deadline = time.time() + 3
            r = None
            while r is None and time.time() < deadline:
                time.sleep(0.02)
                r = w.poll()
            self.assertEqual(r[0], {"n": 1})
            st = w.stats()
            self.assertEqual((st["runs"], st["skipped"]), (1, 1))
            self.assertGreaterEqual(st["last_ms"], 150)
        finally:
            w.close()
        self.assertFalse(w._thread.is_alive())

    def test_errors_are_counted_then_raised(self):
        def boom(frame):
            raise RuntimeError("model file corrupt")

        w = DetectorWorker(boom, threaded=False, max_errors=2)
        w.submit(np.zeros(1), {})
        self.assertIsNone(w.poll())
        w.submit(np.zeros(1), {})
        with self.assertRaises(RuntimeError):
            w.poll()


class TestImageSource(unittest.TestCase):
    # drone at launch, 10 m up, facing north, camera 45 deg down: frame centre = 10 m north
    PERSON = ((300, 180, 340, 240), 0.9)

    def test_projection_uses_the_pose_at_capture(self):
        det = Slow(0.15, [self.PERSON])
        src = ImageSceneSource(det, 66, 0.75, PerimeterConfig(detect_every_s=0.0), threaded=True)
        frame = np.zeros((480, 640, 3), np.uint8)
        try:
            self.assertIsNone(src.step(frame, Observation(now=0.0, pose=pose(), camera_pitch_deg=45)))
            scene = None
            deadline = time.time() + 3
            k = 1
            while scene is None and time.time() < deadline:
                time.sleep(0.03)            # meanwhile the drone flew 50 m east and turned
                scene = src.step(None, Observation(now=k * 0.1, pose=pose(50, 0, 90), camera_pitch_deg=10))
                k += 1
            (p,) = scene.people
            self.assertAlmostEqual(p.x, 0.0, places=3)                   # where it was, not where it is
            self.assertAlmostEqual(p.y, 10.0, places=3)
            self.assertEqual(scene.t, 0.0)
        finally:
            src.close()

    def test_owner_is_skipped_and_far_people_counted(self):
        sky = ((300, 10, 340, 60), 0.9)                     # feet above the horizon at 10 deg down? no: counted as far
        det = Slow(0.0, [self.PERSON, sky])
        src = ImageSceneSource(det, 66, 0.75, PerimeterConfig(detect_every_s=0.0, max_range_m=15), threaded=False)
        owner = Detection((185, 335, 200, 305), 0, 0, 0.03, "face")
        s = src.step(np.zeros((480, 640, 3), np.uint8), Observation(now=0, pose=pose(), detection=owner,
                                                                    camera_pitch_deg=45))
        self.assertEqual(s.people, [])
        self.assertEqual(s.far, 1)


class FakeSender:
    def __init__(self):
        self.calls = []

    def __call__(self, method, url, body, headers, timeout_s):
        self.calls.append((method, url, body, dict(headers)))
        return 200


def call(port, method, path, body=None, token="t0k"):
    sep = "&" if "?" in path else "?"
    url = f"http://127.0.0.1:{port}{path}{sep}token={token}" if token else f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestResponder(unittest.TestCase):
    def setUp(self):
        self.sender = FakeSender()
        self.notifier = AlertNotifier("https://push.example/topic", send=self.sender, sleep=lambda s: None,
                                      threaded=False, drone_id="d1")
        self.airborne = False
        self.now = [1000.0]
        self.r = Responder(ZONES, self.notifier, "http://drone.local:8080", "t0k",
                           is_airborne=lambda: self.airborne, clock=lambda: self.now[0])
        self.server = TriggerServer(self.r, "t0k", host="127.0.0.1", port=0, verbose=False)
        self.server.start()

    def tearDown(self):
        self.server.stop()

    def test_on_the_ground_the_push_has_a_launch_button(self):
        code, body = call(self.server.port, "GET", "/trigger?zone=back%20gate&source=pir-1")
        self.assertEqual((code, body["zone"], body["airborne"]), (200, "back gate", False))
        self.notifier.drain()
        (method, url, data, h), = self.sender.calls
        self.assertIn("pir-1 triggered at back gate", h["Title"])
        actions = h["Actions"]
        self.assertIn("http, Launch + check, http://drone.local:8080/command, method=POST", actions)
        self.assertIn("headers.X-Token=t0k", actions)
        self.assertIn('body={"launch": true}', actions)
        self.assertIn("view, Open drone page, http://drone.local:8080/?token=t0k", actions)
        p = self.r.take_pending()
        self.assertEqual(p["zone"], "back gate")
        self.assertAlmostEqual(p["y"], 10.0)                  # the zone's centre
        self.assertIsNone(self.r.take_pending())              # once

    def test_in_the_air_no_button_and_json_post(self):
        self.airborne = True
        code, body = call(self.server.port, "POST", "/trigger", {"zone": "hall", "source": "door"})
        self.assertEqual((code, body["airborne"]), (200, True))
        self.notifier.drain()
        self.assertNotIn("Actions", self.sender.calls[0][3])

    def test_errors_cooldown_and_expiry(self):
        self.assertEqual(call(self.server.port, "GET", "/trigger?zone=hall", token="nope")[0], 401)
        code, body = call(self.server.port, "GET", "/trigger?zone=moon")
        self.assertEqual(code, 422)
        self.assertIn("hall", body["zones"])
        self.assertEqual(call(self.server.port, "GET", "/nothing")[0], 404)
        call(self.server.port, "GET", "/trigger?zone=hall&source=pir")
        self.assertEqual(call(self.server.port, "GET", "/trigger?zone=hall&source=pir")[1]["ignored"], "cooldown")
        self.now[0] += 1000                                   # nobody launched for 1000 s
        self.assertIsNone(self.r.take_pending())
        code, st = call(self.server.port, "GET", "/status")
        self.assertEqual((code, st["triggers"], st["ignored"]), (200, 1, 1))

    def test_too_big_body_is_refused_cleanly(self):
        big = {"zone": "hall", "pad": "x" * 20_000}
        self.assertEqual(call(self.server.port, "POST", "/trigger", big)[0], 413)

    def test_action_quoting(self):
        h = format_actions([{"action": "view", "label": "Open, now", "url": "http://a/b?x=1;2"}])
        self.assertEqual(h, "view, 'Open, now', 'http://a/b?x=1;2'")
        self.assertEqual(len(launch_actions("http://d:8080/", "t")), 2)


if __name__ == "__main__":
    unittest.main()
