import http.client
import json
import threading
import unittest
import urllib.error
import urllib.request

import cv2
import numpy as np

from comms.phone_server import CommandBox, EnrollmentBox, PhoneServer

TOKEN = "s3cret"


def jpeg(w=200, h=100, value=128) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), value, np.uint8))
    return buf.tobytes()


class FakeMainLoop(threading.Thread):
    """Plays the part of RealPlatform: answers enrolment requests. A black image
    counts as 'no face in this photo'."""

    def __init__(self, box):
        super().__init__(daemon=True)
        self.box, self.stop, self.received = box, False, []

    def run(self):
        while not self.stop:
            req = self.box.poll()
            if req is None:
                threading.Event().wait(0.01)
                continue
            image, reply = req
            self.received.append(image)
            if image is None:
                reply.set_result({"ok": True, "target": False})
            elif image.max() == 0:
                reply.set_exception(ValueError("no face found in that photo"))
            else:
                reply.set_result({"ok": True, "target": True, "faces_in_photo": 1})


class FakeCommandLoop(threading.Thread):
    """Plays the part of RealPlatform servicing a CommandBox: echoes back
    whatever it was asked to do, like the real autopilot hand-off does."""

    def __init__(self, box):
        super().__init__(daemon=True)
        self.box, self.stop, self.received = box, False, []

    def run(self):
        while not self.stop:
            req = self.box.poll()
            if req is None:
                threading.Event().wait(0.01)
                continue
            command, reply = req
            self.received.append(command)
            reply.set_result({"ok": True, **command})


def call(server, method, path, body=None, token=TOKEN, ctype="image/jpeg"):
    url = f"http://127.0.0.1:{server.port}{path}" + (f"?token={token}" if token is not None else "")
    req = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


class PhoneServerTests(unittest.TestCase):
    def setUp(self):
        self.box = EnrollmentBox()
        self.loop = FakeMainLoop(self.box)
        self.loop.start()
        self.server = PhoneServer(self.box, lambda: {"mode": "PATROL"}, TOKEN,
                                  host="127.0.0.1", port=0, max_bytes=200_000, reply_timeout_s=2.0,
                                  verbose=False)
        self.server.start()

    def tearDown(self):
        self.loop.stop = True
        self.server.stop()

    def test_upload_page_is_served_without_a_token(self):
        code, body = call(self.server, "GET", "/", token=None)
        self.assertEqual(code, 200)
        self.assertIn("Send photo", body)

    def test_a_photo_is_enrolled_and_the_phone_gets_the_answer(self):
        code, body = call(self.server, "POST", "/target", jpeg())
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body), {"ok": True, "target": True, "faces_in_photo": 1})
        self.assertEqual(self.loop.received[0].shape[2], 3)

    def test_a_photo_with_no_face_is_reported_clearly(self):
        code, body = call(self.server, "POST", "/target", jpeg(value=0))
        self.assertEqual(code, 422)
        self.assertIn("no face", body)

    def test_the_target_can_be_cleared_mid_flight(self):
        code, body = call(self.server, "DELETE", "/target")
        self.assertEqual((code, json.loads(body)["target"]), (200, False))
        self.assertIsNone(self.loop.received[0])

    def test_wrong_or_missing_token_is_refused_and_nothing_reaches_the_drone(self):
        self.assertEqual(call(self.server, "POST", "/target", jpeg(), token="nope")[0], 401)
        self.assertEqual(call(self.server, "POST", "/target", jpeg(), token=None)[0], 401)
        self.assertEqual(call(self.server, "DELETE", "/target", token="nope")[0], 401)
        self.assertEqual(call(self.server, "GET", "/status", token="nope")[0], 401)
        self.assertEqual(self.loop.received, [])

    def test_token_may_also_arrive_as_a_header(self):
        url = f"http://127.0.0.1:{self.server.port}/status"
        req = urllib.request.Request(url, headers={"X-Token": TOKEN})
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(json.loads(r.read())["mode"], "PATROL")

    def test_oversized_and_garbage_uploads_are_rejected(self):
        self.assertEqual(call(self.server, "POST", "/target", b"x" * 300_000)[0], 413)
        self.assertEqual(call(self.server, "POST", "/target", b"not an image")[0], 415)
        self.assertEqual(call(self.server, "POST", "/target", b"")[0], 400)
        self.assertEqual(self.loop.received, [])

    def test_oversized_photo_always_gets_413_never_a_connection_reset(self):
        # Before the fix the server answered without reading the body, and macOS
        # sometimes reset the connection while the sender was still sending
        # (ConnectionResetError / BrokenPipeError instead of 413).
        for _ in range(20):
            code, body = call(self.server, "POST", "/target", b"x" * 700_000)
            self.assertEqual(code, 413)
            self.assertIn("too large", body)
        self.assertEqual(self.loop.received, [])

    def test_absurdly_large_upload_is_refused_without_reading_it(self):
        # far over DRAIN_FACTOR x limit: answered straight from the headers
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=5)
        try:
            conn.putrequest("POST", f"/target?token={TOKEN}")
            conn.putheader("Content-Length", str(50_000_000))
            conn.endheaders()                      # sends no body at all
            self.assertEqual(conn.getresponse().status, 413)
        finally:
            conn.close()

    def test_garbage_content_length_is_a_400(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=5)
        try:
            conn.putrequest("POST", f"/target?token={TOKEN}")
            conn.putheader("Content-Length", "lots")
            conn.endheaders()
            self.assertEqual(conn.getresponse().status, 400)
        finally:
            conn.close()

    def test_server_keeps_working_after_refusing_a_big_upload(self):
        self.assertEqual(call(self.server, "POST", "/target", b"x" * 300_000)[0], 413)
        self.assertEqual(call(self.server, "POST", "/target", jpeg())[0], 200)

    def test_huge_phone_photos_are_shrunk_before_they_reach_the_matcher(self):
        server = PhoneServer(self.box, dict, TOKEN, host="127.0.0.1", port=0, max_bytes=10_000_000,
                             verbose=False)
        server.start()
        try:
            self.assertEqual(call(server, "POST", "/target", jpeg(w=3000, h=2000))[0], 200)
        finally:
            server.stop()
        self.assertLessEqual(max(self.loop.received[-1].shape[:2]), 1280)

    def test_a_busy_drone_times_out_instead_of_hanging_the_phone(self):
        self.loop.stop = True                      # main loop stops answering
        threading.Event().wait(0.1)
        self.assertEqual(call(self.server, "POST", "/target", jpeg())[0], 504)

    def test_status_reports_what_the_drone_is_doing(self):
        code, body = call(self.server, "GET", "/status")
        self.assertEqual((code, json.loads(body)), (200, {"mode": "PATROL"}))


def call_json(server, method, path, payload, token=TOKEN):
    return call(server, method, path, json.dumps(payload).encode(), token=token,
                ctype="application/json")


class PhoneCommandTests(unittest.TestCase):
    def setUp(self):
        self.box = EnrollmentBox()
        self.commands = CommandBox()
        self.cmd_loop = FakeCommandLoop(self.commands)
        self.cmd_loop.start()
        self.server = PhoneServer(self.box, lambda: {"mode": "TRACK"}, TOKEN,
                                  host="127.0.0.1", port=0, reply_timeout_s=2.0,
                                  verbose=False, commands=self.commands)
        self.server.start()

    def tearDown(self):
        self.cmd_loop.stop = True
        self.server.stop()

    def test_switching_task_reaches_the_main_loop(self):
        code, body = call_json(self.server, "POST", "/command", {"task": "hover"})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body), {"ok": True, "task": "HOVER"})
        self.assertEqual(self.cmd_loop.received, [{"task": "HOVER"}])

    def test_setting_hover_height_reaches_the_main_loop(self):
        code, body = call_json(self.server, "POST", "/command", {"hover_height_m": 0.5})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body), {"ok": True, "hover_height_m": 0.5})
        self.assertEqual(self.cmd_loop.received, [{"hover_height_m": 0.5}])

    def test_task_and_height_can_be_set_together(self):
        call_json(self.server, "POST", "/command", {"task": "FOLLOW", "hover_height_m": 0.4})
        self.assertEqual(self.cmd_loop.received, [{"task": "FOLLOW", "hover_height_m": 0.4}])

    def test_unknown_task_is_rejected(self):
        code, body = call_json(self.server, "POST", "/command", {"task": "FLY_TO_MOON"})
        self.assertEqual(code, 422)
        self.assertIn("unknown task", body)
        self.assertEqual(self.cmd_loop.received, [])

    def test_negative_hover_height_is_rejected(self):
        code, _ = call_json(self.server, "POST", "/command", {"hover_height_m": -1})
        self.assertEqual(code, 422)
        self.assertEqual(self.cmd_loop.received, [])

    def test_empty_command_is_rejected(self):
        code, _ = call_json(self.server, "POST", "/command", {})
        self.assertEqual(code, 400)

    def test_oversized_command_is_a_400_not_a_reset(self):
        for _ in range(10):
            code, _ = call(self.server, "POST", "/command", b"x" * 30_000, ctype="application/json")
            self.assertEqual(code, 400)
        self.assertEqual(self.cmd_loop.received, [])

    def test_garbage_json_is_rejected(self):
        code, _ = call(self.server, "POST", "/command", b"not json", ctype="application/json")
        self.assertEqual(code, 400)

    def test_command_needs_the_token(self):
        code, _ = call_json(self.server, "POST", "/command", {"task": "HOLD"}, token="nope")
        self.assertEqual(code, 401)
        self.assertEqual(self.cmd_loop.received, [])

    def test_command_endpoint_requires_a_configured_command_box(self):
        server = PhoneServer(self.box, dict, TOKEN, host="127.0.0.1", port=0, verbose=False)
        server.start()
        try:
            code, _ = call_json(server, "POST", "/command", {"task": "HOLD"})
            self.assertEqual(code, 501)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
