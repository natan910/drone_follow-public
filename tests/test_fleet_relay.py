"""Console -> drone tasking: queued on the fleet server, collected by the
drone in the reply to its own report, handed to the same boxes the phone page
uses, and acknowledged back into the operator log."""

import base64
import json
import unittest
import urllib.error
import urllib.request

import cv2
import numpy as np

from comms.phone_server import CommandBox, EnrollmentBox
from fleet.reporter import FleetReporter
from fleet.server import FleetServer, FleetStore, UnknownDrone

TOKEN = "s3cret"


def jpeg(w=1600, h=1200, value=90) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), value, np.uint8))
    return buf.tobytes()


def call(server, method, path, body=None, token=TOKEN, ctype="application/json"):
    sep = "&" if "?" in path else "?"
    url = f"http://127.0.0.1:{server.port}{path}" + (f"{sep}token={token}" if token else "")
    data = body if body is None or isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": ctype} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


class StoreTaskingTests(unittest.TestCase):
    def setUp(self):
        self.store = FleetStore(wall=lambda: 1_700_000_000.0)
        self.store.report("d1", {"mode": "PATROL"}, now=0.0)

    def test_cannot_task_a_drone_that_never_reported(self):
        with self.assertRaises(UnknownDrone):
            self.store.enqueue("ghost", {"kind": "command", "command": {"task": "HOLD"}}, "task HOLD")

    def test_outbox_is_delivered_once_and_counts_as_pending_until_acked(self):
        item_id = self.store.enqueue("d1", {"kind": "command", "command": {"task": "HOLD"}}, "task HOLD")
        self.assertEqual(self.store.get("d1", 0.0, 10.0).pending, 1)
        out = self.store.take_outbox("d1")
        self.assertEqual(out, [{"id": item_id, "kind": "command", "command": {"task": "HOLD"}}])
        self.assertEqual(self.store.take_outbox("d1"), [])
        self.assertEqual(self.store.get("d1", 0.0, 10.0).pending, 1)      # sent, not yet confirmed
        self.store.ack("d1", [{"id": item_id, "ok": True, "result": {"ok": True, "task": "HOLD"}}])
        self.assertEqual(self.store.get("d1", 0.0, 10.0).pending, 0)

    def test_event_log_tells_the_story(self):
        item_id = self.store.enqueue("d1", {"kind": "clear_target"}, "forget target")
        self.store.take_outbox("d1")
        self.store.ack("d1", [{"id": item_id, "ok": False, "error": "busy"}])
        kinds = [(e["kind"], e["text"]) for e in self.store.events()]
        self.assertEqual(kinds, [("online", "first contact"), ("sent", "forget target"),
                                 ("error", "forget target: busy")])

    def test_events_since_returns_only_newer_ones(self):
        first = self.store.events()[-1]["seq"]
        self.store.enqueue("d1", {"kind": "clear_target"}, "forget target")
        self.assertEqual([e["kind"] for e in self.store.events(since=first)], ["sent"])

    def test_losing_and_regaining_a_drone_is_logged_once_each(self):
        for _ in range(3):
            self.store.snapshot(now=100.0, stale_after_s=10.0)     # console polls repeatedly
        self.store.report("d1", {}, now=101.0)
        kinds = [e["kind"] for e in self.store.events()]
        self.assertEqual(kinds, ["online", "lost", "online"])
        self.assertEqual(self.store.events()[-1]["text"], "signal reacquired")

    def test_malformed_acks_are_ignored(self):
        self.store.ack("d1", ["junk", {"no": "id"}])
        self.assertEqual([e["kind"] for e in self.store.events()], ["online"])


class ConsoleHTTPTests(unittest.TestCase):
    def setUp(self):
        self.server = FleetServer(FleetStore(), TOKEN, host="127.0.0.1", port=0, verbose=False)
        self.server.start()
        call(self.server, "POST", "/report", {"drone_id": "d1", "mode": "PATROL"})

    def tearDown(self):
        self.server.stop()

    def report(self, **extra):
        return call(self.server, "POST", "/report", {"drone_id": "d1", **extra})[1]["outbox"]

    def test_report_reply_carries_an_empty_outbox_by_default(self):
        self.assertEqual(self.report(), [])

    def test_command_round_trip(self):
        code, body = call(self.server, "POST", "/fleet/d1/command", {"task": "hover"})
        self.assertEqual(code, 202)
        outbox = self.report()
        self.assertEqual(outbox, [{"id": body["id"], "kind": "command", "command": {"task": "HOVER"}}])
        self.assertEqual(self.report(), [])

    def test_commands_are_validated_like_the_phone_page(self):
        code, _ = call(self.server, "POST", "/fleet/d1/command", {"task": "BARREL_ROLL"})
        self.assertEqual(code, 422)
        code, _ = call(self.server, "POST", "/fleet/d1/command", {"calibrate_distance_m": 40})
        self.assertEqual(code, 422)
        self.assertEqual(self.report(), [])

    def test_calibrate_command_is_relayed(self):
        call(self.server, "POST", "/fleet/d1/command", {"calibrate_distance_m": 2})
        self.assertEqual(self.report()[0]["command"], {"calibrate_distance_m": 2.0})

    def test_tasking_an_unknown_drone_is_404(self):
        code, _ = call(self.server, "POST", "/fleet/ghost/command", {"task": "HOLD"})
        self.assertEqual(code, 404)

    def test_tasking_needs_the_token(self):
        code, _ = call(self.server, "POST", "/fleet/d1/command", {"task": "HOLD"}, token=None)
        self.assertEqual(code, 401)

    def test_target_photo_is_shrunk_and_relayed_as_a_decodable_jpeg(self):
        code, _ = call(self.server, "POST", "/fleet/d1/target", jpeg(), ctype="image/jpeg")
        self.assertEqual(code, 202)
        item = self.report()[0]
        self.assertEqual(item["kind"], "target")
        raw = base64.b64decode(item["image_jpeg_b64"])
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(max(image.shape[:2]), 1280)

    def test_garbage_photo_is_rejected(self):
        code, _ = call(self.server, "POST", "/fleet/d1/target", b"not an image", ctype="image/jpeg")
        self.assertEqual(code, 415)
        self.assertEqual(self.report(), [])

    def test_forget_target_is_relayed(self):
        call(self.server, "DELETE", "/fleet/d1/target")
        self.assertEqual(self.report()[0]["kind"], "clear_target")

    def test_acks_in_a_report_reach_the_event_log(self):
        _, body = call(self.server, "POST", "/fleet/d1/command", {"task": "HOLD"})
        self.report()
        self.report(acks=[{"id": body["id"], "ok": True, "result": {"ok": True, "task": "HOLD"}}])
        _, events = call(self.server, "GET", "/events?since=0")
        self.assertIn(("ack", "task HOLD: accepted"), [(e["kind"], e["text"]) for e in events["events"]])

    def test_drone_ids_are_restricted_to_url_safe_names(self):
        for bad in ("a/b", "with space", "x" * 41):
            code, _ = call(self.server, "POST", "/report", {"drone_id": bad})
            self.assertEqual(code, 422, bad)


class ReporterDispatchTests(unittest.TestCase):
    """The drone side, with a fake console reply and no network."""

    def make(self, reply_outbox=(), **kw):
        self.sent = []
        outbox = list(reply_outbox)

        def poster(url, token, body, timeout_s):
            self.sent.append(body)
            items, outbox[:] = list(outbox), []
            return {"ok": True, "outbox": items}

        self.clock = [0.0]
        self.box, self.commands = EnrollmentBox(), CommandBox()
        kw.setdefault("enrollment", self.box)
        kw.setdefault("commands", self.commands)
        return FleetReporter("http://x", "d1", "tok", poster=poster, clock=lambda: self.clock[0], **kw)

    def tick(self, r):
        r.update({"mode": "PATROL"})
        r.post_once()

    def test_command_goes_to_the_command_box_and_its_result_is_acked(self):
        r = self.make([{"id": 5, "kind": "command", "command": {"task": "HOLD"}}])
        self.tick(r)
        command, reply = self.commands.poll()
        self.assertEqual(command, {"task": "HOLD"})
        reply.set_result({"ok": True, "task": "HOLD"})
        self.tick(r)
        self.assertEqual(self.sent[-1]["acks"], [{"id": 5, "ok": True, "result": {"ok": True, "task": "HOLD"}}])
        self.tick(r)
        self.assertEqual(self.sent[-1]["acks"], [])                  # delivered once

    def test_target_photo_is_decoded_and_enrolled(self):
        b64 = base64.b64encode(jpeg(64, 48)).decode()
        r = self.make([{"id": 1, "kind": "target", "image_jpeg_b64": b64}])
        self.tick(r)
        image, _ = self.box.poll()
        self.assertEqual(image.shape, (48, 64, 3))

    def test_clear_target_submits_none(self):
        r = self.make([{"id": 1, "kind": "clear_target"}])
        self.tick(r)
        image, _ = self.box.poll()
        self.assertIsNone(image)

    def test_a_failed_request_is_acked_as_an_error(self):
        r = self.make([{"id": 2, "kind": "target", "image_jpeg_b64": base64.b64encode(jpeg(64, 48)).decode()}])
        self.tick(r)
        _, reply = self.box.poll()
        reply.set_exception(ValueError("no face found in that photo"))
        self.tick(r)
        self.assertEqual(self.sent[-1]["acks"], [{"id": 2, "ok": False, "error": "no face found in that photo"}])

    def test_undecodable_photo_is_acked_as_an_error_without_bothering_the_main_loop(self):
        r = self.make([{"id": 3, "kind": "target", "image_jpeg_b64": base64.b64encode(b"nope").decode()}])
        self.tick(r)
        self.assertIsNone(self.box.poll())
        self.tick(r)
        self.assertFalse(self.sent[-1]["acks"][0]["ok"])

    def test_drone_without_command_support_says_so(self):
        r = self.make([{"id": 4, "kind": "command", "command": {"task": "HOLD"}}], commands=None)
        self.tick(r)
        self.tick(r)
        self.assertIn("without command support", self.sent[-1]["acks"][0]["error"])

    def test_main_loop_that_never_answers_times_out(self):
        r = self.make([{"id": 6, "kind": "command", "command": {"task": "HOLD"}}], ack_timeout_s=5.0)
        self.tick(r)
        self.clock[0] = 10.0
        self.tick(r)
        self.assertIn("in time", self.sent[-1]["acks"][0]["error"])

    def test_acks_survive_a_failed_send(self):
        r = self.make([{"id": 7, "kind": "command", "command": {"task": "HOLD"}}])
        self.tick(r)
        _, reply = self.commands.poll()
        reply.set_result({"ok": True})
        real = r._poster

        def down(*a):
            raise ConnectionError("console unreachable")

        r._poster = down
        self.tick(r)                          # the ack is in this lost request...
        r._poster = real
        self.tick(r)
        self.assertEqual([a["id"] for a in self.sent[-1]["acks"]], [7])    # ...so it is sent again


class EndToEndTests(unittest.TestCase):
    """A real console and a real reporter over HTTP, with a fake main loop
    standing in for RealPlatform."""

    def test_operator_task_reaches_the_drone_and_the_ack_reaches_the_log(self):
        server = FleetServer(FleetStore(), TOKEN, host="127.0.0.1", port=0, verbose=False)
        server.start()
        commands = CommandBox()
        reporter = FleetReporter(f"http://127.0.0.1:{server.port}", "d1", TOKEN, commands=commands)
        try:
            reporter.update({"mode": "PATROL", "x": 1.0, "y": 2.0})
            reporter.post_once()                                           # drone appears
            code, _ = call(server, "POST", "/fleet/d1/command", {"task": "RETURN"})
            self.assertEqual(code, 202)
            reporter.update({"mode": "PATROL"})
            reporter.post_once()                                           # collects the task
            command, reply = commands.poll()                               # main loop handles it
            self.assertEqual(command, {"task": "RETURN"})
            reply.set_result({"ok": True, **command})
            reporter.update({"mode": "RETURN"})
            reporter.post_once()                                           # acks it
            _, events = call(server, "GET", "/events")
            self.assertEqual([e["kind"] for e in events["events"]], ["online", "sent", "ack"])
            _, fleet = call(server, "GET", "/fleet")
            self.assertEqual((fleet[0]["status"]["mode"], fleet[0]["pending"]), ("RETURN", 0))
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
