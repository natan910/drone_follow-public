import json
import unittest
import urllib.error
import urllib.request

from fleet.server import FleetServer, FleetStore

TOKEN = "s3cret"


def call(server, method, path, body=None, token=TOKEN, ctype="application/json"):
    url = f"http://127.0.0.1:{server.port}{path}" + (f"?token={token}" if token is not None else "")
    data = body if body is None or isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": ctype} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


class FleetStoreTests(unittest.TestCase):
    def test_unknown_drone_is_none(self):
        store = FleetStore()
        self.assertIsNone(store.get("nope", now=100.0, stale_after_s=10.0))

    def test_report_then_get_returns_it_with_age(self):
        store = FleetStore()
        store.report("d1", {"mode": "FOLLOW"}, now=100.0)
        row = store.get("d1", now=103.5, stale_after_s=10.0)
        self.assertEqual(row.drone_id, "d1")
        self.assertEqual(row.status, {"mode": "FOLLOW"})
        self.assertAlmostEqual(row.age_s, 3.5)
        self.assertFalse(row.stale)

    def test_old_report_is_flagged_stale(self):
        store = FleetStore()
        store.report("d1", {"mode": "FOLLOW"}, now=100.0)
        row = store.get("d1", now=200.0, stale_after_s=10.0)
        self.assertTrue(row.stale)

    def test_a_second_report_overwrites_not_appends(self):
        store = FleetStore()
        store.report("d1", {"mode": "FOLLOW"}, now=100.0)
        store.report("d1", {"mode": "HOVER"}, now=101.0)
        row = store.get("d1", now=101.0, stale_after_s=10.0)
        self.assertEqual(row.status, {"mode": "HOVER"})

    def test_snapshot_is_sorted_by_drone_id(self):
        store = FleetStore()
        store.report("bravo", {}, now=1.0)
        store.report("alpha", {}, now=1.0)
        rows = store.snapshot(now=1.0, stale_after_s=10.0)
        self.assertEqual([r.drone_id for r in rows], ["alpha", "bravo"])

    def test_forget_removes_it(self):
        store = FleetStore()
        store.report("d1", {}, now=1.0)
        self.assertTrue(store.forget("d1"))
        self.assertIsNone(store.get("d1", now=1.0, stale_after_s=10.0))
        self.assertFalse(store.forget("d1"))  # already gone


class FleetServerHTTPTests(unittest.TestCase):
    def setUp(self):
        self.now = [1000.0]
        self.store = FleetStore()
        self.server = FleetServer(self.store, TOKEN, host="127.0.0.1", port=0,
                                  stale_after_s=5.0, verbose=False, now=lambda: self.now[0])
        self.server.start()

    def tearDown(self):
        self.server.stop()

    def test_dashboard_page_loads_without_a_token(self):
        status, body = call(self.server, "GET", "/", token=None)
        self.assertEqual(status, 200)
        self.assertIn("Fleet Ops", body)

    def test_report_needs_a_token(self):
        status, _ = call(self.server, "POST", "/report", {"drone_id": "d1"}, token=None)
        self.assertEqual(status, 401)

    def test_wrong_token_is_rejected(self):
        status, _ = call(self.server, "GET", "/fleet", token="nope")
        self.assertEqual(status, 401)

    def test_report_then_list_round_trip(self):
        status, _ = call(self.server, "POST", "/report", {"drone_id": "d1", "mode": "FOLLOW", "battery": 80})
        self.assertEqual(status, 200)
        status, body = call(self.server, "GET", "/fleet")
        self.assertEqual(status, 200)
        rows = json.loads(body)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["drone_id"], "d1")
        self.assertEqual(rows[0]["status"]["mode"], "FOLLOW")
        self.assertEqual(rows[0]["status"]["battery"], 80)
        self.assertNotIn("drone_id", rows[0]["status"])  # not duplicated inside status
        self.assertNotIn("acks", rows[0]["status"])

    def test_report_missing_drone_id_is_rejected(self):
        status, body = call(self.server, "POST", "/report", {"mode": "FOLLOW"})
        self.assertEqual(status, 422)
        self.assertIn("drone_id", body)

    def test_report_empty_drone_id_is_rejected(self):
        status, _ = call(self.server, "POST", "/report", {"drone_id": "  "})
        self.assertEqual(status, 422)

    def test_report_non_object_body_is_rejected(self):
        status, _ = call(self.server, "POST", "/report", [1, 2, 3])
        self.assertEqual(status, 400)

    def test_report_garbage_json_is_rejected(self):
        status, _ = call(self.server, "POST", "/report", body=b"not json", ctype="application/json")
        self.assertEqual(status, 400)

    def test_get_single_drone(self):
        call(self.server, "POST", "/report", {"drone_id": "d1", "mode": "FOLLOW"})
        status, body = call(self.server, "GET", "/fleet/d1")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"]["mode"], "FOLLOW")

    def test_get_unknown_single_drone_is_404(self):
        status, _ = call(self.server, "GET", "/fleet/ghost")
        self.assertEqual(status, 404)

    def test_two_drones_both_show_up(self):
        call(self.server, "POST", "/report", {"drone_id": "d1", "mode": "FOLLOW"})
        call(self.server, "POST", "/report", {"drone_id": "d2", "mode": "PATROL"})
        status, body = call(self.server, "GET", "/fleet")
        rows = json.loads(body)
        self.assertEqual({r["drone_id"] for r in rows}, {"d1", "d2"})

    def test_staleness_flips_as_time_passes(self):
        call(self.server, "POST", "/report", {"drone_id": "d1"})
        status, body = call(self.server, "GET", "/fleet")
        self.assertFalse(json.loads(body)[0]["stale"])

        self.now[0] += 100.0  # past stale_after_s=5.0 with nothing new reported
        status, body = call(self.server, "GET", "/fleet")
        self.assertTrue(json.loads(body)[0]["stale"])

    def test_delete_forgets_a_drone(self):
        call(self.server, "POST", "/report", {"drone_id": "d1"})
        status, _ = call(self.server, "DELETE", "/fleet/d1")
        self.assertEqual(status, 200)
        status, _ = call(self.server, "GET", "/fleet/d1")
        self.assertEqual(status, 404)

    def test_delete_needs_a_token(self):
        call(self.server, "POST", "/report", {"drone_id": "d1"})
        status, _ = call(self.server, "DELETE", "/fleet/d1", token=None)
        self.assertEqual(status, 401)

    def test_unknown_path_is_404(self):
        status, _ = call(self.server, "GET", "/nope")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
