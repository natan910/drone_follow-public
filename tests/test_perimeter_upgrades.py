"""Perimeter watch + mission: property filter, privacy mask, patrol profile, alert push (2026-09-27 am),
tracks-based alerts, R1 investigate decisions, R4 pending triggers, R7 wiring, --indoor (2026-09-27 pm).
Fakes only: no models, no cameras, no network (except a local port for the trigger server)."""

import json
import math
import os
import shutil
import tempfile
import unittest
from argparse import Namespace
from types import SimpleNamespace

import numpy as np

from datatypes import Detection, Observation, Pose
from missions.mission_config import PerimeterConfig
from missions.notify import AlertNotifier
from missions.perimeter import Alert, PerimeterWatch, centroid, load_watch_file, privacy_mask
from missions.responder import Responder
from missions.wiring import Mission, PerimeterMission, apply_patrol_profile, make_mission


def pose(x=0.0, y=0.0, yaw_deg=0.0, z=10.0):
    return Pose(x, y, math.radians(yaw_deg), z)


def make_cfg():
    try:
        from config import AppConfig
        return AppConfig()
    except ImportError:
        return SimpleNamespace(camera=SimpleNamespace(hfov_deg=66.0, aspect=0.75, face_height_m=0.22),
                               control=SimpleNamespace(hover_height_above_target_m=0.3),
                               reid=SimpleNamespace(detector="yolo"))


class TempDir(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="df_per_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write_json(self, name, data):
        p = os.path.join(self.dir, name)
        with open(p, "w") as f:
            json.dump(data, f)
        return p


class FakeDetector:
    def __init__(self, found):
        self.found, self.calls = found, 0

    def detect(self, image):
        self.calls += 1
        return list(self.found)


# drone at launch, 10 m up, facing north, camera 45 deg down: frame centre = 10 m north
PERSON_AT_10N = ((300, 180, 340, 240), 0.9)          # feet at (320, 240)
ZONE = [{"name": "gate", "polygon": [(-5, 5), (5, 5), (5, 15), (-5, 15)]}]
YARD = [(-8, -8), (8, -8), (8, 12), (-8, 12)]           # property: 10 N inside
SMALL_YARD = [(-8, -8), (8, -8), (8, 6), (-8, 6)]       # property: 10 N is outside (the street)


# ---- zones file -----------------------------------------------------------------

class TestWatchFile(TempDir):
    def test_areas_and_property(self):
        p = self.write_json("z.json", {"property": {"polygon": YARD}, "areas": [{"name": "gate", "polygon": ZONE[0]["polygon"]}]})
        zones, prop = load_watch_file(p)
        self.assertEqual([z["name"] for z in zones], ["gate"])
        self.assertEqual(len(prop), 4)

    def test_property_alone_is_the_zone(self):
        zones, prop = load_watch_file(self.write_json("z.json", {"property": {"polygon": YARD}}))
        self.assertEqual(zones[0]["name"], "property")
        self.assertEqual(zones[0]["polygon"], prop)

    def test_old_files_still_load(self):
        zones, prop = load_watch_file(self.write_json("z.json", {"areas": [{"name": "gate", "polygon": ZONE[0]["polygon"]}]}))
        self.assertIsNone(prop)
        self.assertEqual(len(zones), 1)

    def test_latlon_property_needs_home_and_errors_are_explained(self):
        with self.assertRaises(ValueError):
            load_watch_file(self.write_json("a.json", {"property": {"latlon": [[1, 1], [1, 2], [2, 2]]}}))
        with self.assertRaises(ValueError):
            load_watch_file(self.write_json("b.json", {"home": [0.0, 0.0]}))
        zones, prop = load_watch_file(self.write_json("c.json", {"home": [0.0, 0.0], "property": {
            "latlon": [[0.0, 0.0], [0.0, 0.001], [0.0001, 0.0001]]}}))
        self.assertAlmostEqual(prop[2][1], 11.1, 0)            # 0.0001 deg lat ~ 11 m north

    def test_centroid(self):
        self.assertEqual(centroid([(0, 0), (4, 0), (4, 2), (0, 2)]), (2.0, 1.0))
        self.assertEqual(centroid([(0, 0), (1, 1), (2, 2)]), (1.0, 1.0))       # degenerate: plain average


# ---- privacy mask ---------------------------------------------------------------

class TestPrivacyMask(unittest.TestCase):
    def test_sky_is_always_masked(self):
        keep = privacy_mask(64, 48, pose(), 0, 66, 0.75, None, cell_px=8)   # camera level
        self.assertEqual(keep.shape, (48, 64))
        self.assertFalse(keep[:24].any())                      # top half = above the horizon
        self.assertTrue(keep[40:].any())                       # bottom rows see the ground

    def test_ground_outside_the_property_is_masked(self):
        keep = privacy_mask(64, 48, pose(), 45, 66, 0.75, SMALL_YARD, cell_px=8)
        self.assertFalse(keep[20:28, 28:36].any())             # frame centre = 10 m N = outside
        self.assertTrue(keep[44:, 28:36].all())                # bottom = near the drone = inside
        whole = privacy_mask(64, 48, pose(), 90, 66, 0.75, None, cell_px=8)
        self.assertTrue(whole.all())                           # straight down, no property: all ground


# ---- the watch with a property ----------------------------------------------------

class TestPerimeterWithProperty(TempDir):
    def run_watch(self, prop, zones=ZONE, **kw):
        saved, got = [], []
        cfg = PerimeterConfig(confirm_s=1.0, detect_every_s=0.5, mask_cell_px=8, **kw)
        w = PerimeterWatch(FakeDetector([PERSON_AT_10N]), zones, 66, 0.75, cfg, alerts_dir=self.dir,
                           imwrite=lambda p, img: saved.append((p, img.copy())) or True,
                           property_polygon=prop, on_alert=[got.append])
        frame = np.full((480, 640, 3), 200, np.uint8)
        alerts = []
        for k in range(30):
            alerts += w.step(frame, Observation(now=k * 0.1, pose=pose(), camera_pitch_deg=45))
        return w, alerts, saved, got

    def test_person_on_the_street_is_not_an_alarm(self):
        w, alerts, _, got = self.run_watch(SMALL_YARD, zones=[{"name": "property", "polygon": SMALL_YARD}])
        self.assertEqual(alerts, [])
        self.assertEqual(got, [])
        self.assertGreater(w.status(3.0)["outside_ignored"], 0)

    def test_person_in_the_yard_alerts_callback_and_masked_snapshot(self):
        w, alerts, saved, got = self.run_watch(YARD)
        self.assertEqual(len(alerts), 1)
        self.assertEqual((alerts[0].kind, alerts[0].track), ("intrusion", "T1"))
        self.assertIs(got[0], alerts[0])                       # on_alert got the same Alert
        (_, img), = saved
        self.assertEqual(int(img[0, 0].max()), 0)              # top-left corner: beyond the property -> black
        self.assertGreater(int(img[470, 320].max()), 0)        # just below the drone: kept
        st = w.status(3.0)
        self.assertTrue(st["property"])
        self.assertEqual(st["tracks"][0]["id"], "T1")
        self.assertEqual(st["occupied"], ["gate"])
        self.assertIn("alert", [e["type"] for e in st["events"]])
        json.dumps(st)

    def test_mask_can_be_switched_off(self):
        _, _, saved, _ = self.run_watch(YARD, privacy_mask=False)
        self.assertGreater(int(saved[0][1][0, 0].max()), 0)

    def test_a_broken_callback_never_stops_the_watch(self):
        cfg = PerimeterConfig(confirm_s=1.0, detect_every_s=0.5)

        def boom(a):
            raise RuntimeError("no network")

        w = PerimeterWatch(FakeDetector([PERSON_AT_10N]), ZONE, 66, 0.75, cfg, on_alert=[boom])
        frame = np.zeros((480, 640, 3), np.uint8)
        alerts = sum((w.step(frame, Observation(now=k * 0.1, pose=pose(), camera_pitch_deg=45)) for k in range(30)), [])
        self.assertEqual(len(alerts), 1)

    def test_owner_still_skipped(self):
        cfg = PerimeterConfig(confirm_s=1.0, detect_every_s=0.5)
        w = PerimeterWatch(FakeDetector([PERSON_AT_10N]), ZONE, 66, 0.75, cfg, property_polygon=YARD)
        owner = Detection((185, 335, 200, 305), 0, 0, 0.03, "face")
        frame = np.zeros((480, 640, 3), np.uint8)
        got = sum((w.step(frame, Observation(now=k * 0.1, pose=pose(), detection=owner, camera_pitch_deg=45))
                   for k in range(30)), [])
        self.assertEqual(got, [])


# ---- notifier ---------------------------------------------------------------------

class FakeSender:
    def __init__(self, codes):
        self.codes, self.calls = list(codes), []

    def __call__(self, method, url, body, headers, timeout_s):
        self.calls.append((method, url, body, dict(headers)))
        c = self.codes.pop(0) if self.codes else 200
        if isinstance(c, Exception):
            raise c
        return c


def alert(zone="gate", snapshot=None, **kw):
    return Alert(zone, 1.0, 0.4, 10.2, 1, wall_time=0.0, snapshot=snapshot, **kw)


class TestNotifier(unittest.TestCase):
    def make(self, codes, **kw):
        s, sleeps = FakeSender(codes), []
        n = AlertNotifier("https://push.example/yard", token="tk_x", send=s, sleep=sleeps.append,
                          threaded=False, drone_id="d1", **kw)
        return n, s, sleeps

    def test_text_alert(self):
        n, s, _ = self.make([200])
        n.submit(alert())
        n.drain()
        (method, url, body, h), = s.calls
        self.assertEqual((method, url), ("POST", "https://push.example/yard"))
        self.assertIn(b"gate", body)
        self.assertEqual(h["Authorization"], "Bearer tk_x")
        self.assertEqual(h["Title"], "d1: person in gate")
        self.assertEqual(h["Priority"], "high")
        self.assertEqual(n.stats()["sent"], 1)

    def test_object_changes_are_not_urgent(self):
        n, s, _ = self.make([200])
        n.submit(alert(zone="study", kind="object_missing", detail="laptop"))
        n.drain()
        self.assertEqual(s.calls[0][3]["Title"], "d1: laptop missing (study)")
        self.assertEqual(s.calls[0][3]["Priority"], "default")

    def test_photo_alert_goes_as_the_body(self):
        n = AlertNotifier("https://push.example/yard", send=FakeSender([200]), sleep=lambda s: None,
                          threaded=False, read_file=lambda p: b"\xff\xd8jpeg")
        n.submit(alert(snapshot="alerts/20260927-120000_intrusion_gate.jpg"))
        n.drain()
        method, _, body, h = n._send.calls[0]
        self.assertEqual((method, body), ("PUT", b"\xff\xd8jpeg"))
        self.assertEqual(h["Filename"], "20260927-120000_intrusion_gate.jpg")
        self.assertIn("gate", h["Message"])
        self.assertNotIn("Authorization", h)

    def test_retries_then_counts_a_failure(self):
        cfg = PerimeterConfig(notify_retries=3, notify_retry_s=2.0)
        n, s, sleeps = self.make([OSError("down"), 500, 503], config=cfg)
        n.submit(alert())
        n.drain()
        self.assertEqual(len(s.calls), 3)
        self.assertEqual(sleeps, [2.0, 2.0])
        self.assertEqual((n.stats()["sent"], n.stats()["failed"]), (0, 1))
        self.assertEqual(n.stats()["last_error"], "HTTP 503")

    def test_recovers_on_retry(self):
        n, s, _ = self.make([OSError("blip"), 200])
        n.submit(alert())
        n.drain()
        self.assertEqual((n.stats()["sent"], n.stats()["failed"], n.stats()["last_error"]), (1, 0, None))

    def test_full_queue_drops_the_oldest(self):
        n, s, _ = self.make([], config=PerimeterConfig(notify_queue=2))
        for z in ("a", "b", "c"):
            n.submit(alert(zone=z))
        self.assertEqual(n.stats()["dropped"], 1)
        n.drain()
        self.assertEqual([c[3]["Title"] for c in s.calls], ["d1: person in b", "d1: person in c"])

    def test_non_ascii_zone_and_bad_url(self):
        n, s, _ = self.make([200])
        n.submit(alert(zone="cancello già"))
        n.drain()
        s.calls[0][3]["Title"].encode("latin-1")               # would raise if not header-safe
        with self.assertRaises(ValueError):
            AlertNotifier("push.example/yard", threaded=False)

    def test_threaded_close_is_quick(self):
        n = AlertNotifier("https://push.example/yard", send=FakeSender([200]), sleep=lambda s: None)
        n.submit(alert())
        n.close(timeout_s=3.0)
        self.assertFalse(n._thread.is_alive())


# ---- the mission's decisions (R1, R4) ---------------------------------------------------

class FakeAutopilot:
    def __init__(self):
        self.calls, self.stopped, self._on, self.last_investigation = [], [], False, ""

    def investigate(self, x, y, now, standoff_m=3.0, timeout_s=60.0, reason="", extend_s=None):
        self.calls.append({"x": x, "y": y, "reason": reason, "extend_s": extend_s, "standoff_m": standoff_m})
        self._on = True

    def stop_investigating(self, why="stopped"):
        self.stopped.append(why)
        self._on, self.last_investigation = False, why

    @property
    def investigating(self):
        return self._on


class Scripted:
    """A scene source that returns the scripted ground points."""
    def __init__(self, script):
        self.script = script                    # t -> list of (x, y)

    def step(self, frame, obs):
        from missions.entities import PersonSighting, Scene
        pts = self.script(obs.now)
        return None if pts is None else Scene(obs.now, obs.pose, 45.0, [PersonSighting(x, y, 0.9) for x, y in pts])


class TestMissionDecisions(TempDir):
    def mission(self, script, investigate=True, zones=ZONE):
        pc = PerimeterConfig(confirm_s=1.0, investigate_standoff_m=4.0)
        watch = PerimeterWatch(zones=zones, config=pc, source=Scripted(script), alerts_dir=self.dir,
                               imwrite=lambda p, img: True)
        m = PerimeterMission(watch, investigate=investigate)
        ap = FakeAutopilot()
        m.bind(ap)
        return m, ap

    def steps(self, m, t0, t1, board=None):
        board = board if board is not None else {}
        for k in range(int(t0 * 10), int(t1 * 10)):
            m.step(None, Observation(now=k / 10, pose=pose()), None, board)
        return board

    def test_a_stranger_is_watched_and_a_zone_alert_marks_it(self):
        # someone at 0, 2 (outside the gate) walking north into it at 1 m/s from t=0
        m, ap = self.mission(lambda t: [(0.0, 2.0 + t)] if t < 10 else None)
        board = self.steps(m, 0, 6)
        self.assertEqual(ap.calls[0]["reason"], "stranger T1")        # watched as soon as confirmed
        self.assertEqual(ap.calls[0]["standoff_m"], 4.0)
        self.assertTrue(m.inv["alerted"])                             # then it walked into the gate
        self.assertEqual(ap.calls[-1]["extend_s"], PerimeterConfig().investigate_keep_s)
        self.assertAlmostEqual(ap.calls[-1]["y"], 7.9, delta=0.2)     # following the walk
        st = board["mission"]
        self.assertEqual(st["investigating"]["track"], "T1")
        self.assertIsNotNone(st["sortie"])
        types = [e["type"] for e in m.watch.events.recent]
        self.assertIn("investigate.start", types)
        self.assertGreaterEqual(m.watch.snapshots, 0)
        json.dumps(st)

    def test_investigation_off_means_alerts_only(self):
        m, ap = self.mission(lambda t: [(0.0, 10.0)], investigate=False)
        self.steps(m, 0, 5)
        self.assertEqual(ap.calls, [])
        self.assertEqual(len(m.watch.recent), 1)

    def test_track_end_stops_the_investigation(self):
        m, ap = self.mission(lambda t: [(0.0, 10.0)] if t < 3 else [])
        self.steps(m, 0, 40)                                          # forget_s = 30
        self.assertIn("T1 left", ap.stopped)
        self.assertIsNone(m.inv)
        self.assertIn("investigate.end", [e["type"] for e in m.watch.events.recent])

    def test_autopilot_timeout_is_noticed(self):
        m, ap = self.mission(lambda t: [(0.0, 10.0)] if t < 3 else [])
        self.steps(m, 0, 5)
        ap.stop_investigating("time up")                              # the autopilot's own deadline
        self.steps(m, 5, 6)
        self.assertIsNone(m.inv)
        ends = [e for e in m.watch.events.recent if e["type"] == "investigate.end"]
        self.assertEqual(ends[-1]["why"], "time up")

    def test_sensor_trigger_is_checked_then_a_stranger_takes_over(self):
        m, ap = self.mission(lambda t: [(1.0, 1.0)] if t >= 3 else [], zones=ZONE + [
            {"name": "hall", "polygon": [(0, 0), (2, 0), (2, 2), (0, 2)]}])
        m.responder = Responder(m.watch.zones, is_airborne=m.airborne, events=m.watch.events)
        self.assertFalse(m.airborne())
        m.responder.trigger("hall", "pir-1")
        self.steps(m, 0, 2)
        self.assertTrue(m.airborne())
        self.assertEqual(ap.calls[0]["reason"], "sensor pir-1 at hall")
        self.assertEqual((ap.calls[0]["x"], ap.calls[0]["y"]), (1.0, 1.0))
        self.steps(m, 2, 5)
        self.assertEqual(m.inv["track"], "T1")                        # found someone: watch them instead
        self.assertIn("switched to T1", ap.stopped)

    def test_base_mission_bind(self):
        m = Mission()
        m.bind("ap")
        self.assertEqual(m.autopilot, "ap")


# ---- wiring -----------------------------------------------------------------------

class TestWiring(TempDir):
    def test_profile_sets_patrol_numbers(self):
        cfg = make_cfg()
        if not hasattr(cfg, "patrol"):
            self.skipTest("config without patrol")
        prof = apply_patrol_profile(cfg, PerimeterConfig(), YARD)
        self.assertEqual(cfg.patrol.altitude_m, 7.0)
        self.assertEqual(cfg.patrol.view_range_m, 15.0)
        self.assertAlmostEqual(cfg.patrol.patrol_radius_m, math.hypot(8, 12) + 2.0, 3)
        json.dumps(prof)

    def test_profile_respects_ceiling_and_geofence(self):
        cfg = make_cfg()
        if not hasattr(cfg, "patrol"):
            self.skipTest("config without patrol")
        big = [(-50, -50), (50, -50), (50, 50), (-50, 50)]
        apply_patrol_profile(cfg, PerimeterConfig(patrol_altitude_m=40.0), big)
        self.assertEqual(cfg.patrol.altitude_m, cfg.safety.max_altitude_m - 1.0)
        self.assertEqual(cfg.patrol.patrol_radius_m, cfg.safety.geofence_radius_m - 3.0)

    def test_indoor_preset(self):
        cfg = make_cfg()
        zones = self.write_json("z.json", {"areas": [{"name": "study", "polygon": [[-5, 1], [-1, 1], [-1, 5]]}]})
        m = make_mission(Namespace(mission="perimeter", zones=zones, alerts_dir=None, indoor=True, detect_sync=True),
                         cfg, detector_factory=lambda: FakeDetector([]))
        self.assertFalse(m.watch.cfg.privacy_mask)
        self.assertEqual(m.watch.cfg.investigate_standoff_m, 2.5)
        if hasattr(cfg, "patrol"):
            self.assertEqual(cfg.patrol.altitude_m, 2.0)
        m.close()

    def test_perimeter_mission_with_notifier_events_and_baseline(self):
        cfg = make_cfg()
        zones = self.write_json("z.json", {"property": {"polygon": YARD}})
        made = []

        def notifier(url, **kw):
            n = AlertNotifier(url, send=FakeSender([200]), sleep=lambda s: None, threaded=False, **kw)
            made.append(n)
            return n

        baseline = os.path.join(self.dir, "baseline.json")
        m = make_mission(Namespace(mission="perimeter", zones=zones, alerts_dir=self.dir, patrol_altitude=6.0,
                                   alert_url="https://push.example/yard", alert_token=None, drone_id="d7",
                                   detect_sync=True, baseline=baseline, investigate=True),
                         cfg, detector_factory=lambda: FakeDetector([PERSON_AT_10N]), notifier_factory=notifier)
        ap = FakeAutopilot()
        m.bind(ap)
        board = {}
        frame = np.zeros((480, 640, 3), np.uint8)
        for k in range(40):
            m.step(frame, Observation(now=k * 0.1, pose=pose(), camera_pitch_deg=45), None, board)
        made[0].drain()
        st = board["mission"]
        self.assertNotIn("failed", st)
        self.assertEqual(st["alert_count"], 1)
        self.assertTrue(st["property"])
        self.assertEqual(made[0].stats()["sent"], 1)
        self.assertIn("notify", st)
        self.assertIn("baseline", st)                                 # the injected detector: baseline kept
        self.assertEqual(ap.calls[0]["reason"], "stranger T1")
        if hasattr(cfg, "patrol"):
            self.assertEqual(st["patrol"]["altitude_m"], 6.0)
        json.dumps(st)
        m.close()
        self.assertTrue(os.path.exists(baseline))
        with open(os.path.join(self.dir, "events.jsonl")) as f:
            types = [json.loads(line)["type"] for line in f]
        self.assertEqual(types[0], "sortie.start")
        self.assertEqual(types[-1], "sortie.end")
        self.assertIn("alert", types)

    def test_no_alert_url_no_notifier(self):
        zones = self.write_json("z.json", {"areas": [{"name": "gate", "polygon": ZONE[0]["polygon"]}]})
        m = make_mission(Namespace(mission="perimeter", zones=zones, alerts_dir=None), make_cfg(),
                         detector_factory=lambda: FakeDetector([]))
        self.assertIsNone(m.notifier)
        self.assertIsNone(m.responder)
        m.close()

    def test_responder_needs_a_token_and_starts_a_server(self):
        import socket
        import urllib.request
        with socket.socket() as s:                              # a free port
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        zones = self.write_json("z.json", {"areas": [{"name": "gate", "polygon": ZONE[0]["polygon"]}]})
        base = dict(mission="perimeter", zones=zones, alerts_dir=None, detect_sync=True, responder_port=port,
                    phone_url="http://drone.local:8080")
        with self.assertRaises(SystemExit):
            make_mission(Namespace(**base, token=None), make_cfg(), detector_factory=lambda: FakeDetector([]))
        m = make_mission(Namespace(**base, token="t"), make_cfg(), detector_factory=lambda: FakeDetector([]))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/trigger?token=t&zone=gate&source=pir",
                                        timeout=5) as r:
                self.assertEqual(json.loads(r.read())["zone"], "gate")
            self.assertEqual(m.responder.take_pending()["source"], "pir")
        finally:
            m.close()
        self.assertIsNone(m.trigger_server._httpd)             # stopped


if __name__ == "__main__":
    unittest.main()
