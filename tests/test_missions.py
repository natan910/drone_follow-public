"""missions/, perception/pet_finder.py, perception/veg_index.py, perception/thermal.py, fleet/sectors.py.
Fakes only: no models, no cameras, no network."""

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
from fleet.sectors import assign_sectors, to_shared, too_close
from missions.coverage import (CoverageTracker, lawnmower, parse_latlon, path_length, swath_width,
                               write_waypoints)
from missions.geo import (foot_of, ground_point, latlon_to_local, load_polygons, local_to_latlon,
                          point_in_polygon, polygon_area)
from missions.mission_config import COCO_CLASSES, PerimeterConfig, PetConfig, ThermalConfig
from missions.perimeter import PerimeterWatch, ZoneMonitor
from missions.wiring import Mission, make_mission
from perception.pet_finder import PetFinder, YoloClassesDetector, pet_detection
from perception.thermal import HeatSpotter, LeptonSource, ThermalWatch, centikelvin_to_c
from perception.veg_index import FieldMap, SurveyLogger, colorize, ndvi_bluefilter, summary, vari


def pose(x=0.0, y=0.0, yaw_deg=0.0, z=10.0):
    return Pose(x, y, math.radians(yaw_deg), z)


class TempDir(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="df_mis_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write_json(self, name, data):
        p = os.path.join(self.dir, name)
        with open(p, "w") as f:
            json.dump(data, f)
        return p


# ---- geometry -------------------------------------------------------------------

class TestGeo(unittest.TestCase):
    def test_straight_down_centre_is_below_the_drone(self):
        p = ground_point(320, 240, 640, 480, pose(3, 4), 90, 66)
        self.assertAlmostEqual(p[0], 3, 6)
        self.assertAlmostEqual(p[1], 4, 6)

    def test_45_degrees_lands_one_height_ahead_along_the_heading(self):
        p = ground_point(320, 240, 640, 480, pose(yaw_deg=0), 45, 66)
        self.assertAlmostEqual(p[0], 0, 6)
        self.assertAlmostEqual(p[1], 10, 6)                      # facing north
        p = ground_point(320, 240, 640, 480, pose(yaw_deg=90), 45, 66)
        self.assertAlmostEqual(p[0], 10, 6)                      # facing east
        self.assertAlmostEqual(p[1], 0, 6)

    def test_right_edge_is_to_the_right(self):
        p = ground_point(640, 240, 640, 480, pose(yaw_deg=0), 90, 66)
        self.assertAlmostEqual(p[0], 10 * math.tan(math.radians(33)), 6)   # east of a north-facing drone
        p = ground_point(640, 240, 640, 480, pose(yaw_deg=90), 90, 66)
        self.assertAlmostEqual(p[1], -10 * math.tan(math.radians(33)), 6)  # south of an east-facing drone

    def test_horizon_range_and_ground_level(self):
        self.assertIsNone(ground_point(320, 0, 640, 480, pose(), 0, 66))           # sky
        self.assertIsNone(ground_point(320, 240, 640, 480, pose(z=0.0), 90, 66))   # on the ground
        self.assertIsNone(ground_point(320, 250, 640, 480, pose(), 10, 66, max_range_m=20))

    def test_polygons(self):
        sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertTrue(point_in_polygon(5, 5, sq))
        self.assertFalse(point_in_polygon(15, 5, sq))
        self.assertEqual(polygon_area(sq), 100)
        self.assertEqual(foot_of((10, 20, 30, 80)), (20, 80))

    def test_latlon_roundtrip(self):
        lat, lon = local_to_latlon(120.0, -45.0, 0.0, 0.0)
        x, y = latlon_to_local(lat, lon, 0.0, 0.0)
        self.assertAlmostEqual(x, 120.0, 3)
        self.assertAlmostEqual(y, -45.0, 3)


class TestLoadPolygons(TempDir):
    def test_metres_and_latlon(self):
        a, b = local_to_latlon(0, 0, 0.0, 0.0), local_to_latlon(10, 0, 0.0, 0.0)
        c = local_to_latlon(10, 10, 0.0, 0.0)
        p = self.write_json("z.json", {"home": [0.0, 0.0], "areas": [
            {"name": "gate", "polygon": [[0, 0], [5, 0], [5, 5]]},
            {"name": "shed", "latlon": [list(a), list(b), list(c)]}]})
        zones = load_polygons(p)
        self.assertEqual([z["name"] for z in zones], ["gate", "shed"])
        self.assertAlmostEqual(zones[1]["polygon"][2][0], 10, 2)

    def test_errors_are_explained(self):
        with self.assertRaises(ValueError):
            load_polygons(self.write_json("a.json", {"areas": [{"name": "x", "latlon": [[1, 1], [1, 2], [2, 2]]}]}))
        with self.assertRaises(ValueError):
            load_polygons(self.write_json("b.json", {"areas": [{"name": "x", "polygon": [[0, 0], [1, 1]]}]}))


# ---- mission 1: perimeter ------------------------------------------------------

ZONE = [{"name": "gate", "polygon": [(-5, 5), (5, 5), (5, 15), (-5, 15)]}]


class TestZoneMonitor(unittest.TestCase):
    def test_confirm_then_cooldown(self):
        m = ZoneMonitor(ZONE, PerimeterConfig(confirm_s=1.5, gap_s=3, cooldown_s=60))
        self.assertEqual(m.update(0, [(0, 10)]), [])
        self.assertEqual(m.update(1, [(0, 10)]), [])
        a = m.update(2, [(0, 10), (1, 11), (50, 50)])
        self.assertEqual((len(a), a[0].zone, a[0].people), (1, "gate", 2))
        self.assertEqual(m.update(3, [(0, 10)]), [])                  # cooldown
        self.assertEqual(m.occupied(3), ["gate"])

    def test_one_glimpse_is_not_an_alarm_and_gaps_reset(self):
        m = ZoneMonitor(ZONE, PerimeterConfig(confirm_s=1.5, gap_s=3))
        m.update(0, [(0, 10)])
        m.update(1, [])
        m.update(10, [])                                              # long gap: timer reset
        self.assertEqual(m.update(11, [(0, 10)]), [])
        self.assertEqual(len(m.update(12.6, [(0, 10)])), 1)

    def test_outside_never_alerts(self):
        m = ZoneMonitor(ZONE, PerimeterConfig(confirm_s=0))
        self.assertEqual(m.update(0, [(20, 20)]), [])


class FakeDetector:
    def __init__(self, found):
        self.found, self.calls = found, 0

    def detect(self, image):
        self.calls += 1
        return list(self.found)


class TestPerimeterWatch(TempDir):
    # drone at the launch point, 10 m up, facing north, camera 45 deg down:
    # the frame centre is 10 m north, inside the "gate" zone
    PERSON = ((300, 180, 340, 240), 0.9)       # feet at the frame centre (320, 240)

    def make(self, found, **kw):
        cfg = PerimeterConfig(confirm_s=1.0, detect_every_s=0.5, **kw)
        saved = []
        w = PerimeterWatch(FakeDetector(found), ZONE, 66, 0.75, cfg, alerts_dir=self.dir,
                           imwrite=lambda p, img: saved.append(p) or True)
        return w, saved

    def obs(self, t, det=None):
        return Observation(now=t, pose=pose(), detection=det, camera_pitch_deg=45)

    def test_alert_with_snapshot_and_rate_limited_detector(self):
        w, saved = self.make([self.PERSON])
        frame = np.zeros((480, 640, 3), np.uint8)
        alerts = []
        for k in range(30):                                          # 3 s at 10 Hz
            alerts += w.step(frame, self.obs(k * 0.1))
        self.assertEqual(len(alerts), 1)
        self.assertAlmostEqual(alerts[0].y, 10, 1)
        self.assertLessEqual(w.detector.calls, 7)                    # every 0.5 s, not every loop
        self.assertEqual(len(saved), 1)
        with open(saved[0][:-4] + ".json") as f:
            self.assertEqual(json.load(f)["zone"], "gate")
        st = w.status(3.0)
        self.assertEqual(st["alert_count"], 1)
        json.dumps(st)                                               # the status page can send it

    def test_the_owner_is_not_an_intruder(self):
        w, _ = self.make([self.PERSON])
        owner = Detection((185, 335, 200, 305), 0, 0, 0.03, "face")   # a face inside the person's box
        frame = np.zeros((480, 640, 3), np.uint8)
        alerts = []
        for k in range(30):
            alerts += w.step(frame, self.obs(k * 0.1, det=owner))
        self.assertEqual(alerts, [])

    def test_low_scores_are_ignored(self):
        w, _ = self.make([(self.PERSON[0], 0.2)])
        frame = np.zeros((480, 640, 3), np.uint8)
        self.assertEqual(sum((w.step(frame, self.obs(k * 0.1)) for k in range(30)), []), [])


# ---- mission 2: pets --------------------------------------------------------------

class FakeYoloxNet:
    """(1, 8400, 85) YOLOX-640 output with one object in the stride-32 grid."""

    def __init__(self, cls, gx=10, gy=10, size_cells=4.0, score=0.9):
        self.raw = np.zeros((1, 8400, 85), np.float32)
        row = 80 * 80 + 40 * 40 + gy * 20 + gx
        self.raw[0, row, :4] = [0.5, 0.5, math.log(size_cells), math.log(size_cells)]
        self.raw[0, row, 4] = score
        self.raw[0, row, 5 + cls] = score

    def setInput(self, blob):
        self.blob = blob

    def forward(self):
        return self.raw


class TestYoloClasses(unittest.TestCase):
    def test_keeps_only_the_asked_classes(self):
        img = np.zeros((640, 640, 3), np.uint8)
        dog = YoloClassesDetector("unused", [COCO_CLASSES["dog"]], net=FakeYoloxNet(COCO_CLASSES["dog"]))
        found = dog.detect(img)
        self.assertEqual(len(found), 1)
        (l, t, r, b), s = found[0]
        self.assertEqual((l + r) // 2, 336)                          # (10 + 0.5) * 32
        self.assertAlmostEqual(r - l, 128, delta=2)                  # 4 cells * 32 px
        cat = YoloClassesDetector("unused", [COCO_CLASSES["cat"]], net=FakeYoloxNet(COCO_CLASSES["dog"]))
        self.assertEqual(cat.detect(img), [])


class ColourEmbedder:
    """Mean colour of the box, centred on grey: different colours point different ways."""
    default_acquire, default_keep = 0.8, 0.7

    def embed(self, frame, boxes):
        return np.stack([frame[t:b, l:r].reshape(-1, 3).mean(0) - 128.0 for l, t, r, b in boxes])


BROWN, WHITE = (40, 60, 120), (230, 230, 230)
B_BOX, W_BOX = (20, 100, 120, 180), (300, 100, 400, 180)


def scene(*pets):
    img = np.full((240, 480, 3), 128, np.uint8)
    for (l, t, r, b), colour in pets:
        img[t:b, l:r] = colour
    return img


class TestPetFinder(unittest.TestCase):
    def make(self, boxes):
        det = FakeDetector([(b, 0.9) for b in boxes])
        return PetFinder(det, ColourEmbedder(), PetConfig(body_height_m=0.5), clock=lambda: 0.0), det

    def test_enrol_then_pick_the_right_dog(self):
        f, det = self.make([B_BOX])
        self.assertFalse(f.has_target)
        f.set_target(scene((B_BOX, BROWN)))
        self.assertTrue(f.has_target)
        det.found = [(W_BOX, 0.95), (B_BOX, 0.9)]
        d = f.find(scene((B_BOX, BROWN), (W_BOX, WHITE)))
        self.assertEqual(d.bbox, (100, 120, 180, 20))               # the brown one, (t, r, b, l)
        self.assertEqual(d.source, "face")                          # confident = identity confirmed
        self.assertLess(d.offset_x, 0)                              # left half of the frame

    def test_stranger_only_is_not_followed(self):
        f, det = self.make([B_BOX])
        f.set_target(scene((B_BOX, BROWN)))
        det.found = [(W_BOX, 0.9)]
        self.assertIsNone(f.find(scene((W_BOX, WHITE))))
        self.assertIsNotNone(f.stats["best_sim"])

    def test_no_animal_in_the_photo(self):
        f, det = self.make([])
        with self.assertRaises(ValueError):
            f.set_target(scene())

    def test_distance_proxy_scales_with_body_height(self):
        d = pet_detection((0, 0, 50, 100), 480, 240, body_height_m=0.5, face_height_m=0.22, source="track")
        self.assertAlmostEqual(d.size, 100 * 0.22 / 0.5 / 240, 6)


# ---- mission 5: survey ------------------------------------------------------------

class TestCoverage(TempDir):
    def test_rectangle_passes(self):
        wps = lawnmower([(0, 0), (20, 0), (20, 10), (0, 10)], 5.0)
        self.assertEqual(len(wps), 4)                                # 2 passes along the long side
        self.assertEqual([round(p[1], 3) for p in wps], [2.5, 2.5, 7.5, 7.5])
        self.assertEqual([round(p[0], 3) for p in wps], [0, 20, 20, 0])   # back and forth

    def test_rotated_field_stays_inside(self):
        a = math.radians(30)
        base = [(0, 0), (40, 0), (40, 15), (0, 15)]
        poly = [(x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)) for x, y in base]
        wps = lawnmower(poly, 4.0, inset_m=0.5)
        self.assertEqual(len(wps), 2 * math.ceil(15 / 4.0))
        for x, y in wps:
            # rotate back: every waypoint is inside the original rectangle
            u, v = x * math.cos(-a) - y * math.sin(-a), x * math.sin(-a) + y * math.cos(-a)
            self.assertTrue(-1e-6 <= u <= 40 + 1e-6 and -1e-6 <= v <= 15 + 1e-6, (u, v))

    def test_thin_field_gets_one_pass(self):
        self.assertEqual(len(lawnmower([(0, 0), (30, 0), (30, 2), (0, 2)], 8.0)), 2)

    def test_swath_tracker_and_waypoint_file(self):
        self.assertAlmostEqual(swath_width(10, 90, 0.0), 20.0, 6)
        wps = lawnmower([(0, 0), (20, 0), (20, 10), (0, 10)], 5.0)
        t = CoverageTracker(wps, 1.0)
        self.assertEqual(t.update(0, 2.5), wps[1])
        self.assertAlmostEqual(t.progress(), 0.25)
        self.assertGreater(path_length(wps), 40)
        p = os.path.join(self.dir, "s.waypoints")
        n = write_waypoints(p, wps, 0.0, 0.0, 10.0)
        with open(p) as f:
            lines = f.read().splitlines()
        self.assertEqual(lines[0], "QGC WPL 110")
        self.assertEqual(len(lines), n + 1)
        self.assertEqual(lines[2].split("\t")[3], "22")              # takeoff
        self.assertEqual(lines[-1].split("\t")[3], "20")             # return to launch
        lat, lon = float(lines[3].split("\t")[8]), float(lines[3].split("\t")[9])
        x, y = latlon_to_local(lat, lon, 0.0, 0.0)
        self.assertAlmostEqual(x, wps[0][0], 2)
        self.assertEqual(parse_latlon("0.0, 0.0"), (0.0, 0.0))
        with self.assertRaises(ValueError):
            parse_latlon("200,8")


class TestVegIndex(TempDir):
    def test_green_is_positive_red_is_negative(self):
        green = np.zeros((10, 10, 3), np.uint8)
        green[..., 1] = 180
        red = np.zeros((10, 10, 3), np.uint8)
        red[..., 2] = 180
        self.assertGreater(float(vari(green).mean()), 0.5)
        self.assertLess(float(vari(red).mean()), -0.5)
        nir = np.zeros((10, 10, 3), np.uint8)
        nir[..., 2], nir[..., 0] = 200, 40                          # NoIR + blue filter: plants are "red"
        self.assertGreater(float(ndvi_bluefilter(nir).mean()), 0.5)
        self.assertEqual(float(vari(np.zeros((4, 4, 3), np.uint8)).max()), 0.0)   # black: no divide-by-zero
        self.assertEqual(colorize(vari(green)).shape, (10, 10, 3))
        self.assertEqual(summary(vari(green))["green_frac"], 1.0)

    def test_field_map_save_load_coverage(self):
        m = FieldMap(2.0)
        m.add(1, 1, 0.2, 100.0)
        m.add(1.5, 1.5, 0.4, 200.0)
        self.assertAlmostEqual(m.mean_at(1, 1), 0.3)
        p = os.path.join(self.dir, "field.json")
        m.save(p)
        m2 = FieldMap.load(p)
        self.assertAlmostEqual(m2.mean_at(1, 1), 0.3)
        self.assertEqual(m2.cells[(0, 0)][2], 200.0)
        self.assertAlmostEqual(m2.coverage([(0, 0), (4, 0), (4, 4), (0, 4)]), 0.25)
        m2.to_csv(os.path.join(self.dir, "field.csv"))

    def test_logger_needs_a_downward_camera(self):
        lg = SurveyLogger(66, 0.75, polygon=[(-5, -5), (5, -5), (5, 5), (-5, 5)], wall_clock=lambda: 1.0)
        frame = np.zeros((120, 160, 3), np.uint8)
        frame[..., 1] = 150
        level = Observation(now=0, pose=pose(), camera_pitch_deg=20)
        self.assertIsNone(lg.step(frame, level))
        down = Observation(now=1, pose=pose(1, 1), camera_pitch_deg=90)
        r = lg.step(frame, down)
        self.assertGreater(r["mean"], 0.5)
        self.assertEqual((r["x"], r["y"]), (1.0, 1.0))
        self.assertIsNone(lg.step(frame, Observation(now=1.1, pose=pose(), camera_pitch_deg=90)))   # rate limit
        self.assertGreater(lg.status()["coverage_pct"], 0)


# ---- mission 3: thermal ------------------------------------------------------------

def thermal_scene(blobs=(), bg=10.0):
    t = np.full((120, 160), bg, np.float32)
    for (l, tp, r, b), c in blobs:
        t[tp:b, l:r] = c
    return t


class FakeCapture:
    def __init__(self, frames):
        self.frames, self.released = list(frames), False

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self):
        self.released = True


class TestThermal(unittest.TestCase):
    def test_spotter_finds_people_not_engines(self):
        s = HeatSpotter(ThermalConfig())
        blobs = s.detect(thermal_scene([((50, 50, 53, 55), 30.0), ((100, 20, 110, 30), 70.0)]))
        self.assertEqual(len(blobs), 1)
        self.assertEqual(blobs[0].box, (50, 50, 53, 55))
        self.assertAlmostEqual(blobs[0].peak_c, 30.0)
        self.assertEqual(s.detect(thermal_scene(bg=31.0)), [])      # warm everywhere = no blob

    def test_lepton_frames_in_centikelvin(self):
        raw = np.full((120, 160), int((30 + 273.15) * 100), np.uint16)
        src = LeptonSource(capture=FakeCapture([raw, np.zeros((120, 160, 3), np.uint8)]), threaded=False)
        self.assertTrue(src.grab())
        self.assertAlmostEqual(float(src.latest()[0, 0]), 30.0, 1)
        self.assertFalse(src.grab())                                # 8-bit fallback: rejected
        src.close()
        self.assertTrue(src.cap.released)
        self.assertAlmostEqual(float(centikelvin_to_c(np.array([27315], np.uint16))[0]), 0.0, 3)

    def test_watch_confirms_before_reporting(self):
        class Src:
            frames = 5

            def latest(self):
                return thermal_scene([((78, 58, 82, 62), 30.0)])     # warm spot in the middle

        w = ThermalWatch(Src(), ThermalConfig(confirm_hits=3, every_s=0.2, pitch_deg=90))
        got = []
        for k in range(10):
            got += w.step(Observation(now=k * 0.25, pose=pose()))
        self.assertEqual(len(got), 1)                                # 3rd sighting, then cooldown
        self.assertLess(math.hypot(got[0].x, got[0].y), 1.0)         # straight below
        self.assertEqual(w.status()["sighting_count"], 1)


# ---- mission 6: sectors -----------------------------------------------------------

class TestSectors(unittest.TestCase):
    def test_slices_cover_everything_once(self):
        s = assign_sectors(["c", "a", "b"], 30)
        self.assertEqual(sorted(s), ["a", "b", "c"])
        for bearing in range(0, 360, 7):
            x, y = 10 * math.sin(math.radians(bearing)), 10 * math.cos(math.radians(bearing))
            self.assertEqual(sum(sec.contains(x, y) for sec in s.values()), 1, bearing)
        self.assertFalse(s["a"].contains(0, 40))                    # outside the radius
        mx, my = s["b"].middle()
        self.assertTrue(s["b"].contains(mx, my))
        self.assertTrue(assign_sectors(["solo"], 10)["solo"].contains(-3, -3))

    def test_separation(self):
        pos = {"a": (0, 0, 10), "b": (5, 0, 11), "c": (5, 0, 20), "d": to_shared(100, 0, (5, 5)) + (10,)}
        self.assertEqual(too_close(pos), [("a", "b", 5.0)])


# ---- wiring -------------------------------------------------------------------------

def make_cfg():
    try:
        from config import AppConfig
        return AppConfig()
    except ImportError:                     # sandbox without the repo's config.py
        return SimpleNamespace(camera=SimpleNamespace(hfov_deg=66.0, aspect=0.75, face_height_m=0.22),
                               control=SimpleNamespace(hover_height_above_target_m=0.3),
                               reid=SimpleNamespace(detector="yolo"))


class TestWiring(TempDir):
    def test_follow_is_a_no_op(self):
        m = make_mission(Namespace(mission="follow"), make_cfg())
        self.assertIsNone(m.finder)
        board = {}
        m.step(None, None, None, board)
        self.assertEqual(board, {})

    def test_perimeter_needs_zones_then_reports(self):
        cfg = make_cfg()
        with self.assertRaises(SystemExit):
            make_mission(Namespace(mission="perimeter"), cfg)
        zones = self.write_json("z.json", {"areas": [{"name": "gate", "polygon": [[-5, 5], [5, 5], [5, 15], [-5, 15]]}]})
        m = make_mission(Namespace(mission="perimeter", zones=zones, alerts_dir=None), cfg,
                         detector_factory=lambda: FakeDetector([]))
        board = {}
        m.step(np.zeros((48, 64, 3), np.uint8), Observation(now=0, pose=pose()), None, board)
        self.assertEqual(board["mission"]["mission"], "perimeter")

    def test_pet_raises_the_hover_height(self):
        cfg = make_cfg()
        m = make_mission(Namespace(mission="pet", pet_species="dog,cat", pet_height=0.3), cfg,
                         detector_factory=lambda: FakeDetector([]), embedder_factory=ColourEmbedder)
        self.assertIsNotNone(m.finder)
        self.assertEqual(m.finder.cfg.species, ("dog", "cat"))
        self.assertGreaterEqual(cfg.control.hover_height_above_target_m, 3.0)

    def test_survey_writes_waypoints_and_saves_the_map(self):
        field = self.write_json("field.json", {"areas": [{"name": "f", "polygon": [[0, 0], [40, 0], [40, 20], [0, 20]]}]})
        fmap = os.path.join(self.dir, "field_map.json")
        m = make_mission(Namespace(mission="survey", survey_area=field, home_latlon="0.0,0.0",
                                   field_map=fmap, survey_index="vari"), make_cfg())
        self.assertTrue(os.path.exists(os.path.join(self.dir, "survey.waypoints")))
        frame = np.zeros((120, 160, 3), np.uint8)
        frame[..., 1] = 150
        board = {}
        m.step(frame, Observation(now=0, pose=pose(5, 5), camera_pitch_deg=90), None, board)
        self.assertEqual(board["mission"]["readings"], 1)
        m.close()
        self.assertTrue(os.path.exists(fmap) and os.path.exists(fmap[:-5] + ".csv"))

    def test_an_error_switches_the_mission_off_not_the_drone(self):
        class Broken(Mission):
            name = "broken"

            def _step(self, frame, obs, decision):
                raise RuntimeError("camera unplugged")

        m, board = Broken(), {}
        m.step(None, None, None, board)
        m.step(None, None, None, board)                             # no second print, no raise
        self.assertIn("camera unplugged", board["mission"]["failed"])

    def test_attach_gives_the_curator_the_matcher_stats(self):
        from dataset.curator import Curator
        rec = SimpleNamespace(curator=Curator())
        matcher = SimpleNamespace(stats={"best_sim": 0.5})
        Mission().attach(matcher, rec)
        self.assertEqual(rec.curator.stats_fn(), {"best_sim": 0.5})
        Mission().attach(matcher, None)                             # no recorder: nothing to do


if __name__ == "__main__":
    unittest.main()
