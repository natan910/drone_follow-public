"""Our own training data: layout, recorder, labels, tracks, autolabel, export,
the review tool's logic, and tools/data.py. Temp dirs only; no models needed."""

import json
import os
import shutil
import tempfile
import threading
import unittest

import numpy as np

from dataset import layout
from dataset.autolabel import autolabel_session
from dataset.export import export, load, to_coco
from dataset.labels import accept, add_box, box_at, delete_box, kept_people, link_tracks, teacher_row
from dataset.layout import SessionInfo, split_of
from dataset.recorder import Recorder, frame_meta
from dataset.review import Reviewer
from dataset.synthetic import make_dataset, make_session
from datatypes import Decision, Detection, DriveCommand, Mode, Observation, Pose
from perception.person_detector import PersonDetector


class TempDir(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="df_test_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def frame(w=64, h=48, v=100):
    return np.full((h, w, 3), v, np.uint8)


class TestLayout(TempDir):
    def test_session_roundtrip_and_listing(self):
        info = SessionInfo("s1", subject="subject-01", source="webcam", width=640, height=480)
        layout.write_session(self.root, info)
        self.assertEqual(layout.list_sessions(self.root), ["s1"])
        self.assertEqual(layout.read_session(self.root, "s1"), info)

    def test_session_id_is_filesystem_safe(self):
        sid = layout.new_session_id("Na tan/../x", now=0)
        self.assertNotIn("/", sid)
        self.assertNotIn(" ", sid)
        self.assertTrue(layout.new_session_id(None, now=0).endswith("-na"))

    def test_jsonl_skips_a_line_cut_short_by_a_crash(self):
        p = os.path.join(self.root, "x.jsonl")
        with open(p, "w") as f:
            f.write('{"a": 1}\n{"a": 2')
        self.assertEqual(layout.read_jsonl(p), [{"a": 1}])

    def test_split_is_stable_and_overridable(self):
        ids = [f"2026-{i}" for i in range(400)]
        first = [split_of(s) for s in ids]
        self.assertEqual(first, [split_of(s) for s in ids])
        share = first.count("eval") / len(ids)
        self.assertTrue(0.08 < share < 0.25, share)
        some_train = ids[first.index("train")]
        self.assertEqual(split_of(some_train, {"eval": [some_train]}), "eval")


class TestRecorder(TempDir):
    def make(self, **kw):
        clock = Clock()
        kw.setdefault("threaded", False)
        rec = Recorder(self.root, SessionInfo("s1", subject="subject-01"), clock=clock,
                       free_bytes=lambda p: 10 ** 12, **kw)
        return rec, clock

    def test_saves_at_most_fps_frames_per_second(self):
        rec, clock = self.make(fps=4)
        saved = 0
        for k in range(100):                     # 10 s at 10 Hz
            clock.t = k * 0.1
            saved += rec.maybe_record(frame())
        rec.close()
        self.assertTrue(38 <= saved <= 42, saved)
        self.assertEqual(len(layout.read_frames(self.root, "s1")), saved)
        self.assertEqual(len(os.listdir(os.path.join(self.root, "s1", "frames"))), saved)

    def test_records_resolution_and_metadata(self):
        rec, clock = self.make(fps=1)
        obs = Observation(now=0, pose=Pose(1.234, 2.0, 0.5, 3.21), camera_pitch_deg=30,
                          detection=Detection((10, 50, 90, 20), 0.0, 0.0, 0.1, "track"))
        rec.maybe_record(frame(80, 60), obs, Decision(DriveCommand(), Mode.TRACK))
        rec.close()
        self.assertEqual((layout.read_session(self.root, "s1").width,
                          layout.read_session(self.root, "s1").height), (80, 60))
        row = layout.read_frames(self.root, "s1")[0]
        self.assertEqual(row["alt_m"], 3.21)
        self.assertEqual(row["pitch_deg"], 30)
        self.assertEqual(row["mode"], "TRACK")
        self.assertEqual(row["live_box"], [20, 10, 50, 90])
        self.assertEqual(row["live_source"], "track")

    def test_meta_works_without_an_autopilot(self):
        self.assertEqual(frame_meta(), {})

    def test_frame_is_copied_so_the_overlay_cannot_reach_the_saved_image(self):
        rec, clock = self.make(fps=1, threaded=True)
        img = frame(v=50)
        rec.maybe_record(img)
        img[:] = 255                               # the overlay drawing, right after
        rec.close()
        import cv2
        saved = cv2.imread(os.path.join(self.root, "s1", "frames", "000000.jpg"))
        self.assertLess(abs(float(saved.mean()) - 50), 3)

    def test_stops_when_the_disk_is_nearly_full(self):
        clock = Clock()
        rec = Recorder(self.root, SessionInfo("s1"), threaded=False, clock=clock,
                       free_bytes=lambda p: 10 ** 9, min_free_gb=2.0)
        self.assertFalse(rec.maybe_record(frame()))
        self.assertIn("free", rec.stopped)

    def test_stops_at_the_size_cap(self):
        rec, clock = self.make(fps=100, max_gb=1e-9)
        clock.t = 0
        rec.maybe_record(frame())
        clock.t = 1
        self.assertFalse(rec.maybe_record(frame()))
        self.assertIsNotNone(rec.stopped)

    def test_full_queue_drops_instead_of_blocking(self):
        gate = threading.Event()

        class SlowDisk(Recorder):
            def _write(self, *item):
                gate.wait(5)
                super()._write(*item)

        clock = Clock()
        rec = SlowDisk(self.root, SessionInfo("s1"), fps=0, queue_size=1, clock=clock,
                       free_bytes=lambda p: 10 ** 12)
        for k in range(10):
            clock.t = k
            rec.maybe_record(frame())
        gate.set()
        rec.close()
        self.assertGreaterEqual(rec.dropped, 5)
        self.assertEqual(rec.frames + rec.dropped, 10)

    def test_none_frame_is_ignored(self):
        rec, clock = self.make()
        self.assertFalse(rec.maybe_record(None))


class TestLabels(unittest.TestCase):
    def test_teacher_row_and_trusted_boxes(self):
        row = teacher_row("a.jpg", [((0, 0, 10, 20), 0.9), ((5, 5, 9, 9), 0.3)], "yolox")
        self.assertEqual(len(kept_people(row, 0.5)), 1)
        row = add_box(row, (30, 30, 40, 60))
        self.assertEqual(len(kept_people(row, 0.5)), 2)      # human boxes always trusted

    def test_click_is_not_a_box(self):
        row = {"file": "a", "people": []}
        self.assertIs(add_box(row, (5, 5, 6, 6)), row)
        self.assertEqual(add_box(row, (20, 30, 5, 2))["people"][0]["box"], [5, 2, 20, 30])

    def test_box_at_picks_the_smallest_containing_box(self):
        row = {"people": [{"box": [0, 0, 100, 100]}, {"box": [40, 40, 60, 60]}]}
        self.assertEqual(box_at(row, 50, 50), 1)
        self.assertEqual(box_at(row, 10, 10), 0)
        self.assertIsNone(box_at(row, 200, 200))
        self.assertEqual(len(delete_box(row, 1)["people"]), 1)

    def test_accept_promotes_boxes_the_human_left_in(self):
        row = teacher_row("a.jpg", [((0, 0, 10, 20), 0.3)], "yolox")
        done = accept(row, 0.5)
        self.assertTrue(done["verified"])
        self.assertEqual(len(kept_people(done, 0.5)), 1)
        self.assertEqual(done["by"], "yolox+human")
        self.assertEqual(accept(done, 0.5)["by"], "yolox+human")

    def test_tracks_follow_a_walking_person_and_split_on_gaps(self):
        rows, times = [], []
        for k in range(6):
            rows.append({"file": f"{k}", "people": [{"box": [10 + 3 * k, 10, 40 + 3 * k, 90], "score": 0.9},
                                                    {"box": [200, 10, 230, 90], "score": 0.9}]})
            times.append(k * 0.25)
        rows.append({"file": "late", "people": [{"box": [28, 10, 58, 90], "score": 0.9}]})
        times.append(10.0)
        out = link_tracks(rows, times, 0.5)
        walker = {r["people"][0]["track"] for r in out[:6]}
        other = {r["people"][1]["track"] for r in out[:6]}
        self.assertEqual(len(walker), 1)
        self.assertEqual(len(other), 1)
        self.assertNotEqual(walker, other)
        self.assertNotIn(out[6]["people"][0]["track"], walker | other)   # after a 9 s gap: new track

    def test_untrusted_boxes_get_no_track(self):
        out = link_tracks([{"file": "a", "people": [{"box": [0, 0, 9, 9], "score": 0.2}]}], [0.0], 0.5)
        self.assertEqual(out[0]["people"][0]["track"], -1)


class ListDetector(PersonDetector):
    def __init__(self, boxes):
        self.boxes, self.calls = boxes, 0

    def detect(self, image):
        self.calls += 1
        return list(self.boxes)


class TestAutolabel(TempDir):
    def test_labels_new_frames_and_never_touches_verified_ones(self):
        rng = np.random.default_rng(0)
        make_session(self.root, "s1", rng, n_frames=5, subject=0, verified=False)
        labels = layout.read_labels(self.root, "s1")
        # pretend: a human verified frame 0 with no one in it; frames 1-4 have no labels yet
        labels = {"000000.jpg": {**labels["000000.jpg"], "people": [], "verified": True}}
        layout.write_labels(self.root, "s1", labels)
        det = ListDetector([((10, 10, 50, 120), 0.8)])
        r = autolabel_session(self.root, "s1", det, "fake")
        self.assertEqual((r["labelled"], r["kept"]), (4, 1))
        after = layout.read_labels(self.root, "s1")
        self.assertEqual(after["000000.jpg"]["people"], [])
        self.assertEqual(after["000003.jpg"]["by"], "fake")
        self.assertEqual({row["people"][0]["track"] for f, row in after.items() if row["people"]}, {0})
        # second run: nothing new to do; --redo relabels only teacher frames
        self.assertEqual(autolabel_session(self.root, "s1", det, "fake")["labelled"], 0)
        self.assertEqual(autolabel_session(self.root, "s1", det, "fake", redo=True)["labelled"], 4)


class TestExport(TempDir):
    def setUp(self):
        super().setUp()
        make_dataset(self.root, n_frames=10)
        self.out = os.path.join(self.root, "_export")

    def test_splits_by_session_and_counts_add_up(self):
        m = export(self.root, self.out)
        tr, ev = load(self.out, "detector_train.json"), load(self.out, "detector_eval.json")
        self.assertTrue(tr and ev)
        self.assertFalse({i["session"] for i in tr} & {i["session"] for i in ev})
        self.assertEqual(m["counts"]["train"]["frames"], len(tr))
        coco = load(self.out, "coco_train.json")
        self.assertEqual(len(coco["annotations"]), sum(len(i["boxes"]) for i in tr))

    def test_reid_identities(self):
        export(self.root, self.out)
        crops = load(self.out, "reid_train.json")
        ids = {c["id"] for c in crops}
        self.assertTrue(any(i.startswith("subj:") for i in ids))       # subject sessions
        self.assertTrue(any(":t" in i for i in ids))                    # crowd sessions: track ids
        subj = [c for c in crops if c["id"] == "subj:p0"]
        self.assertGreater(len({c["session"] for c in subj}), 1)        # one identity across sessions

    def test_eval_uses_only_verified_frames(self):
        with open(os.path.join(self.root, "splits.json")) as f:
            sid = json.load(f)["eval"][0]
        labels = layout.read_labels(self.root, sid)
        for row in labels.values():
            row["verified"] = False
        layout.write_labels(self.root, sid, labels)
        export(self.root, self.out)
        self.assertNotIn(sid, {i["session"] for i in load(self.out, "detector_eval.json")})
        export(self.root, self.out, eval_verified_only=False)
        self.assertIn(sid, {i["session"] for i in load(self.out, "detector_eval.json")})

    def test_coco_boxes_are_xywh(self):
        c = to_coco([{"image": "a.jpg", "width": 10, "height": 10, "boxes": [[1, 2, 5, 9]]}])
        self.assertEqual(c["annotations"][0]["bbox"], [1, 2, 4, 7])


class TestReviewLogic(TempDir):
    """The window itself needs a screen; the state machine behind it doesn't."""

    def setUp(self):
        super().setUp()
        make_session(self.root, "s1", np.random.default_rng(0), n_frames=4, subject=0, verified=False)
        self.r = Reviewer(self.root, None)

    def test_accept_saves_and_moves_on(self):
        self.r.on_key(32)
        self.assertEqual(self.r.i, 1)
        self.assertTrue(layout.read_labels(self.root, "s1")["000000.jpg"]["verified"])

    def test_draw_delete_undo_and_empty(self):
        self.r.scale = 0.5                                     # window shows the frame at half size
        self.r.on_mouse(1, 10, 10, 0, None)                     # EVENT_LBUTTONDOWN
        self.r.on_mouse(4, 30, 60, 0, None)                     # EVENT_LBUTTONUP
        boxes = [p["box"] for p in self.r.row()["people"]]
        self.assertIn([20, 20, 60, 120], boxes)                 # back in full-frame pixels
        n = len(boxes)
        self.r.on_mouse(2, 20, 40, 0, None)                     # EVENT_RBUTTONDOWN inside the new box
        self.assertEqual(len(self.r.row()["people"]), n - 1)
        self.r.on_key(ord("z"))
        self.assertEqual(len(self.r.row()["people"]), n)
        self.r.on_key(ord("x"))
        self.assertEqual(self.r.row()["people"], [])

    def test_navigation_and_quit(self):
        self.r.on_key(ord("d"))
        self.r.on_key(ord("d"))
        self.r.on_key(ord("a"))
        self.assertEqual(self.r.i, 1)
        self.r.on_key(32)                                        # verify frame 1
        self.r.i = 0
        self.r.on_key(ord("u"))
        self.assertEqual(self.r.i, 2)                            # skips verified frame 1
        self.assertFalse(self.r.on_key(ord("q")))

    def test_render_does_not_crash(self):
        img = self.r.render()
        self.assertEqual(img.ndim, 3)


class TestMainRecordFlags(TempDir):
    def test_main_builds_a_recorder_from_its_flags(self):
        import main
        from argparse import Namespace
        self.assertIsNone(main.make_recorder(Namespace(record=None)))
        args = Namespace(record=self.root, record_subject="subject-01", record_source="drone", camera="pi",
                         camera_index=0, record_fps=2.0, record_max_gb=1.0)
        rec = main.make_recorder(args)
        try:
            self.assertEqual(rec.info.subject, "subject-01")
            self.assertEqual(rec.info.camera, "pi")
            self.assertTrue(rec.info.session_id.endswith("-subject-01"))
        finally:
            rec.close()

    def test_own_detector_flag_reaches_the_config(self):
        import main
        from argparse import Namespace
        from config import AppConfig
        cfg = AppConfig()
        args = Namespace(reid="fused", person_detector="own", yolo_format="yolox", yolo_model=None,
                         reid_model=None, own_model="runs/x/person_own.onnx")
        main.configure_reid(args, cfg)
        self.assertEqual((cfg.reid.detector, cfg.reid.own_model), ("own", "runs/x/person_own.onnx"))


class TestDataCli(TempDir):
    def test_export_and_stats(self):
        import contextlib
        import io
        from tools import data as data_cli
        make_dataset(self.root, n_frames=5)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(data_cli.main(["--root", self.root, "export", "--out",
                                            os.path.join(self.root, "_e")]), 0)
            self.assertEqual(data_cli.main(["--root", self.root, "stats"]), 0)
        self.assertIn("TOTAL eval", buf.getvalue())

    def test_autolabel_without_teacher_file_fails_cleanly(self):
        import contextlib
        import io
        from tools import data as data_cli
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(data_cli.main(["--root", self.root, "autolabel", "--teacher",
                                            os.path.join(self.root, "nope.onnx")]), 1)


if __name__ == "__main__":
    unittest.main()
