"""Body re-identification: embedders, gallery, geometry, motion gate, detectors,
and the TargetFinder that ties them to the face matcher. Synthetic images:
people are painted rectangles (shirt + trousers + a skin-coloured head)."""

import unittest

import cv2
import numpy as np

from config import BodyReIDConfig
from perception.body_reid import (AppearanceGallery, ColorHistogramEmbedder, FaceBodyModel,
                                  FusedEmbedder, MotionGate, OnnxReidEmbedder)
from perception.face_matcher import Embedder, FaceMatcher, FaceObservation, unit
from perception.person_detector import (HOGPersonDetector, PersonDetector, YoloPersonDetector,
                                        decode_yolo, detect_in_roi, letterbox)
from perception.target_finder import TargetFinder

W, H = 640, 480
RED, BLUE, GREEN, BLACK, WHITE = (30, 30, 200), (200, 60, 30), (40, 170, 40), (15, 15, 15), (240, 240, 240)
JEANS, KHAKI = (120, 70, 40), (120, 170, 190)
ALICE, BOB = [1, 0, 0, 0], [0, 1, 0, 0]


def scene(*people, bg=(128, 128, 128), w=W, h=H):
    """people: (left, top, right, bottom, shirt, trousers). Returns image, boxes."""
    img = np.full((h, w, 3), bg, np.uint8)
    rng = np.random.default_rng(0)
    img = np.clip(img.astype(int) + rng.integers(-8, 8, img.shape), 0, 255).astype(np.uint8)
    boxes = []
    for l, t, r, b, shirt, trousers in people:
        ph = b - t
        cv2.rectangle(img, (l + (r - l) // 3, t), (r - (r - l) // 3, t + ph // 7), (150, 180, 220), -1)
        cv2.rectangle(img, (l, t + ph // 7), (r, t + ph // 2), shirt, -1)
        cv2.rectangle(img, (l + (r - l) // 8, t + ph // 2), (r - (r - l) // 8, b), trousers, -1)
        boxes.append((l, t, r, b))
    return img, boxes


def face_of(body):
    l, t, r, b = body
    ph, cx = b - t, (l + r) // 2
    return (cx - ph // 20, t + ph // 60, cx + ph // 20, t + ph // 7)


class FaceTable(Embedder):
    """Face model stand-in: reports whatever faces the test says are visible."""
    default_threshold = 0.4

    def __init__(self):
        self.current = []

    def faces(self, image):
        return list(self.current)


def make_finder(**cfg):
    faces = FaceTable()
    matcher = FaceMatcher(faces)
    detector = RoiListDetector()
    clock = [0.0]
    finder = TargetFinder(matcher, detector, ColorHistogramEmbedder(),
                          BodyReIDConfig(**cfg), clock=lambda: clock[0])
    return finder, faces, detector, clock


class RoiListDetector(PersonDetector):
    """Knows the true boxes of the current frame; when given a crop, it reports
    the boxes lying inside that crop in crop coordinates (as a real detector
    would), which the caller maps back to the full frame."""

    def __init__(self):
        self.boxes, self.frame, self.calls, self.full_calls = [], None, 0, 0

    def detect(self, image):
        self.calls += 1
        if image is self.frame:
            self.full_calls += 1
            return [(b, 0.9) for b in self.boxes]
        # a crop of self.frame: find its offset
        off = image.__array_interface__["data"][0] - self.frame.__array_interface__["data"][0]
        row_bytes = self.frame.strides[0]
        oy, ox = off // row_bytes, (off % row_bytes) // self.frame.strides[1]
        ch, cw = image.shape[:2]
        return [((l - ox, t - oy, r - ox, b - oy), 0.9) for l, t, r, b in self.boxes
                if l >= ox and t >= oy and r <= ox + cw and b <= oy + ch]


def show(finder, faces, detector, img, boxes, face_boxes=()):
    """Feed one frame: `face_boxes` are faces the face model can see (Alice's)."""
    detector.boxes, detector.frame = boxes, img
    faces.current = [FaceObservation(f, unit(np.array(ALICE, np.float32))) for f in face_boxes]
    return finder.find(img)


def enrol(finder, faces, detector, clock, person=(200, 100, 280, 400, RED, JEANS), frames=12):
    finder.face._target = unit(np.array(ALICE, np.float32))
    for _ in range(frames):
        img, boxes = scene(person)
        show(finder, faces, detector, img, boxes, [face_of(boxes[0])])
        clock[0] += 0.1


# ---- embedders -------------------------------------------------------------------

class ColorEmbedderTests(unittest.TestCase):
    def setUp(self):
        self.e = ColorHistogramEmbedder()

    def sim(self, a, b):
        ia, ba = scene(a)
        ib, bb = scene(b, bg=(90, 110, 100))
        return float(self.e.embed(ia, ba)[0] @ self.e.embed(ib, bb)[0])

    def test_same_outfit_matches_across_position_size_and_background(self):
        self.assertGreater(self.sim((100, 50, 180, 350, RED, JEANS), (400, 200, 440, 350, RED, JEANS)), 0.9)

    def test_different_shirt_does_not_match(self):
        s = self.sim((100, 50, 180, 350, RED, JEANS), (100, 50, 180, 350, GREEN, JEANS))
        self.assertLess(s, ColorHistogramEmbedder.default_acquire)

    def test_black_and_white_are_told_apart(self):
        s = self.sim((100, 50, 180, 350, BLACK, JEANS), (100, 50, 180, 350, WHITE, JEANS))
        self.assertLess(s, ColorHistogramEmbedder.default_acquire)

    def test_rows_are_unit_length_and_tiny_boxes_do_not_crash(self):
        img, _ = scene()
        v = self.e.embed(img, [(10, 10, 60, 200), (5, 5, 7, 7), (630, 470, 700, 600)])
        self.assertAlmostEqual(float(np.linalg.norm(v[0])), 1.0, places=5)
        self.assertEqual(v.shape[0], 3)


class FakeNet:
    """Stands in for cv2.dnn: records the blob, returns a canned output."""

    def __init__(self, output):
        self.output, self.blob = output, None

    def setInput(self, blob):
        self.blob = blob

    def forward(self):
        return self.output(self.blob) if callable(self.output) else self.output


class OnnxAndFusedTests(unittest.TestCase):
    def test_onnx_embedder_batches_all_crops_in_imagenet_layout(self):
        net = FakeNet(lambda blob: blob.reshape(len(blob), 3, -1).mean(axis=2))
        e = OnnxReidEmbedder("unused", net=net)
        img, boxes = scene((100, 50, 180, 350, RED, JEANS), (300, 50, 380, 350, GREEN, JEANS))
        v = e.embed(img, boxes)
        self.assertEqual(net.blob.shape, (2, 3, 256, 128))
        self.assertEqual(v.shape, (2, 3))
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=1e-5)

    def test_fused_similarity_is_the_weighted_mean_of_the_parts(self):
        a = ColorHistogramEmbedder()
        b = OnnxReidEmbedder("unused", net=FakeNet(lambda blob: blob.reshape(len(blob), 3, -1).mean(axis=2)))
        f = FusedEmbedder([(a, 3.0), (b, 1.0)])
        img, boxes = scene((100, 50, 180, 350, RED, JEANS), (300, 50, 380, 350, GREEN, BLACK))
        va, vb, vf = a.embed(img, boxes), b.embed(img, boxes), f.embed(img, boxes)
        self.assertAlmostEqual(float(vf[0] @ vf[1]), 0.75 * float(va[0] @ va[1]) + 0.25 * float(vb[0] @ vb[1]),
                               places=5)
        self.assertAlmostEqual(f.default_acquire, 0.75 * a.default_acquire + 0.25 * b.default_acquire)


# ---- gallery / geometry / motion ------------------------------------------------------

def v(*xs):
    return unit(np.array(xs, np.float32))


class GalleryTests(unittest.TestCase):
    def test_empty_gallery_scores_minus_one(self):
        self.assertEqual(float(AppearanceGallery().score(np.stack([v(1, 0)]))[0]), -1.0)

    def test_similar_looks_merge_distinct_ones_are_added(self):
        g = AppearanceGallery(capacity=3, merge_above=0.9)
        g.add(v(1, 0, 0), 0)
        g.add(v(1, 0.05, 0), 1)       # same look, refined
        g.add(v(0, 1, 0), 2)          # a new look
        self.assertEqual(len(g), 2)
        self.assertGreater(float(g.score(np.stack([v(0, 1, 0)]))[0]), 0.99)

    def test_full_gallery_evicts_the_stalest_look(self):
        g = AppearanceGallery(capacity=2, merge_above=0.99)
        g.add(v(1, 0, 0), 0)
        g.add(v(0, 1, 0), 5)
        g.add(v(0, 0, 1), 6)          # evicts (1,0,0), refreshed longest ago
        self.assertLess(float(g.score(np.stack([v(1, 0, 0)]))[0]), 0.5)

    def test_looks_expire_with_a_ttl(self):
        g = AppearanceGallery(ttl_s=10)
        g.add(v(1, 0), 0)
        g.add(v(0, 1), 8)
        g.expire(12)
        self.assertEqual(len(g), 1)
        g.expire(30)
        self.assertEqual(len(g), 0)


class FaceBodyModelTests(unittest.TestCase):
    def test_learns_this_detectors_habits(self):
        m = FaceBodyModel()
        body, face = (200, 100, 300, 400), (235, 110, 265, 150)
        for _ in range(60):
            m.learn(face, body, W, H)
        np.testing.assert_allclose(m.face_box(body, W, H), face, atol=2)

    def test_cut_off_at_the_bottom_uses_the_width_for_size(self):
        m = FaceBodyModel()
        whole = m.face_box((200, 100, 300, 370), W, H)
        cut = m.face_box((200, 300, 300, H - 1), W, H)            # legs run off the bottom
        self.assertAlmostEqual(whole[3] - whole[1], 0.11 * 270, delta=2)   # full body: from its height
        self.assertAlmostEqual(cut[3] - cut[1], 0.37 * 100, delta=2)       # cut off: from its width

    def test_cut_off_at_the_top_places_the_face_above_the_frame(self):
        m = FaceBodyModel()
        f = m.face_box((200, 0, 300, 300), W, H)                   # head out of frame
        self.assertLess((f[1] + f[3]) / 2, 0)


class MotionGateTests(unittest.TestCase):
    def test_constant_velocity_prediction(self):
        g = MotionGate(smoothing=1.0)
        g.update(0.2, 0.5, 0.05, 0.0)
        g.update(0.3, 0.5, 0.05, 1.0)
        x, y, s = g.predict(2.0)
        self.assertAlmostEqual(x, 0.4)

    def test_distance_is_in_body_heights_and_prediction_goes_stale(self):
        g = MotionGate(rh=0.1, max_age_s=2.0)
        g.update(0.5, 0.5, 0.05, 0.0)                  # body height 0.5 frame heights
        self.assertAlmostEqual(g.distance(0.75, 0.5, 0.0), 0.5)
        self.assertIsNone(g.distance(0.75, 0.5, 3.0))


# ---- detectors ---------------------------------------------------------------------

class YoloDecodeTests(unittest.TestCase):
    def test_v8_layout(self):
        raw = np.zeros((1, 84, 100), np.float32)   # real exports have N >> 84
        raw[0, :4, 0] = [100, 120, 40, 80]
        raw[0, 4, 0] = 0.9            # person
        raw[0, 5, 1] = 0.9            # a bicycle: ignored
        raw[0, 4, 2] = 0.1            # too weak
        boxes, scores = decode_yolo(raw, "auto", 320)
        np.testing.assert_allclose(boxes, [[100, 120, 40, 80]])
        np.testing.assert_allclose(scores, [0.9])

    def test_v5_layout_multiplies_objectness(self):
        raw = np.zeros((1, 100, 85), np.float32)
        raw[0, 0, :6] = [50, 60, 10, 20, 0.8, 0.9]
        raw[0, 1, :6] = [50, 60, 10, 20, 0.3, 0.9]
        boxes, scores = decode_yolo(raw, "auto", 320, conf=0.4)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(float(scores[0]), 0.72, places=5)

    def test_yolox_grid_decoding(self):
        size = 64
        rows = (size // 8) ** 2 + (size // 16) ** 2 + (size // 32) ** 2
        raw = np.zeros((1, rows, 85), np.float32)
        raw[0, 9, :6] = [0.5, 0.5, np.log(2), np.log(4), 0.9, 1.0]   # stride-8 cell (x=1, y=1)
        boxes, _ = decode_yolo(raw, "yolox", size)
        np.testing.assert_allclose(boxes[0], [12, 12, 16, 32])

    def test_detector_maps_letterboxed_boxes_back_to_the_frame(self):
        size = 320
        # a 640x480 frame letterboxes to scale 0.5 with 40 px of padding top and bottom
        _, scale, px, py = letterbox(np.zeros((480, 640, 3), np.uint8), size)
        self.assertEqual((scale, px, py), (0.5, 0, 40))
        raw = np.zeros((1, 84, 200), np.float32)
        raw[0, :5, 0] = [160, 40 + 120, 50, 100, 0.95]           # centre (320, 240) in frame pixels
        det = YoloPersonDetector("unused", input_size=size, net=FakeNet(raw))
        (box, score), = det.detect(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(box, (270, 140, 370, 340))


class HOGAndRoiTests(unittest.TestCase):
    def test_hog_runs_and_finds_nobody_in_an_empty_room(self):
        self.assertEqual(HOGPersonDetector().detect(np.full((480, 640, 3), 128, np.uint8)), [])
        self.assertEqual(HOGPersonDetector().detect(np.zeros((50, 50, 3), np.uint8)), [])

    def test_roi_boxes_come_back_in_frame_coordinates(self):
        class One(PersonDetector):
            def detect(self, image):
                return [((10, 20, 30, 60), 0.8)]
        self.assertEqual(detect_in_roi(One(), np.zeros((480, 640, 3), np.uint8), (100, 50, 300, 400)),
                         [((110, 70, 130, 110), 0.8)])
        self.assertEqual(detect_in_roi(One(), np.zeros((480, 640, 3), np.uint8), (-50, -50, 5, 5)), [])


# ---- TargetFinder --------------------------------------------------------------------

class TargetFinderTests(unittest.TestCase):
    def test_learns_the_body_while_the_face_is_visible(self):
        finder, faces, det, clock = make_finder()
        enrol(finder, faces, det, clock)
        self.assertEqual(len(finder.long_term), 1)
        self.assertEqual(finder.stats["source"], "face")

    def test_picks_them_up_from_behind_once_learned(self):
        finder, faces, det, clock = make_finder()
        enrol(finder, faces, det, clock)
        img, boxes = scene((210, 100, 290, 400, RED, JEANS))
        d = show(finder, faces, det, img, boxes)            # no face visible
        self.assertIsNotNone(d)
        self.assertEqual(d.source, "track")
        # the estimated face is about where the real one would be
        f = face_of(boxes[0])
        self.assertAlmostEqual(d.offset_x, ((f[0] + f[2]) / 2 - W / 2) / (W / 2), delta=0.05)
        self.assertAlmostEqual(d.offset_y, -((f[1] + f[3]) / 2 - H / 2) / (H / 2), delta=0.05)

    def test_someone_else_is_not_picked_up(self):
        finder, faces, det, clock = make_finder()
        enrol(finder, faces, det, clock)
        img, boxes = scene((210, 100, 290, 400, GREEN, KHAKI))
        self.assertIsNone(show(finder, faces, det, img, boxes))

    def test_two_lookalikes_with_no_recent_track_means_no_guess(self):
        finder, faces, det, clock = make_finder()
        enrol(finder, faces, det, clock)
        clock[0] += 10                                       # the motion prediction is long stale
        img, boxes = scene((60, 100, 140, 400, RED, JEANS), (450, 100, 530, 400, RED, JEANS))
        self.assertIsNone(show(finder, faces, det, img, boxes))

    def test_with_a_fresh_track_the_lookalike_across_the_frame_is_gated_out(self):
        finder, faces, det, clock = make_finder()
        enrol(finder, faces, det, clock)
        img, boxes = scene((205, 100, 285, 400, RED, JEANS), (520, 100, 600, 400, RED, JEANS))
        d = show(finder, faces, det, img, boxes)
        self.assertIsNotNone(d)
        self.assertLess(d.offset_x, 0)                       # the one on the left, where she was

    def test_body_only_matches_never_teach_the_long_term_gallery(self):
        finder, faces, det, clock = make_finder()
        enrol(finder, faces, det, clock)
        before = len(finder.long_term)
        for i in range(30):
            clock[0] += 0.1
            img, boxes = scene((200 + i, 100, 280 + i, 400, RED, JEANS))
            self.assertIsNotNone(show(finder, faces, det, img, boxes))
        self.assertEqual(len(finder.long_term), before)
        self.assertGreaterEqual(len(finder.short_term), 1)   # but it does remember short-term...
        clock[0] += 60
        finder.short_term.expire(clock[0])
        self.assertEqual(len(finder.short_term), 0)           # ...and forgets

    def test_a_full_length_photo_seeds_the_body_gallery(self):
        finder, faces, det, clock = make_finder()
        photo, boxes = scene((250, 40, 390, 470, BLUE, KHAKI))
        det.boxes, det.frame = boxes, photo
        faces.current = [FaceObservation(face_of(boxes[0]), unit(np.array(ALICE, np.float32)))]
        self.assertEqual(finder.set_target(photo), 1)
        self.assertEqual(len(finder.long_term), 1)
        clock[0] += 30                                       # later, seen from behind
        img, boxes = scene((100, 150, 160, 380, BLUE, KHAKI))
        self.assertEqual(show(finder, faces, det, img, boxes).source, "track")

    def test_a_face_photo_alone_still_enrols(self):
        finder, faces, det, clock = make_finder()
        photo = np.full((300, 300, 3), 128, np.uint8)
        det.boxes, det.frame = [], photo
        faces.current = [FaceObservation((100, 100, 200, 200), unit(np.array(ALICE, np.float32)))]
        self.assertEqual(finder.set_target(photo), 1)
        self.assertTrue(finder.has_target)
        self.assertEqual(len(finder.long_term), 0)

    def test_clearing_the_target_forgets_the_body_too(self):
        finder, faces, det, clock = make_finder()
        enrol(finder, faces, det, clock)
        finder.clear_target()
        self.assertEqual(len(finder.long_term), 0)
        img, boxes = scene((210, 100, 290, 400, RED, JEANS))
        self.assertIsNone(show(finder, faces, det, img, boxes))

    def test_no_detector_work_at_all_before_anyone_is_enrolled(self):
        finder, faces, det, clock = make_finder()
        img, boxes = scene((210, 100, 290, 400, RED, JEANS))
        self.assertIsNone(show(finder, faces, det, img, boxes))
        self.assertEqual(det.calls, 0)

    def test_while_the_face_is_visible_the_body_is_only_studied_every_nth_frame(self):
        finder, faces, det, clock = make_finder(enroll_every_n=5)
        enrol(finder, faces, det, clock, frames=20)
        self.assertEqual(det.calls, 4)                       # frames 1, 6, 11, 16

    def test_a_face_that_could_belong_to_two_bodies_teaches_nothing(self):
        finder, faces, det, clock = make_finder()
        finder.face._target = unit(np.array(ALICE, np.float32))
        img, _ = scene((200, 100, 280, 400, RED, JEANS))
        overlapping = [(200, 100, 280, 400), (190, 90, 300, 420)]
        show(finder, faces, det, img, overlapping, [face_of(overlapping[0])])
        self.assertEqual(len(finder.long_term), 0)

    def test_the_crop_search_finds_them_without_a_full_frame_pass(self):
        finder, faces, det, clock = make_finder()
        enrol(finder, faces, det, clock)
        full_before = det.full_calls
        img, boxes = scene((215, 100, 295, 400, RED, JEANS))
        self.assertIsNotNone(show(finder, faces, det, img, boxes))
        self.assertEqual(det.full_calls, full_before)        # found in the crop around the prediction

    def test_if_the_crop_misses_the_whole_frame_is_searched(self):
        finder, faces, det, clock = make_finder()
        enrol(finder, faces, det, clock)
        clock[0] += 3.0                                      # a while since last seen: gate is wider
        img, boxes = scene((480, 100, 560, 400, RED, JEANS))
        self.assertIsNotNone(show(finder, faces, det, img, boxes))
        self.assertGreater(det.full_calls, 0)


if __name__ == "__main__":
    unittest.main()
