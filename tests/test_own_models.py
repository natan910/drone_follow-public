"""Our own detector and re-ID, the parts that need no PyTorch: training targets
<-> flight decoder round trip, the CenterPersonDetector wrapper, augmentation,
metrics, the benchmark and its gate. PyTorch parts: tests/test_training.py."""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from config import BodyReIDConfig
from dataset.export import export
from dataset.synthetic import make_dataset
from evaluation.benchmark import bench_detector, bench_reid, headline
from evaluation.metrics import average_precision, recall_by_size, reid_scores, suggest_thresholds
from perception.body_reid import ColorHistogramEmbedder
from perception.person_detector import CenterPersonDetector, PersonDetector, decode_center, letterbox
from perception.target_finder import make_person_detector
from training.augment import (REID_H, REID_W, crop_person, detector_sample, hflip, letterbox_boxes,
                              motion_blur, random_affine, random_erase, reid_sample)
from training.targets import as_network_output, draw_gaussian, encode_center, gaussian_radius


class TestTargetsRoundTrip(unittest.TestCase):
    def test_decoder_recovers_the_boxes_the_targets_were_built_from(self):
        boxes = [[40, 30, 100, 200], [150, 60, 190, 150], [250, 250, 270, 300], [10, 10, 26, 58]]
        heat, reg, logwh, ind, mask = encode_center(boxes, 320)
        self.assertEqual(heat.shape, (1, 80, 80))
        self.assertEqual(int(mask.sum()), 4)
        got, scores = decode_center(as_network_output(heat, reg, logwh, ind, mask), conf=0.5)
        self.assertEqual(len(got), 4)
        self.assertTrue(np.allclose(scores, 1.0))
        rebuilt = sorted([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] for cx, cy, w, h in got])
        self.assertTrue(np.allclose(rebuilt, sorted(boxes), atol=1e-3), rebuilt)

    def test_heat_peaks_are_exactly_one_and_bumps_smaller(self):
        heat, *_ = encode_center([[100, 100, 160, 260]], 320)
        self.assertEqual(float(heat.max()), 1.0)
        self.assertEqual(int((heat == 1.0).sum()), 1)
        self.assertGreater(int((heat > 0.1).sum()), 5)

    def test_boxes_outside_or_empty_are_skipped_and_capped(self):
        _, _, _, _, mask = encode_center([[400, 400, 500, 500], [5, 5, 5, 40]], 320)
        self.assertEqual(mask.sum(), 0)
        many = [[i % 300, 0, i % 300 + 10, 30] for i in range(0, 3000, 7)]
        _, _, _, _, mask = encode_center(many, 320, max_objects=64)
        self.assertEqual(mask.sum(), 64)

    def test_gaussian_helpers(self):
        self.assertGreater(gaussian_radius(40, 20), 0)
        h = np.zeros((10, 10), np.float32)
        draw_gaussian(h, 0, 9, 3)                               # clipped at the corner: no crash
        self.assertEqual(h[9, 0], 1.0)


class FakeNet:
    """Stands in for cv2.dnn: returns a prepared output map."""

    def __init__(self, out):
        self.out, self.blob = out, None

    def setInput(self, blob):
        self.blob = blob

    def forward(self):
        return self.out


class TestCenterPersonDetector(unittest.TestCase):
    def test_boxes_come_back_in_original_image_pixels(self):
        image = np.zeros((480, 640, 3), np.uint8)
        _, scale, px, py = letterbox(image, 320)
        truth = [(100, 50, 180, 300), (400, 200, 440, 330)]
        in_net = [[l * scale + px, t * scale + py, r * scale + px, b * scale + py] for l, t, r, b in truth]
        out = as_network_output(*encode_center(in_net, 320))
        det = CenterPersonDetector("unused", 320, conf=0.5, net=FakeNet(out))
        found = sorted(b for b, s in det.detect(image))
        self.assertEqual(len(found), 2)
        for f, t in zip(found, sorted(truth)):
            self.assertTrue(np.allclose(f, t, atol=2), (f, t))
        self.assertEqual(det.net.blob.shape, (1, 3, 320, 320))
        self.assertGreater(float(det.net.blob.max()), 1.0)   # RGB 0..255, not 0..1

    def test_nothing_above_threshold(self):
        det = CenterPersonDetector("unused", 320, net=FakeNet(np.zeros((1, 5, 80, 80), np.float32)))
        self.assertEqual(det.detect(np.zeros((240, 320, 3), np.uint8)), [])

    def test_wrong_output_shape_is_a_clear_error(self):
        with self.assertRaises(ValueError):
            decode_center(np.zeros((1, 84, 8400), np.float32))

    def test_config_builds_it(self):
        with self.assertRaises(FileNotFoundError):           # proves "own" is wired to the ONNX path
            make_person_detector(BodyReIDConfig(detector="own", own_model="/nope/person_own.onnx"))


class TestAugment(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.img = np.random.default_rng(1).integers(0, 255, (240, 320, 3), dtype=np.uint8)
        self.boxes = [[50, 40, 90, 200], [200, 100, 230, 180]]

    def test_eval_sample_matches_flight_letterbox(self):
        rgb, bx = detector_sample(self.img, self.boxes, 320, None)
        flight, scale, px, py = letterbox(self.img, 320)
        self.assertTrue(np.array_equal(rgb, flight[..., ::-1]))
        self.assertAlmostEqual(bx[0][0], 50 * scale + px)

    def test_affine_keeps_boxes_on_their_pixels(self):
        img = np.zeros((240, 320, 3), np.uint8)
        img[40:200, 50:90] = 255                                  # a white "person" exactly at box 0
        for _ in range(20):
            out, bx = random_affine(img, [self.boxes[0]], 160, self.rng)
            for l, t, r, b in bx:
                inside = out[int(t) + 2:int(b) - 2, int(l) + 2:int(r) - 2]
                if inside.size:
                    self.assertGreater(float(inside.mean()), 200)

    def test_flip_and_training_sample(self):
        _, bx = hflip(self.img, [[10, 0, 30, 5]])
        self.assertEqual(bx, [[290, 0, 310, 5]])
        rgb, bx = detector_sample(self.img, self.boxes, 160, self.rng)
        self.assertEqual(rgb.shape, (160, 160, 3))
        for l, t, r, b in bx:
            self.assertTrue(0 <= l < r <= 160 and 0 <= t < b <= 160)
        self.assertEqual(motion_blur(self.img, self.rng).shape, self.img.shape)

    def test_letterbox_boxes(self):
        _, bx = letterbox_boxes(self.img, [[0, 0, 320, 240]], 320)
        self.assertEqual(bx, [[0.0, 40.0, 320.0, 280.0]])

    def test_reid_sample_matches_the_flight_embedder_input(self):
        crop = crop_person(self.img, self.boxes[0])
        x = reid_sample(crop, None)
        self.assertEqual(x.shape, (3, REID_H, REID_W))
        # same preprocessing as perception.body_reid.OnnxReidEmbedder
        import cv2
        from perception.body_reid import OnnxReidEmbedder
        rgb = cv2.cvtColor(cv2.resize(crop, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32)
        want = ((rgb / 255.0 - OnnxReidEmbedder.MEAN) / OnnxReidEmbedder.STD).transpose(2, 0, 1)
        self.assertTrue(np.allclose(x, want, atol=1e-5))
        self.assertEqual(reid_sample(crop, self.rng).shape, (3, REID_H, REID_W))
        self.assertEqual(random_erase(np.zeros((256, 128, 3), np.uint8), self.rng).shape, (256, 128, 3))

    def test_crop_is_capped_for_the_cache(self):
        big = np.zeros((2000, 1000, 3), np.uint8)
        self.assertEqual(crop_person(big, [0, 0, 500, 1900]).shape[0], 320)


class TestMetrics(unittest.TestCase):
    def test_perfect_and_empty_detector(self):
        gt = [[0, 0, 10, 20], [30, 30, 50, 80]]
        perfect = [([(tuple(b), 0.9) for b in gt], gt)]
        self.assertEqual(average_precision(perfect)["ap"], 1.0)
        self.assertEqual(average_precision([([], gt)])["ap"], 0.0)

    def test_false_positive_ranked_first_lowers_ap(self):
        gt = [[0, 0, 10, 20]]
        frames = [([((100, 100, 110, 120), 0.99), ((0, 0, 10, 20), 0.5)], gt)]
        r = average_precision(frames)
        self.assertAlmostEqual(r["ap"], 0.5)
        self.assertEqual(r["recall"], 1.0)

    def test_duplicate_detection_counts_as_false_positive(self):
        gt = [[0, 0, 10, 20]]
        r = average_precision([([((0, 0, 10, 20), 0.9), ((0, 0, 10, 21), 0.8)], gt)])
        self.assertEqual(r["precision"], 0.5)

    def test_recall_by_size(self):
        gt = [[0, 0, 10, 20], [0, 0, 60, 200]]
        r = recall_by_size([([((0, 0, 60, 200), 0.9)], gt)])
        self.assertEqual(r["small"]["recall"], 0.0)
        self.assertEqual(r["large"]["recall"], 1.0)

    def test_reid_scores_ignore_own_track(self):
        emb = np.array([[1, 0], [1, 0.01], [0.9, 0.1], [0, 1]], np.float32)
        ids = ["a", "a", "a", "b"]
        groups = ["a1", "a1", "a2", "b1"]
        r = reid_scores(emb, ids, groups)
        self.assertEqual(r["rank1"], 1.0)
        self.assertEqual(r["queries"], 3)                      # b has no positive elsewhere
        bad = reid_scores(emb, ["a", "b", "a", "b"], ["x", "y", "z", "w"])
        self.assertLess(bad["map"], 1.0)

    def test_thresholds_sit_between_strangers_and_matches(self):
        rng = np.random.default_rng(0)
        a = rng.normal([5, 0, 0], 0.3, (20, 3))
        b = rng.normal([0, 5, 0], 0.3, (20, 3))
        emb = np.vstack([a, b])
        ids = ["a"] * 20 + ["b"] * 20
        groups = [f"g{i}" for i in range(40)]
        t = suggest_thresholds(emb, ids, groups)
        self.assertGreater(t["pos_median"], t["neg_median"])
        self.assertLessEqual(t["keep"], t["acquire"])


class TruthDetector(PersonDetector):
    """Reports the ground truth for any frame of the synthetic set (by lookup)."""

    def __init__(self, table, drop_every=0):
        self.table, self.drop, self.n = table, drop_every, 0

    def detect(self, image):
        self.n += 1
        boxes = self.table.get(image.tobytes()[:4096], [])
        if self.drop and self.n % self.drop == 0:
            boxes = boxes[1:]
        return [(tuple(b), 0.9) for b in boxes]


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="df_bench_")
        cls.root, cls.out = os.path.join(cls.dir, "d"), os.path.join(cls.dir, "e")
        make_dataset(cls.root, n_frames=8)
        export(cls.root, cls.out)
        with open(os.path.join(cls.out, "detector_eval.json")) as f:
            cls.items = json.load(f)
        import cv2
        cls.table = {cv2.imread(os.path.join(cls.root, it["image"])).tobytes()[:4096]: it["boxes"]
                     for it in cls.items}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_detector_benchmark(self):
        r = bench_detector(TruthDetector(self.table), self.items, self.root)
        self.assertEqual(r["ap50"], 1.0)
        self.assertIn("alt<3m", r["ap50_by_altitude"])
        worse = bench_detector(TruthDetector(self.table, drop_every=2), self.items, self.root)
        self.assertLess(headline("detector", worse), headline("detector", r))

    def test_reid_benchmark_with_colour_histograms(self):
        with open(os.path.join(self.out, "reid_eval.json")) as f:
            items = json.load(f)
        r = bench_reid(ColorHistogramEmbedder(), items, self.root)
        self.assertGreater(r["crops"], 10)
        self.assertGreater(r["cross_session"]["queries"], 0)
        self.assertGreater(r["cross_session"]["rank1"], 0.5)   # painted clothes: colour should win
        self.assertTrue(np.isfinite(headline("reid", r)))

    def test_cli_gate(self):
        from tools import benchmark as bench_cli
        res = os.path.join(self.dir, "res")
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            code = bench_cli.main(["reid", "--data", self.out, "--model", "color", "--model", "color",
                                   "--gate", "--out-dir", res])
        self.assertEqual(code, 0, buf.getvalue())                # equal is not worse
        self.assertTrue(os.listdir(res))
        with contextlib.redirect_stdout(io.StringIO()):
            code = bench_cli.main(["reid", "--data", self.out, "--model", "color", "--model", "color",
                                   "--gate", "--min-gain", "0.5", "--out-dir", res])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
