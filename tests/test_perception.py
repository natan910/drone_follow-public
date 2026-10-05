import math
import unittest

import numpy as np

from perception.face_matcher import Embedder, FaceMatcher, FaceObservation, make_detection, unit
from perception.range_sensors import TFLunaParser


class FakeEmbedder(Embedder):
    """Reads 'faces' straight from a dict keyed by the image's first pixel value."""
    default_threshold = 0.4

    def __init__(self, table):
        self.table = table

    def faces(self, image):
        return self.table.get(int(image[0, 0, 0]), [])


def face(box, vec):
    return FaceObservation(box, unit(np.array(vec, dtype=np.float32)))


def img(tag):
    im = np.zeros((480, 640, 3), np.uint8)
    im[0, 0, 0] = tag
    return im


ALICE, BOB = [1, 0, 0, 0], [0, 1, 0, 0]


class MatcherTests(unittest.TestCase):
    def setUp(self):
        self.m = FaceMatcher(FakeEmbedder({
            1: [face((0, 0, 40, 40), BOB), face((100, 100, 300, 300), ALICE)],  # enrolment: Alice is the bigger face
            2: [face((400, 100, 480, 200), BOB), face((280, 150, 360, 250), [0.9, 0.1, 0, 0])],
            3: [face((10, 10, 50, 50), BOB)],
            4: [],
        }))

    def test_no_target_means_no_detection(self):
        self.assertFalse(self.m.has_target)
        self.assertIsNone(self.m.find(img(2)))

    def test_enrols_the_biggest_face_and_reports_how_many_there_were(self):
        self.assertEqual(self.m.set_target(img(1)), 2)
        self.assertTrue(self.m.has_target)

    def test_finds_the_right_person_among_several(self):
        self.m.set_target(img(1))
        det = self.m.find(img(2))
        self.assertEqual(det.bbox, (150, 360, 250, 280))            # top, right, bottom, left
        self.assertAlmostEqual(det.offset_x, 0.0, places=2)         # face centred horizontally
        self.assertAlmostEqual(det.offset_y, 40 / 240, places=2)    # box sits above the frame's centre
        self.assertAlmostEqual(det.size, 100 / 480)

    def test_someone_else_is_not_a_match(self):
        self.m.set_target(img(1))
        self.assertIsNone(self.m.find(img(3)))
        self.assertIsNone(self.m.find(img(4)))

    def test_photo_without_a_face_is_an_error_and_keeps_the_old_target(self):
        self.m.set_target(img(1))
        with self.assertRaises(ValueError):
            self.m.set_target(img(4))
        self.assertTrue(self.m.has_target)

    def test_clear_target_forgets_everyone(self):
        self.m.set_target(img(1))
        self.m.clear_target()
        self.assertIsNone(self.m.find(img(2)))

    def test_detection_geometry(self):
        right = make_detection((500, 100, 600, 200), 640, 480)
        left = make_detection((40, 100, 140, 200), 640, 480)
        self.assertGreater(right.offset_x, 0)
        self.assertLess(left.offset_x, 0)

        top = make_detection((100, 20, 200, 120), 640, 480)     # box near the top of the frame
        bottom = make_detection((100, 300, 200, 400), 640, 480)  # box near the bottom
        self.assertGreater(top.offset_y, 0)     # up is positive
        self.assertLess(bottom.offset_y, 0)

    def test_make_detection_defaults_to_face_source_and_can_be_overridden(self):
        det = make_detection((100, 100, 200, 200), 640, 480)
        self.assertEqual(det.source, "face")
        track = make_detection((100, 100, 200, 200), 640, 480, source="track")
        self.assertEqual(track.source, "track")


def tf_frame(dist_cm, amp=500, temp=0):
    body = bytes([0x59, 0x59, dist_cm & 255, dist_cm >> 8, amp & 255, amp >> 8, temp & 255, temp >> 8])
    return body + bytes([sum(body) & 255])


class TFLunaParserTests(unittest.TestCase):
    def test_decodes_a_good_frame(self):
        self.assertEqual(TFLunaParser().feed(tf_frame(250)), (2.5, True))

    def test_needs_no_particular_alignment(self):
        p = TFLunaParser()
        self.assertIsNone(p.feed(b"\x00\x13" + tf_frame(100)[:5]))       # junk, then half a frame
        self.assertEqual(p.feed(tf_frame(100)[5:]), (1.0, True))

    def test_bad_checksum_is_skipped_and_it_resyncs(self):
        bad = bytearray(tf_frame(300))
        bad[8] ^= 0xFF
        self.assertEqual(TFLunaParser().feed(bytes(bad) + tf_frame(120)), (1.2, True))

    def test_returns_the_newest_frame_when_several_arrive(self):
        self.assertEqual(TFLunaParser().feed(tf_frame(100) + tf_frame(200)), (2.0, True))

    def test_weak_signal_or_zero_distance_means_no_usable_return(self):
        self.assertEqual(TFLunaParser().feed(tf_frame(500, amp=20)), (None, True))
        self.assertEqual(TFLunaParser().feed(tf_frame(0, amp=500)), (None, True))
        self.assertEqual(TFLunaParser().feed(tf_frame(500, amp=65535)), (None, True))


if __name__ == "__main__":
    unittest.main()
