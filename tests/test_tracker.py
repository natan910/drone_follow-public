import unittest

from config import TrackerConfig
from datatypes import Detection, TrackState
from tracking.target_tracker import TargetTracker


def det(offset_x=0.3, offset_y=0.1, size=0.1, source="face"):
    return Detection(bbox=(0, 10, 10, 0), offset_x=offset_x, offset_y=offset_y,
                     size=size, source=source)


DET = det()


def make(**kw):
    base = dict(lock_frames=3, grace_period=0.5, give_up_timeout=2.0, smoothing=0.5,
                track_only_timeout_s=45.0)
    base.update(kw)
    return TargetTracker(TrackerConfig(**base))


class TrackerTests(unittest.TestCase):
    def test_needs_consecutive_hits_to_lock(self):
        t = make()
        self.assertEqual(t.update(DET, 0.0).state, TrackState.SEARCHING)
        self.assertEqual(t.update(DET, 0.1).state, TrackState.SEARCHING)
        self.assertEqual(t.update(DET, 0.2).state, TrackState.TRACKING)

    def test_gap_resets_lock_progress(self):
        t = make()
        t.update(DET, 0.0)
        t.update(DET, 0.1)
        t.update(None, 0.2)
        self.assertEqual(t.update(DET, 0.3).state, TrackState.SEARCHING)

    def test_short_dropout_stays_tracking_and_holds_values(self):
        t = make()
        for i in range(3):
            t.update(DET, i * 0.1)
        out = t.update(None, 0.5)
        self.assertEqual(out.state, TrackState.TRACKING)
        self.assertAlmostEqual(out.target.offset_x, 0.3)
        self.assertAlmostEqual(out.target.offset_y, 0.1)

    def test_long_dropout_goes_lost_keeping_last_estimate(self):
        t = make()
        for i in range(3):
            t.update(DET, i * 0.1)
        out = t.update(None, 1.0)
        self.assertEqual(out.state, TrackState.LOST)
        self.assertAlmostEqual(out.target.offset_x, 0.3)

    def test_reacquire_from_lost_discards_stale_smoothing(self):
        t = make()
        for i in range(3):
            t.update(DET, i * 0.1)
        t.update(None, 1.0)
        out = t.update(det(offset_x=-0.5, offset_y=0.0, size=0.2), 1.1)
        self.assertEqual(out.state, TrackState.TRACKING)
        self.assertAlmostEqual(out.target.offset_x, -0.5)

    def test_gives_up_after_timeout(self):
        t = make()
        for i in range(3):
            t.update(DET, i * 0.1)
        t.update(None, 1.0)
        out = t.update(None, 3.5)
        self.assertEqual(out.state, TrackState.SEARCHING)
        self.assertIsNone(out.target)

    def test_smoothing_blends_values(self):
        t = make(lock_frames=1)
        t.update(det(offset_x=0.0, offset_y=0.0, size=0.1), 0.0)
        out = t.update(det(offset_x=1.0, offset_y=0.2, size=0.1), 0.1)
        self.assertAlmostEqual(out.target.offset_x, 0.5)
        self.assertAlmostEqual(out.target.offset_y, 0.1)

    def test_identity_age_grows_since_the_last_confirmed_face(self):
        t = make(lock_frames=1)
        t.update(DET, 0.0)
        out = t.update(DET, 1.0)
        self.assertAlmostEqual(out.identity_age_s, 0.0)  # a "face" detection just confirmed it
        out = t.update(None, 1.5)
        self.assertAlmostEqual(out.identity_age_s, 0.5)  # nothing has confirmed it since

    def test_track_only_detection_cannot_start_a_lock(self):
        t = make(lock_frames=1)
        self.assertEqual(t.update(det(source="track"), 0.0).state, TrackState.SEARCHING)
        self.assertIsNone(t.update(det(source="track"), 0.1).target)

    def test_track_only_detection_can_extend_an_existing_lock(self):
        t = make(lock_frames=1)
        t.update(DET, 0.0)                                    # face: locks on
        out = t.update(det(offset_x=0.6, source="track"), 0.1)  # track-only: still trusted
        self.assertEqual(out.state, TrackState.TRACKING)
        self.assertAlmostEqual(out.target.offset_x, 0.45)  # blended toward the new value

    def test_track_only_detection_can_relock_after_the_track_was_dropped(self):
        t = make(lock_frames=2, give_up_timeout=1.0)
        t.update(DET, 0.0)
        t.update(DET, 0.1)                                   # locked by face
        t.update(None, 1.0)                                  # LOST
        self.assertEqual(t.update(None, 3.0).state, TrackState.SEARCHING)   # dropped
        t.update(det(source="track"), 3.1)
        self.assertEqual(t.update(det(source="track"), 3.2).state, TrackState.TRACKING)

    def test_forget_means_the_body_alone_can_never_lock_again(self):
        t = make(lock_frames=1)
        t.update(DET, 0.0)
        t.forget()
        self.assertEqual(t.update(det(source="track"), 0.1).state, TrackState.SEARCHING)

    def test_track_only_detection_stops_being_trusted_after_the_timeout(self):
        t = make(lock_frames=1, track_only_timeout_s=10.0, grace_period=5.0)
        t.update(DET, 0.0)                                     # last confirmed face at t=0
        out = t.update(det(offset_x=0.9, source="track"), 20.0)  # long after the timeout
        self.assertEqual(out.state, TrackState.LOST)           # treated as a miss, not a hit
        self.assertAlmostEqual(out.target.offset_x, 0.3)       # untouched: the stale estimate


if __name__ == "__main__":
    unittest.main()
