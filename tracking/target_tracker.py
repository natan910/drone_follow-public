"""
Tracking state machine. Turns a noisy per-frame Detection (or None) into a
stable decision. Knows nothing about faces, cameras, or drones.

    SEARCHING  No confirmed target. Needs `lock_frames` consecutive FACE
               detections before we trust it (rejects one-frame false matches;
               a "track"-only detection can never start a lock on its own).
    TRACKING   Confirmed and seen recently. Brief dropouts are tolerated for
               `grace_period` seconds, holding the last smoothed values.
    LOST       Missing longer than the grace period. The last known estimate is
               kept so the controller knows which way they went. After
               `give_up_timeout` seconds we return to SEARCHING.

Detections come in two kinds (Detection.source):
    "face"   a face matched the enrolled photo: identity is confirmed this frame.
    "track"  a visual tracker (e.g. colour/shape blob under the drone) followed
             someone whose face isn't visible right now — typically while
             hovering directly overhead, or following them from behind. A
             "track" detection is only trusted within `track_only_timeout_s`
             of the last confirmed face (so the drone can't silently drift onto
             the wrong person over a long hover). Within that window it can
             extend a lock and also RE-lock after the track was dropped (they
             walked out of view and were recognised again from behind); it can
             never start the first lock on anyone.

The time of the last confirmed face survives the track being dropped
(`reset`); only `forget` (a new target) clears it.
"""

from typing import Optional

from config import TrackerConfig
from datatypes import Detection, TargetEstimate, TrackerOutput, TrackState


class TargetTracker:
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.cfg = config or TrackerConfig()
        self._last_face: Optional[float] = None
        self.reset()

    def reset(self) -> None:
        """Drop the track (not the memory of when their identity was last confirmed)."""
        self.state = TrackState.SEARCHING
        self._hits = 0
        self._last_seen: Optional[float] = None
        self._estimate: Optional[TargetEstimate] = None

    def forget(self) -> None:
        """Drop everything, including identity: for a new or cleared target."""
        self.reset()
        self._last_face = None

    def update(self, detection: Optional[Detection], now: float) -> TrackerOutput:
        """Call once per frame with the perception result (or None)."""
        if detection is not None and not self._trusted(detection, now):
            detection = None
        if detection is not None:
            self._on_detection(detection, now)
        else:
            self._on_miss(now)
        known = self.state != TrackState.SEARCHING
        return TrackerOutput(
            state=self.state,
            target=self._estimate if known else None,
            bbox=detection.bbox if detection is not None else None,
            identity_age_s=0.0 if self._last_face is None else now - self._last_face,
        )

    def _trusted(self, det: Detection, now: float) -> bool:
        if det.source == "face":
            return True
        if self._last_face is None:
            return False  # a visual-only detection can never start the first lock
        return now - self._last_face <= self.cfg.track_only_timeout_s

    def _blend(self, det: Detection) -> TargetEstimate:
        old, a = self._estimate, self.cfg.smoothing
        if old is None:
            return TargetEstimate(det.offset_x, det.offset_y, det.size)
        return TargetEstimate(a * det.offset_x + (1 - a) * old.offset_x,
                              a * det.offset_y + (1 - a) * old.offset_y,
                              a * det.size + (1 - a) * old.size)

    def _on_detection(self, det: Detection, now: float) -> None:
        self._last_seen = now
        if det.source == "face":
            self._last_face = now
        self._hits += 1

        if self.state == TrackState.LOST:
            # Re-acquired: the held estimate is stale, so restart smoothing.
            self._estimate = TargetEstimate(det.offset_x, det.offset_y, det.size)
            self.state = TrackState.TRACKING
            return

        self._estimate = self._blend(det)
        if self.state == TrackState.SEARCHING and self._hits >= self.cfg.lock_frames:
            self.state = TrackState.TRACKING

    def _on_miss(self, now: float) -> None:
        self._hits = 0
        if self.state == TrackState.SEARCHING:
            self._estimate = None  # forget unconfirmed candidates
            return

        missing_for = now - (self._last_seen if self._last_seen is not None else now)
        if self.state == TrackState.TRACKING and missing_for > self.cfg.grace_period:
            self.state = TrackState.LOST
        elif self.state == TrackState.LOST and missing_for > self.cfg.give_up_timeout:
            self.reset()
