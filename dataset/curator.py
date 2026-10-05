"""
Record the frames that teach the models something, not four ordinary frames a second.

    --record datasets --curate        (main.py, via build_recorder below)

A plain Recorder saves every 4th of a second: mostly empty sky and the same
person standing still. The curator watches what the drone knows each step and
flags the moments a model got wrong or nearly wrong:

    lost                     tracking just dropped the target
    reacquired               found them again after LOST / SEARCH / PATROL
    face_to_body             the face disappeared, body re-ID took over
    body_confirmed_by_face   body re-ID's pick was then confirmed by the face:
                             a FREE correct re-ID label (the best one we get)
    reid_borderline          body re-ID score close to its threshold (needs matcher stats)
    far_target               target tiny in the frame (the detector's weak spot)
    close_obstacle           a range sensor sees something close, no target in view
    new_place                first visit to this part of the map (variety)

Each event records a short burst (`burst_s`) at up to `event_fps`, plus a slow
background trickle (`background_fps`). Every saved frame gets a "why" list in
frames.jsonl, so later you can review "reid_borderline" frames first.

Privacy (--record-blur): FaceBlur pixelates every face in a saved frame except
the target's, on the recorder's own thread (flight never waits for it). Frames
of people who never agreed to be filmed are personal data (GDPR); for a future
commercial dataset, keep only faces of people who signed consent.

Everything downstream is unchanged: tools/data.py autolabel / review / export
read these sessions exactly like plain recordings.
"""

import threading
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from datatypes import Decision, Mode, Observation
from dataset.recorder import Recorder
from missions.mission_config import CuratorConfig

SEARCHING_MODES = {Mode.LOST, Mode.SEARCH, Mode.EXPLORE, Mode.PATROL}


class Curator:
    """Decides, from one step's Observation / Decision, why (if at all) this frame matters.
    Pure logic: no frames, no disk. `stats_fn` (optional) returns TargetFinder.stats."""

    def __init__(self, config: Optional[CuratorConfig] = None,
                 stats_fn: Optional[Callable[[], Optional[dict]]] = None):
        self.cfg = config or CuratorConfig()
        self.stats_fn = stats_fn
        self._prev_mode: Optional[Mode] = None
        self._prev_source: Optional[str] = None
        self._last_by_reason: Dict[str, float] = {}
        self._event_times: List[float] = []
        self._burst_until = -float("inf")
        self._burst_reason = ""
        self._visited: set = set()
        self.counts: Dict[str, int] = {}

    def check(self, obs: Observation, decision: Optional[Decision], now: float) -> List[str]:
        """Reasons to keep this frame; [] = nothing special."""
        found = self._triggers(obs, decision)
        self._prev_mode = decision.mode if decision is not None else self._prev_mode
        self._prev_source = obs.detection.source if obs.detection is not None else None

        c = self.cfg
        fresh = [r for r in found if now - self._last_by_reason.get(r, -float("inf")) >= c.cooldown_s]
        self._event_times = [t for t in self._event_times if now - t < 60.0]
        if fresh and len(self._event_times) < c.max_events_per_min:
            for r in fresh:
                self._last_by_reason[r] = now
                self.counts[r] = self.counts.get(r, 0) + 1
            self._event_times.append(now)
            self._burst_until = now + c.burst_s
            self._burst_reason = fresh[0]
            return fresh
        if now < self._burst_until:
            return [f"burst:{self._burst_reason}"]
        return []

    # ---- the triggers ----------------------------------------------------------
    def _triggers(self, obs: Observation, decision: Optional[Decision]) -> List[str]:
        if decision is None:                       # on the ground: background trickle only
            return []
        c = self.cfg
        out: List[str] = []
        mode, prev = decision.mode, self._prev_mode
        if prev is not None and mode != prev:
            if mode == Mode.LOST:
                out.append("lost")
            elif mode == Mode.TRACK and prev in SEARCHING_MODES:
                out.append("reacquired")

        det = obs.detection
        src = det.source if det is not None else None
        if self._prev_source == "face" and src == "track":
            out.append("face_to_body")
        elif self._prev_source == "track" and src == "face":
            out.append("body_confirmed_by_face")
        if det is not None and det.size < c.far_target_size:
            out.append("far_target")

        stats = self.stats_fn() if self.stats_fn is not None else None
        if stats and stats.get("best_sim") is not None and stats.get("threshold") is not None:
            if abs(float(stats["best_sim"]) - float(stats["threshold"])) <= c.borderline_band:
                out.append("reid_borderline")

        if det is None and obs.scan is not None:
            near = [b.distance for b in obs.scan.beams if b.distance is not None]
            if near and min(near) < c.obstacle_m:
                out.append("close_obstacle")

        cell = (int(np.floor(obs.pose.x / c.new_place_cell_m)), int(np.floor(obs.pose.y / c.new_place_cell_m)))
        if cell not in self._visited:
            self._visited.add(cell)
            if len(self._visited) > 1:             # the launch spot itself is not news
                out.append("new_place")
        return out


class CuratedRecorder(Recorder):
    """A Recorder that saves event frames (curator) on top of a slow background rate, tags
    each frame with why it was kept, and can redact frames on its writer thread.

    curator=None: behaves like a plain Recorder at `fps` (useful with redact only)."""

    def __init__(self, root: str, info, curator: Optional[Curator] = None,
                 redact: Optional[Callable[[np.ndarray, dict], np.ndarray]] = None,
                 fps: Optional[float] = None, **kw):
        self.curator = curator
        self.redact = redact
        if fps is None:
            fps = curator.cfg.background_fps if curator is not None else 4.0
        self._event_gap = 1.0 / curator.cfg.event_fps if curator is not None else 0.0
        self._last_event_t = -float("inf")
        self._why: Dict[int, Sequence[str]] = {}
        self._why_lock = threading.Lock()
        super().__init__(root, info, fps=fps, **kw)

    def maybe_record(self, frame, obs=None, decision=None) -> bool:
        if frame is None or self.stopped:
            return False
        if self.curator is None or obs is None:
            return super().maybe_record(frame, obs, decision)
        now = self.clock()
        why = self.curator.check(obs, decision, now)
        urgent = any(not r.startswith("burst:") for r in why)   # a new event is never rate-limited away
        if why and (urgent or now - self._last_event_t >= self._event_gap):
            idx = self._next_index
            with self._why_lock:
                self._why[idx] = list(why)
            self._next_t = None                     # take this one now, whatever the background rate says
            ok = super().maybe_record(frame, obs, decision)
            if ok:
                self._last_event_t = now
            else:
                with self._why_lock:
                    self._why.pop(idx, None)
            return ok
        return super().maybe_record(frame, obs, decision)

    def status(self) -> dict:
        st = super().status()
        if self.curator is not None:
            st["events"] = dict(self.curator.counts)
        return st

    def _write(self, idx: int, frame: np.ndarray, t: float, meta: dict) -> None:
        with self._why_lock:
            why = self._why.pop(idx, None)
        meta = {**meta, "why": why or ["background"]} if self.curator is not None else meta
        if self.redact is not None:
            try:
                frame = self.redact(frame, meta)
            except Exception as e:                  # never save an unredacted frame by accident
                self._stop(f"redaction failed: {type(e).__name__}: {e}")
                return
        super()._write(idx, frame, t, meta)


# ---- privacy -----------------------------------------------------------------

def _inside(point, box) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def pixelate(frame: np.ndarray, box, blocks: int = 6) -> None:
    """Coarse mosaic over box (l, t, r, b), in place."""
    import cv2
    h, w = frame.shape[:2]
    l, t, r, b = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
    if r - l < 2 or b - t < 2:
        return
    roi = frame[t:b, l:r]
    small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_AREA)
    frame[t:b, l:r] = cv2.resize(small, (r - l, b - t), interpolation=cv2.INTER_NEAREST)


class FaceBlur:
    """redact() for CuratedRecorder: pixelate every face except the target's.

    The target = faces whose centre lies inside the box the drone was following
    (meta "live_box": the face box, or the body box when following by body).
    Uses its OWN YuNet instance, so it is safe on the recorder thread.
    `detect` (tests): frame -> list of (l, t, r, b) face boxes."""

    def __init__(self, model_path: str = "models/face_detection_yunet_2023mar.onnx",
                 detect: Optional[Callable[[np.ndarray], List]] = None, score: float = 0.6, pad: float = 0.15):
        self.pad = pad
        if detect is not None:
            self._detect = detect
        else:
            import cv2
            import os
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"face blur: YuNet model {model_path!r} not found (./setup.sh --models)")
            self._yunet = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score)
            self._detect = self._yunet_faces

    def _yunet_faces(self, frame: np.ndarray) -> List:
        h, w = frame.shape[:2]
        self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(frame)
        if faces is None:
            return []
        return [(int(f[0]), int(f[1]), int(f[0] + f[2]), int(f[1] + f[3])) for f in faces]

    def __call__(self, frame: np.ndarray, meta: dict) -> np.ndarray:
        keep = meta.get("live_box")
        for l, t, r, b in self._detect(frame):
            centre = ((l + r) / 2, (t + b) / 2)
            if keep is not None and _inside(centre, keep):
                continue
            pw, ph = (r - l) * self.pad, (b - t) * self.pad
            pixelate(frame, (l - pw, t - ph, r + pw, b + ph))
        return frame


# ---- construction (main.py make_recorder) -------------------------------------

def build_recorder(args, info):
    """The recorder main.py asks for. Plain Recorder unless --curate / --record-blur.
    Flags read with getattr: callers (tests) may pass a partial Namespace."""
    fps = getattr(args, "record_fps", None) or 4.0
    max_gb = getattr(args, "record_max_gb", None) or 20.0
    curate = bool(getattr(args, "curate", False))
    blur = bool(getattr(args, "record_blur", False))
    if not (curate or blur):
        return Recorder(args.record, info, fps=fps, max_gb=max_gb)
    redact = FaceBlur(getattr(args, "blur_model", None) or "models/face_detection_yunet_2023mar.onnx") if blur else None
    curator = Curator() if curate else None
    return CuratedRecorder(args.record, info, curator=curator, redact=redact,
                           fps=None if curate else fps, max_gb=max_gb)
