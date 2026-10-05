"""
R7 learn normal: what does each viewpoint usually show? Report what changed.

A Place is a viewpoint: a cell_m square the drone was in x the compass sector it
faced x the camera tilt. The patrol keeps coming back to the same places, so for
each one we learn how often each object label (COCO: chair, laptop, bicycle ...)
is in view. Then:

    appeared   seen now, at a place where it (almost) never is      "new backpack near the gate"
    missing    not seen now, at a place where it (almost) always is  "laptop missing in the study"

Visits, not frames: one pass through a viewpoint (frames until the drone moves to
another viewpoint, or revisit_gap_s without one) counts once. A label is "there"
on a visit if it was in at least present_frac of that visit's frames. So one
missed detection, or a person briefly blocking the view, is not a change.

A change must hold confirm_visits visits in a row before it is reported; while
it is being confirmed that label is not learned (or the change would be averaged
away before it is confirmed). Once reported, learning resumes quietly until the
new state is fully normal (a new sofa stops being news), then the label is
watched again. Each (place, label) is reported at most once per cooldown_s, and
each (label, kind) at most once per label_cooldown_s across all places.

Needs a repeatable pose: launch from the same spot (like --map); indoor optical
flow drifts, so keep cell_m >= 1 m. Saved to / loaded from JSON (--baseline).
Pure logic: no camera, no I/O except save/load.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from missions.entities import Scene
from missions.mission_config import BaselineConfig


@dataclass
class Change:
    t: float
    place: str
    x: float                      # viewpoint centre (where the drone was)
    y: float
    label: str
    kind: str                     # "appeared" | "missing"
    p: float                      # how often it was there before (0..1)
    visits: int                   # visits the place had
    frame: Optional[np.ndarray] = field(default=None, repr=False)
    box: Optional[Tuple[int, int, int, int]] = None
    pose: object = None
    pitch_deg: float = 0.0

    def to_dict(self) -> dict:
        return {"t": round(self.t, 2), "place": self.place, "x": round(self.x, 1), "y": round(self.y, 1),
                "label": self.label, "kind": self.kind, "p": round(self.p, 2), "visits": self.visits}


@dataclass
class _Place:
    n: int = 0
    p: Dict[str, float] = field(default_factory=dict)
    appear: Dict[str, int] = field(default_factory=dict)
    missing: Dict[str, int] = field(default_factory=dict)
    last_report: Dict[str, float] = field(default_factory=dict)
    settling: Dict[str, str] = field(default_factory=dict)  # label -> change kind, being learned as normal


@dataclass
class _Visit:
    key: str
    first_t: float
    last_t: float
    frames: int = 0
    counts: Dict[str, int] = field(default_factory=dict)
    frame: Optional[np.ndarray] = None
    boxes: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    pose: object = None
    pitch_deg: float = 0.0


class SceneBaseline:
    def __init__(self, config: Optional[BaselineConfig] = None, labels: Optional[Sequence[str]] = None):
        self.cfg = config or BaselineConfig()
        self.labels: Set[str] = set(labels if labels is not None else self.cfg.labels)
        self.places: Dict[str, _Place] = {}
        self._visit: Optional[_Visit] = None
        self._label_last: Dict[Tuple[str, str], float] = {}
        self.changes = 0

    # ---- viewpoints ---------------------------------------------------------------
    def key(self, pose, pitch_deg: float) -> str:
        c = self.cfg
        ix, iy = math.floor(pose.x / c.cell_m), math.floor(pose.y / c.cell_m)
        ih = int(round(pose.yaw / (2 * math.pi / c.headings))) % c.headings
        ip = int(round(pitch_deg / c.pitch_bucket_deg))
        return f"{ix},{iy},{ih},{ip}"

    def centre(self, key: str) -> Tuple[float, float]:
        ix, iy = (int(v) for v in key.split(",")[:2])
        return (ix + 0.5) * self.cfg.cell_m, (iy + 0.5) * self.cfg.cell_m

    # ---- updates ------------------------------------------------------------------
    def observe(self, scene: Scene) -> List[Change]:
        """One detector frame. Returns changes found by closing the previous visit, if any."""
        k = self.key(scene.pose, scene.pitch_deg)
        out: List[Change] = []
        v = self._visit
        if v is not None and (v.key != k or scene.t - v.last_t > self.cfg.revisit_gap_s):
            out = self._close(v, scene.t)
            v = self._visit = None
        if v is None:
            v = self._visit = _Visit(k, scene.t, scene.t)
        v.last_t = scene.t
        v.frames += 1
        seen = set()
        for o in scene.objects:
            if o.label in self.labels and o.label not in seen:
                seen.add(o.label)
                v.counts[o.label] = v.counts.get(o.label, 0) + 1
                if o.box is not None:
                    v.boxes[o.label] = o.box
        if scene.frame is not None:
            v.frame, v.pose, v.pitch_deg = scene.frame, scene.pose, scene.pitch_deg
        return out

    def flush(self, now: float) -> List[Change]:
        """Close a visit that has gone quiet (the drone left, or detection stopped)."""
        v = self._visit
        if v is not None and now - v.last_t > self.cfg.revisit_gap_s:
            self._visit = None
            return self._close(v, now)
        return []

    def _close(self, v: _Visit, now: float) -> List[Change]:
        c = self.cfg
        if v.frames < c.min_frames:
            return []
        present = {l for l, n in v.counts.items() if n / v.frames >= c.present_frac}
        place = self.places.setdefault(v.key, _Place())
        labels = set(place.p) | present
        out: List[Change] = []
        if place.n >= c.min_visits:
            for l in labels:
                if l in place.settling:
                    continue                             # reported: let it become the new normal quietly
                p = place.p.get(l, 0.0)
                place.appear[l] = place.appear.get(l, 0) + 1 if (l in present and p <= c.rare_p) else 0
                place.missing[l] = place.missing.get(l, 0) + 1 if (l not in present and p >= c.common_p) else 0
                for kind, streak in (("appeared", place.appear), ("missing", place.missing)):
                    if streak[l] < c.confirm_visits:
                        continue
                    streak[l] = 0
                    place.settling[l] = kind
                    if now - place.last_report.get(l, -math.inf) < c.cooldown_s:
                        continue
                    place.last_report[l] = now
                    if now - self._label_last.get((l, kind), -math.inf) < c.label_cooldown_s:
                        continue                         # another viewpoint already reported it
                    self._label_last[(l, kind)] = now
                    x, y = self.centre(v.key)
                    out.append(Change(now, v.key, x, y, l, kind, p, place.n, v.frame,
                                      v.boxes.get(l) if kind == "appeared" else None, v.pose, v.pitch_deg))
        place.n += 1
        a = 1.0 / place.n if place.n <= c.learn_visits else c.alpha
        for l in labels:
            if place.appear.get(l, 0) or place.missing.get(l, 0):
                continue                                 # a change being confirmed: don't learn it away yet
            p = place.p.get(l, 0.0) + a * ((1.0 if l in present else 0.0) - place.p.get(l, 0.0))
            done = {"missing": p <= c.rare_p, "appeared": p >= c.common_p}
            if done.get(place.settling.get(l), False):
                del place.settling[l]                    # the new state is fully learned: watch it again
            if p < 0.01 and l not in present:
                place.p.pop(l, None)
                place.appear.pop(l, None)
                place.missing.pop(l, None)
                place.settling.pop(l, None)
            else:
                place.p[l] = p
        self.changes += len(out)
        return out

    # ---- status / persistence ---------------------------------------------------------
    def status(self) -> dict:
        learned = sum(1 for p in self.places.values() if p.n >= self.cfg.min_visits)
        return {"places": len(self.places), "learned": learned, "changes": self.changes,
                "objects": sum(len(p.p) for p in self.places.values())}

    def to_json(self) -> dict:
        c = self.cfg
        return {"version": 1, "cell_m": c.cell_m, "headings": c.headings, "pitch_bucket_deg": c.pitch_bucket_deg,
                "places": {k: {"n": p.n, "p": {l: round(v, 4) for l, v in p.p.items()}}
                           for k, p in self.places.items()}}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f)

    def load(self, path: str) -> int:
        """Returns how many places were loaded. Raises ValueError on a file made with other settings."""
        with open(path) as f:
            data = json.load(f)
        c = self.cfg
        if (data.get("version") != 1 or data.get("cell_m") != c.cell_m or data.get("headings") != c.headings
                or data.get("pitch_bucket_deg") != c.pitch_bucket_deg):
            raise ValueError(f"{path}: made with other baseline settings (cell_m / headings / pitch bucket)")
        self.places = {k: _Place(n=int(v["n"]), p={l: float(x) for l, x in v["p"].items()})
                       for k, v in data["places"].items()}
        return len(self.places)
