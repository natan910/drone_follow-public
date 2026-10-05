"""
The toy world for the perimeter watch: people who walk in, objects that go
missing, all seen by a pretend detector. No images: the toy world has none.

    Walker      a scripted person: appears at start_t, walks a path at speed_mps,
                waits, then vanishes (leaves)
    SimObject   a thing at a fixed spot (laptop, bicycle), present from since_t until until_t
    SimSceneSource   each every_s: whatever is inside the camera's horizontal field of
                view, between min_range_m and view_range_m, and not behind a wall
                (the world's boxes) -> a Scene, like missions/detect.py makes from a frame.
                Positions get noise_m of jitter; miss_prob drops a sighting now and then.
    render_topdown   a picture: walls, zones, the drone and its view, true people
                (blue), tracks (orange, with ids), objects, the investigate spot.

Used by tools/patrol_sim.py and the tests. The world object only needs
`.boxes` (each with x0, y0, x1, y1), like sim/scenarios.py demo_world().
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from missions.entities import ObjectSighting, PersonSighting, Scene

XY = Tuple[float, float]


@dataclass
class Walker:
    path: List[XY]
    start_t: float = 0.0
    speed_mps: float = 1.0
    wait_s: float = 30.0          # at the end of the path, then gone
    name: str = "intruder"

    def _length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.path, self.path[1:]))

    def pos(self, t: float) -> Optional[XY]:
        if t < self.start_t or not self.path:
            return None
        d = (t - self.start_t) * self.speed_mps
        total = self._length()
        if d > total + self.wait_s * self.speed_mps:
            return None
        for a, b in zip(self.path, self.path[1:]):
            seg = math.dist(a, b)
            if d <= seg:
                k = d / seg if seg > 0 else 0.0
                return a[0] + k * (b[0] - a[0]), a[1] + k * (b[1] - a[1])
            d -= seg
        return self.path[-1]


@dataclass
class SimObject:
    label: str
    x: float
    y: float
    since_t: float = -math.inf
    until_t: float = math.inf

    def present(self, t: float) -> bool:
        return self.since_t <= t < self.until_t


def _blocked(world, a: XY, b: XY) -> bool:
    """Does the segment a-b cross any of the world's boxes (slab test)?"""
    for bx in getattr(world, "boxes", ()):
        t0, t1 = 0.0, 1.0
        ok = True
        for p, d, lo, hi in ((a[0], b[0] - a[0], bx.x0, bx.x1), (a[1], b[1] - a[1], bx.y0, bx.y1)):
            if abs(d) < 1e-12:
                if p < lo or p > hi:
                    ok = False
                    break
                continue
            u0, u1 = (lo - p) / d, (hi - p) / d
            if u0 > u1:
                u0, u1 = u1, u0
            t0, t1 = max(t0, u0), min(t1, u1)
            if t0 > t1:
                ok = False
                break
        if ok:
            return True
    return False


class SimSceneSource:
    def __init__(self, world, walkers: Sequence[Walker] = (), objects: Sequence[SimObject] = (),
                 hfov_deg: float = 66.0, view_range_m: float = 8.0, min_range_m: float = 0.8,
                 every_s: float = 0.5, noise_m: float = 0.15, miss_prob: float = 0.0, seed: int = 1):
        self.world, self.walkers, self.objects = world, list(walkers), list(objects)
        self.hfov, self.view_range, self.min_range = hfov_deg, view_range_m, min_range_m
        self.every_s, self.noise, self.miss_prob = every_s, noise_m, miss_prob
        self.rng = random.Random(seed)
        self._last = -math.inf
        self.runs = 0

    def visible(self, pose, p: XY) -> bool:
        dx, dy = p[0] - pose.x, p[1] - pose.y
        d = math.hypot(dx, dy)
        if not (self.min_range <= d <= self.view_range):
            return False
        rel = (math.atan2(dx, dy) - pose.yaw + math.pi) % (2 * math.pi) - math.pi
        return abs(math.degrees(rel)) <= self.hfov / 2 and not _blocked(self.world, (pose.x, pose.y), p)

    def step(self, frame, obs) -> Optional[Scene]:
        if obs.now - self._last < self.every_s:
            return None
        self._last = obs.now
        self.runs += 1
        people, objects = [], []
        for w in self.walkers:
            p = w.pos(obs.now)
            if p is not None and self.visible(obs.pose, p) and self.rng.random() >= self.miss_prob:
                people.append(PersonSighting(p[0] + self.rng.gauss(0, self.noise),
                                             p[1] + self.rng.gauss(0, self.noise), 0.9))
        for o in self.objects:
            if o.present(obs.now) and self.visible(obs.pose, (o.x, o.y)) and self.rng.random() >= self.miss_prob:
                objects.append(ObjectSighting(o.label, 0.9))
        return Scene(obs.now, obs.pose, obs.camera_pitch_deg, people, objects, None, 0.0)

    def stats(self) -> dict:
        return {"runs": self.runs, "sim": True}


# ---- picture ---------------------------------------------------------------------------

def render_topdown(world, pose, watch=None, walkers: Sequence[Walker] = (), objects: Sequence[SimObject] = (),
                   t: float = 0.0, investigating: Optional[dict] = None, hfov_deg: float = 66.0,
                   view_range_m: float = 8.0, px_per_m: float = 30.0, lines: Sequence[str] = ()) -> np.ndarray:
    import cv2
    boxes = list(getattr(world, "boxes", ()))
    xs = [v for b in boxes for v in (b.x0, b.x1)] or [-10, 10]
    ys = [v for b in boxes for v in (b.y0, b.y1)] or [-10, 10]
    x0, x1, y0, y1 = min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1
    w, h = int((x1 - x0) * px_per_m), int((y1 - y0) * px_per_m)
    panel = 22 * (len(lines) + 1)
    img = np.full((h + panel, w, 3), 245, np.uint8)

    def P(x, y):
        return int((x - x0) * px_per_m), int((y1 - y) * px_per_m)

    zones = watch.zones if watch is not None else []
    occupied = set(watch.tracker.occupied(t)) if watch is not None else set()
    for z in zones:
        pts = np.array([P(*p) for p in z["polygon"]], np.int32)
        col = (60, 60, 230) if z["name"] in occupied else (170, 170, 170)
        cv2.polylines(img, [pts], True, col, 2)
        cv2.putText(img, z["name"], pts[0] + (4, -6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    for b in boxes:
        cv2.rectangle(img, P(b.x0, b.y1), P(b.x1, b.y0), (90, 90, 90), -1)
    for o in objects:
        col = (60, 160, 60) if o.present(t) else (200, 200, 200)
        cv2.rectangle(img, (P(o.x, o.y)[0] - 5, P(o.x, o.y)[1] - 5), (P(o.x, o.y)[0] + 5, P(o.x, o.y)[1] + 5), col, -1)
        cv2.putText(img, o.label, (P(o.x, o.y)[0] + 7, P(o.x, o.y)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    for wk in walkers:
        p = wk.pos(t)
        if p is not None:
            cv2.circle(img, P(*p), 6, (220, 120, 30), -1)
    if watch is not None:
        for tr in watch.tracker.confirmed():
            cv2.circle(img, P(tr.x, tr.y), 11, (0, 140, 255), 2)
            cv2.putText(img, tr.id, (P(tr.x, tr.y)[0] + 12, P(tr.x, tr.y)[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 110, 230), 1)
    if investigating:
        c = P(investigating["x"], investigating["y"])
        cv2.drawMarker(img, c, (200, 0, 200), cv2.MARKER_TILTED_CROSS, 18, 2)
    half = math.radians(hfov_deg) / 2
    for a in (pose.yaw - half, pose.yaw + half):
        cv2.line(img, P(pose.x, pose.y), P(pose.x + view_range_m * math.sin(a), pose.y + view_range_m * math.cos(a)),
                 (200, 200, 120), 1)
    tip = P(pose.x + 0.6 * math.sin(pose.yaw), pose.y + 0.6 * math.cos(pose.yaw))
    left = P(pose.x + 0.35 * math.sin(pose.yaw - 2.5), pose.y + 0.35 * math.cos(pose.yaw - 2.5))
    right = P(pose.x + 0.35 * math.sin(pose.yaw + 2.5), pose.y + 0.35 * math.cos(pose.yaw + 2.5))
    cv2.fillPoly(img, [np.array([tip, left, right], np.int32)], (30, 30, 30))
    for i, text in enumerate(lines):
        cv2.putText(img, text, (8, h + 18 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
    return img
