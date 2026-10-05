"""
Mission 1, perimeter watch: fly the normal PATROL (tap "Patrol" on the phone
page), watch for people and for things that changed, raise ALERTS.

    zones.json    {"home": [lat, lon],                                  <- only for "latlon"
                   "property": {"polygon": [[x, y], ...]},              <- optional, see below
                   "areas": [{"name": "back gate", "polygon": [[x, y], ...]}]}
                  metres east / north of the launch point, or "latlon" instead of
                  "polygon" (see missions/geo.py load_polygons)

"property" (optional, recommended outdoors): the land you are allowed to watch.
  - people whose feet land outside it (street, neighbours) never alert
  - alert snapshots black out the sky and every ground pixel outside it
  - the patrol radius is sized to it (missions/wiring.py apply_patrol_profile)
  - with no "areas", the whole property is the one zone ("property")

One step (main loop, every frame):
  1. source.step(frame, obs) -> a Scene now and then (missions/detect.py: the
     detector runs in its own thread, R9; or the toy world's ground truth,
     missions/simsource.py)
  2. people outside the property dropped; the rest -> GroundTracker (R6,
     missions/tracks.py): anonymous tracks; a confirmed track inside a zone for
     confirm_s -> ALERT "intrusion", once per track per zone
  3. objects -> SceneBaseline (R7, missions/baseline.py, --baseline): an object
     that appeared / went missing at a viewpoint -> ALERT "object_appeared" / "object_missing"
  4. every alert: snapshot (privacy-masked outdoors), status JSON, events.jsonl,
     on_alert callbacks (missions/notify.py pushes it to a phone)

The watch never steers. Going to look (R1) is PerimeterMission's job
(missions/wiring.py), through the autopilot's investigate() call.

The enrolled person (the owner) is skipped: walking your own yard is not an
alarm. Only works while the matcher has them in that frame.

Legal: filming people for security = GDPR. Your own property; signs up; no
filming of neighbours' gardens or the street (the privacy mask helps, the
camera still sees them: keep the mask on and the recorder off).
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from datatypes import Observation
from missions.detect import ImageSceneSource, covers as _covers        # noqa: F401 (re-exported)
from missions.entities import Alert, EventLog, Scene
from missions.geo import XY, ground_point, latlon_to_local, load_polygons, point_in_polygon
from missions.mission_config import PerimeterConfig, TrackConfig
from missions.tracks import GroundTracker


@dataclass
class _ZoneState:
    first: Optional[float] = None
    last: Optional[float] = None
    last_alert: float = -float("inf")


class ZoneMonitor:
    """The simple point-based monitor (no tracks): one alert per zone per cooldown_s.
    Kept for tools and tests; PerimeterWatch uses GroundTracker."""

    def __init__(self, zones: Sequence[dict], config: Optional[PerimeterConfig] = None):
        self.zones = list(zones)
        self.cfg = config or PerimeterConfig()
        self._st: Dict[str, _ZoneState] = {z["name"]: _ZoneState() for z in self.zones}

    def update(self, now: float, points: Sequence[XY]) -> List[Alert]:
        c = self.cfg
        alerts = []
        for z in self.zones:
            st = self._st[z["name"]]
            inside = [p for p in points if point_in_polygon(p[0], p[1], z["polygon"])]
            if inside:
                if st.first is None or st.last is None or now - st.last > c.gap_s:
                    st.first = now
                st.last = now
                if now - st.first >= c.confirm_s and now - st.last_alert >= c.cooldown_s:
                    st.last_alert = now
                    x, y = inside[0]
                    alerts.append(Alert(z["name"], now, x, y, len(inside)))
            elif st.last is not None and now - st.last > c.gap_s:
                st.first = None
        return alerts

    def occupied(self, now: float) -> List[str]:
        return [n for n, st in self._st.items() if st.last is not None and now - st.last <= self.cfg.gap_s]


# ---- zones file ----------------------------------------------------------------

def load_watch_file(path: str) -> Tuple[List[dict], Optional[List[XY]]]:
    """zones.json -> (zones, property polygon or None). See the module docstring."""
    with open(path) as f:
        data = json.load(f)
    prop = None
    if isinstance(data, dict) and data.get("property") is not None:
        prop = _property_polygon(path, data)
    has_areas = isinstance(data, list) or (isinstance(data, dict) and data.get("areas"))
    if has_areas:
        zones = load_polygons(path)
    elif prop is not None:
        zones = [{"name": "property", "polygon": prop}]
    else:
        raise ValueError(f"{path}: needs \"areas\", \"property\", or both")
    return zones, prop


def _property_polygon(path: str, data: dict) -> List[XY]:
    p = data["property"]
    if "polygon" in p:
        poly = [(float(q[0]), float(q[1])) for q in p["polygon"]]
    elif "latlon" in p:
        if "home" not in data:
            raise ValueError(f"{path}: \"property\" uses latlon but the file has no \"home\"")
        home = tuple(float(v) for v in data["home"])
        poly = [latlon_to_local(float(q[0]), float(q[1]), *home) for q in p["latlon"]]
    else:
        raise ValueError(f"{path}: \"property\" needs \"polygon\" or \"latlon\"")
    if len(poly) < 3:
        raise ValueError(f"{path}: \"property\" needs at least 3 corners")
    return poly


def centroid(poly: Sequence[XY]) -> XY:
    """Area centroid of a simple polygon (the plain average for a degenerate one)."""
    a = cx = cy = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a) < 1e-9:
        return sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n
    return cx / (3 * a), cy / (3 * a)


# ---- privacy mask ----------------------------------------------------------------

def privacy_mask(width: int, height: int, pose, pitch_deg: float, hfov_deg: float, aspect: float,
                 prop: Optional[Sequence[XY]], cell_px: int = 16) -> np.ndarray:
    """HxW bool, True = may be kept: the pixel looks at the ground inside `prop`
    (any ground when prop is None). Sky / horizon / outside = False. Tested on a
    grid of cell centres, so edges are cell_px-blocky (on purpose: cheap on a Pi)."""
    ch, cw = max(1, -(-height // cell_px)), max(1, -(-width // cell_px))
    keep = np.zeros((ch, cw), bool)
    for j in range(ch):
        v = min(height - 1, j * cell_px + cell_px / 2)
        for i in range(cw):
            u = min(width - 1, i * cell_px + cell_px / 2)
            p = ground_point(u, v, width, height, pose, pitch_deg, hfov_deg, aspect, min_down_deg=0.0)
            keep[j, i] = p is not None and (prop is None or point_in_polygon(p[0], p[1], prop))
    return np.repeat(np.repeat(keep, cell_px, 0), cell_px, 1)[:height, :width]


# ---- the watch -------------------------------------------------------------------

class PerimeterWatch:
    """Called once per main-loop step. `detector` builds the usual camera source;
    `source` replaces it (the toy world's ground truth)."""

    def __init__(self, detector=None, zones: Sequence[dict] = (), hfov_deg: float = 66.0, aspect: float = 0.75,
                 config: Optional[PerimeterConfig] = None, alerts_dir: Optional[str] = None,
                 imwrite: Optional[Callable[[str, np.ndarray], bool]] = None,
                 property_polygon: Optional[Sequence[XY]] = None,
                 on_alert: Sequence[Callable[[Alert], None]] = (), source=None, baseline=None,
                 events: Optional[EventLog] = None, threaded: bool = False,
                 track_config: Optional[TrackConfig] = None):
        self.cfg = config or PerimeterConfig()
        self.detector = detector
        if source is None:
            if detector is None:
                raise ValueError("PerimeterWatch needs a detector or a source")
            source = ImageSceneSource(detector, hfov_deg, aspect, self.cfg, threaded=threaded)
        self.source = source
        self.zones = list(zones)
        self.hfov, self.aspect = hfov_deg, aspect
        self.alerts_dir = alerts_dir
        self.property = list(property_polygon) if property_polygon else None
        self.on_alert = list(on_alert)
        self.baseline = baseline
        self.events = events or EventLog()
        self.tracker = GroundTracker(track_config or TrackConfig(), self.zones, self.cfg.confirm_s,
                                     self.cfg.zone_cooldown_s)
        self._imwrite = imwrite
        self.recent: List[Alert] = []
        self.last_people: List[XY] = []
        self.last_scene: Optional[Scene] = None
        self.outside_count = 0          # people ignored because they stood outside the property
        self.far_count = 0              # people seen but not placeable (too far, above the horizon)
        self.snapshots = 0

    # ---- main loop ------------------------------------------------------------------
    def step(self, frame: Optional[np.ndarray], obs: Observation) -> List[Alert]:
        scene = self.source.step(frame, obs)
        alerts: List[Alert] = []
        if scene is not None:
            self.last_scene = scene
            alerts += self._people(scene)
            if self.baseline is not None:
                alerts += [self._change(c) for c in self.baseline.observe(scene)]
        else:
            self._log(self.tracker.expire(obs.now), obs.now)
        if self.baseline is not None:
            alerts += [self._change(c) for c in self.baseline.flush(obs.now)]
        for a in alerts:
            self._emit(a)
        del self.recent[:-self.cfg.keep_alerts]
        return alerts

    def snapshot(self, tag: str, note: str = "") -> Optional[str]:
        """Save the latest detector frame (used while investigating). Not pushed."""
        s = self.last_scene
        if s is None or s.frame is None:
            return None
        a = Alert(tag, s.t, s.pose.x, s.pose.y, 0, kind="snapshot", detail=note)
        path = self._save(s.frame, [p.box for p in s.people if p.box], a, s.pose, s.pitch_deg)
        if path:
            self.snapshots += 1
        return path

    def status(self, now: float) -> dict:
        st = {"mission": "perimeter", "zones": len(self.zones), "property": self.property is not None,
              "people_seen": len(self.last_people), "occupied": self.tracker.occupied(now, self.cfg.gap_s),
              "outside_ignored": self.outside_count, "too_far": self.far_count,
              "tracks": [t.to_dict(now) for t in self.tracker.confirmed()],
              "alerts": [a.to_dict() for a in self.recent[-5:]], "alert_count": len(self.recent),
              "events": list(self.events.recent)[-5:], "snapshots": self.snapshots}
        if hasattr(self.source, "stats"):
            st["detector"] = self.source.stats()
        if self.baseline is not None:
            st["baseline"] = self.baseline.status()
        return st

    def close(self) -> None:
        if hasattr(self.source, "close"):
            self.source.close()

    # ---- internals --------------------------------------------------------------------
    def _people(self, scene: Scene) -> List[Alert]:
        c = self.cfg
        self.far_count += scene.far
        points, boxes = [], []
        for s in scene.people:
            if self.property and c.ignore_outside_property and not point_in_polygon(s.x, s.y, self.property):
                self.outside_count += 1
                continue
            points.append((s.x, s.y))
            if s.box is not None:
                boxes.append(s.box)
        self.last_people = points
        events, hits = self.tracker.update(scene.t, points)
        self._log(events, scene.t)
        alerts = []
        for tr, zone in hits:
            a = Alert(zone, scene.t, tr.x, tr.y, self.tracker.people_in(zone, scene.t, c.gap_s),
                      kind="intrusion", track=tr.id)
            a.snapshot = self._save(scene.frame, boxes, a, scene.pose, scene.pitch_deg)
            alerts.append(a)
        return alerts

    def _change(self, ch) -> Alert:
        zone = next((z["name"] for z in self.zones if point_in_polygon(ch.x, ch.y, z["polygon"])),
                    f"view from ({ch.x:.0f}, {ch.y:.0f})")
        a = Alert(zone, ch.t, ch.x, ch.y, 0, kind=f"object_{ch.kind}", detail=ch.label)
        self.events.write("change", ch.t, **ch.to_dict(), zone=zone)
        if ch.frame is not None and ch.pose is not None:
            a.snapshot = self._save(ch.frame, [ch.box] if ch.box else [], a, ch.pose, ch.pitch_deg)
        return a

    def _emit(self, a: Alert) -> None:
        self.recent.append(a)
        self.events.write("alert", a.t, **a.to_dict())
        what = f"person ({a.track})" if a.kind == "intrusion" else f"{a.detail} {a.kind.split('_')[-1]}"
        print(f"ALERT: {what} in zone {a.zone!r} at ({a.x:.1f}, {a.y:.1f}) m")
        for cb in self.on_alert:
            try:
                cb(a)
            except Exception as e:                      # a broken notifier never stops the watch
                print(f"Alert callback failed: {type(e).__name__}: {e}")

    def _log(self, events: List[dict], t: float) -> None:
        for e in events:
            kind = e.pop("type")
            self.events.write(kind, t, **e)

    def _save(self, frame: Optional[np.ndarray], boxes, alert: Alert, pose, pitch_deg: float) -> Optional[str]:
        if not self.alerts_dir or frame is None:
            return None
        import cv2
        os.makedirs(self.alerts_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(alert.wall_time))
        safe = "".join(ch if ch.isalnum() else "_" for ch in alert.zone)
        base = os.path.join(self.alerts_dir, f"{stamp}_{alert.kind}_{safe}")
        n, path = 1, base
        while os.path.exists(path + ".jpg"):          # several in one second
            n += 1
            path = f"{base}_{n}"
        img = frame.copy()
        if self.cfg.privacy_mask:
            h, w = img.shape[:2]
            keep = privacy_mask(w, h, pose, pitch_deg, self.hfov, self.aspect, self.property, self.cfg.mask_cell_px)
            img[~keep] = 0
        for l, t, r, b in boxes:
            cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), (0, 0, 255), 2)
        (self._imwrite or cv2.imwrite)(path + ".jpg", img)
        alert.snapshot = path + ".jpg"
        with open(path + ".json", "w") as f:
            json.dump(alert.to_dict(), f)
        return alert.snapshot
