"""
R6: anonymous people tracks on the ground, and the zone events they cause.

Detections are points on the ground ((x, y) metres east / north of launch), one
batch per detector frame. The tracker links them into Tracks: "the same someone,
moving". A track remembers where it entered, the path it walked, which zones it
went through and for how long. Alerts become track events: "T3 has been in the
back gate zone for 1.5 s", once per track per zone, instead of "a point was in
the zone" (which re-alerted the same person every minute and missed a second one).

Linking: each track predicts where it is now (last position + velocity, for up
to max_extrapolate_s); a sighting within the gate (gate_m, growing by
gate_speed_mps per second unseen, max max_gate_m) can be that track. Closest
pairs first (greedy): fine for the few people a yard or a house has.

Lifecycle: new (tentative) -> confirmed after confirm_hits sightings within
confirm_window_s (else dropped: one false detection is not a person) -> ended
after forget_s unseen (the patrol camera looks away for long stretches, so
this is generous). Pure logic: no camera, no clock, no I/O.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from missions.geo import point_in_polygon
from missions.mission_config import TrackConfig

XY = Tuple[float, float]


@dataclass
class Track:
    id: str
    first_t: float
    last_t: float
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    confirmed: bool = False
    entry: XY = (0.0, 0.0)
    path: List[Tuple[float, float, float]] = field(default_factory=list)   # (t, x, y)
    zones_in: Dict[str, float] = field(default_factory=dict)               # zone -> entered at
    zones_seen: List[str] = field(default_factory=list)
    alerted: Set[str] = field(default_factory=set)

    def predict(self, t: float, max_extrapolate_s: float) -> XY:
        dt = min(max(0.0, t - self.last_t), max_extrapolate_s)
        return self.x + self.vx * dt, self.y + self.vy * dt

    def walked_m(self) -> float:
        return sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(self.path, self.path[1:]))

    def to_dict(self, now: float) -> dict:
        return {"id": self.id, "x": round(self.x, 1), "y": round(self.y, 1),
                "speed_mps": round(math.hypot(self.vx, self.vy), 2), "confirmed": self.confirmed,
                "age_s": round(now - self.first_t, 1), "unseen_s": round(now - self.last_t, 1),
                "zones": sorted(self.zones_in), "hits": self.hits}

    def summary(self) -> dict:
        return {"track": self.id, "duration_s": round(self.last_t - self.first_t, 1),
                "entry": [round(v, 1) for v in self.entry], "exit": [round(self.x, 1), round(self.y, 1)],
                "walked_m": round(self.walked_m(), 1), "zones": list(self.zones_seen), "hits": self.hits}


class GroundTracker:
    def __init__(self, config: Optional[TrackConfig] = None, zones: Sequence[dict] = (),
                 confirm_s: float = 1.5, zone_cooldown_s: float = 15.0):
        self.cfg = config or TrackConfig()
        self.zones = list(zones)
        self.confirm_s, self.zone_cooldown_s = confirm_s, zone_cooldown_s
        self.tracks: Dict[str, Track] = {}
        self._next = 1
        self._zone_last_alert: Dict[str, float] = {}

    # ---- queries ------------------------------------------------------------------
    def get(self, track_id: str) -> Optional[Track]:
        return self.tracks.get(track_id)

    def confirmed(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.confirmed]

    def people_in(self, zone: str, now: float, gap_s: float = 3.0) -> int:
        return sum(1 for t in self.confirmed() if zone in t.zones_in and now - t.last_t <= gap_s)

    def occupied(self, now: float, gap_s: float = 3.0) -> List[str]:
        return [z["name"] for z in self.zones if self.people_in(z["name"], now, gap_s)]

    # ---- updates ------------------------------------------------------------------
    def update(self, now: float, points: Sequence[XY]) -> Tuple[List[dict], List[Tuple[Track, str]]]:
        """One detector frame's ground points (possibly none). Returns (events,
        zone hits); a zone hit = (track, zone) that should raise an alert now."""
        c = self.cfg
        events: List[dict] = []
        pairs = []
        for tid, tr in self.tracks.items():
            px, py = tr.predict(now, c.max_extrapolate_s)
            gate = min(c.max_gate_m, c.gate_m + c.gate_speed_mps * max(0.0, now - tr.last_t))
            for j, (x, y) in enumerate(points):
                d = math.hypot(x - px, y - py)
                if d <= gate:
                    pairs.append((d, tid, j))
        pairs.sort()
        used_t, used_p = set(), set()
        hit: List[Track] = []
        for d, tid, j in pairs:
            if tid in used_t or j in used_p:
                continue
            used_t.add(tid)
            used_p.add(j)
            tr = self.tracks[tid]
            self._hit(tr, now, points[j], events)
            hit.append(tr)
        for j, p in enumerate(points):
            if j not in used_p:
                tr = self._new(now, p)
                events.append({"type": "track.new", "track": tr.id, "x": round(p[0], 1), "y": round(p[1], 1)})
                hit.append(tr)
        zone_hits = self._zones(now, hit, events)
        events += self.expire(now)
        return events, zone_hits

    def expire(self, now: float) -> List[dict]:
        """Drop tentative tracks that never confirmed; end confirmed ones unseen for forget_s."""
        c, events = self.cfg, []
        for tid, tr in list(self.tracks.items()):
            if not tr.confirmed and now - tr.first_t > c.confirm_window_s:
                del self.tracks[tid]
            elif tr.confirmed and now - tr.last_t > c.forget_s:
                del self.tracks[tid]
                events.append({"type": "track.end", **tr.summary()})
        return events

    # ---- internals ------------------------------------------------------------------
    def _new(self, now: float, p: XY) -> Track:
        tr = Track(f"T{self._next}", now, now, p[0], p[1], entry=(p[0], p[1]), path=[(now, p[0], p[1])])
        self._next += 1
        self.tracks[tr.id] = tr
        if self.cfg.confirm_hits <= 1:
            tr.confirmed = True
        return tr

    def _hit(self, tr: Track, now: float, p: XY, events: List[dict]) -> None:
        c = self.cfg
        dt = now - tr.last_t
        if dt > 1e-6:
            vx, vy = (p[0] - tr.x) / dt, (p[1] - tr.y) / dt
            s = math.hypot(vx, vy)
            if s > c.max_speed_mps:
                vx, vy = vx * c.max_speed_mps / s, vy * c.max_speed_mps / s
            a = c.velocity_alpha
            tr.vx, tr.vy = a * vx + (1 - a) * tr.vx, a * vy + (1 - a) * tr.vy
        tr.x, tr.y, tr.last_t = p[0], p[1], now
        tr.hits += 1
        tr.path.append((now, p[0], p[1]))
        del tr.path[:-c.path_points]
        if not tr.confirmed and tr.hits >= c.confirm_hits:
            tr.confirmed = True
            events.append({"type": "track.confirmed", "track": tr.id, "x": round(tr.x, 1), "y": round(tr.y, 1),
                           "entry": [round(v, 1) for v in tr.entry]})

    def _zones(self, now: float, hit: List[Track], events: List[dict]) -> List[Tuple[Track, str]]:
        out = []
        for tr in hit:
            for z in self.zones:
                name = z["name"]
                inside = point_in_polygon(tr.x, tr.y, z["polygon"])
                if inside and name not in tr.zones_in:
                    tr.zones_in[name] = now
                    if name not in tr.zones_seen:
                        tr.zones_seen.append(name)
                    events.append({"type": "track.zone", "track": tr.id, "zone": name, "enter": True})
                elif not inside and name in tr.zones_in:
                    since = tr.zones_in.pop(name)
                    events.append({"type": "track.zone", "track": tr.id, "zone": name, "enter": False,
                                   "dwell_s": round(now - since, 1)})
            if not tr.confirmed:
                continue
            for name, since in tr.zones_in.items():
                if (name not in tr.alerted and now - since >= self.confirm_s
                        and now - self._zone_last_alert.get(name, -math.inf) >= self.zone_cooldown_s):
                    tr.alerted.add(name)
                    self._zone_last_alert[name] = now
                    out.append((tr, name))
        return out
