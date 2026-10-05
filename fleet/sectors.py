"""
Mission 6 groundwork, several drones in one area: split it into sectors and
warn when two drones get too close. Pure functions; not wired into
fleet/server.py yet.

The catch, first: every drone's x / y is relative to ITS OWN launch point.
Positions of two drones are only comparable in a shared frame. Until drones
report GPS, give each one its launch point's offset from a shared origin
(e.g. measured on Google Maps): to_shared(x, y, offset).

Law (EU open category): one remote pilot flies one drone at a time. A swarm
needs the "specific" category (an operational authorisation from [redacted]).
Simulate it first (tools/fleet_demo.py).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from missions.mission_config import SectorConfig

XY = Tuple[float, float]


@dataclass(frozen=True)
class Sector:
    start_deg: float           # compass bearing from the centre, clockwise from north
    end_deg: float
    radius_m: float
    center: XY = (0.0, 0.0)

    def contains(self, x: float, y: float) -> bool:
        dx, dy = x - self.center[0], y - self.center[1]
        if math.hypot(dx, dy) > self.radius_m:
            return False
        b = math.degrees(math.atan2(dx, dy)) % 360.0
        width = (self.end_deg - self.start_deg) % 360.0 or 360.0
        return (b - self.start_deg) % 360.0 < width

    def middle(self, frac: float = 0.6) -> XY:
        """A point inside the sector (bearing half-way, frac of the radius): a first goal."""
        width = (self.end_deg - self.start_deg) % 360.0 or 360.0
        b = math.radians(self.start_deg + width / 2)
        r = self.radius_m * frac
        return self.center[0] + r * math.sin(b), self.center[1] + r * math.cos(b)


def assign_sectors(drone_ids: Sequence[str], radius_m: float, center: XY = (0.0, 0.0),
                   start_deg: float = 0.0) -> Dict[str, Sector]:
    """Equal pie slices, in sorted-id order (stable: the same drones always get the same slices)."""
    ids = sorted(drone_ids)
    if not ids:
        return {}
    width = 360.0 / len(ids)
    return {d: Sector((start_deg + k * width) % 360.0, (start_deg + (k + 1) * width) % 360.0, radius_m, center)
            for k, d in enumerate(ids)}


def to_shared(x: float, y: float, launch_offset: XY) -> XY:
    return x + launch_offset[0], y + launch_offset[1]


def too_close(positions: Dict[str, Tuple[float, float, float]],
              config: SectorConfig = SectorConfig()) -> List[Tuple[str, str, float]]:
    """Pairs closer than min_horizontal_m AND min_vertical_m (x, y, z in the shared frame).
    Returns (a, b, horizontal distance), closest first."""
    ids = sorted(positions)
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            (xa, ya, za), (xb, yb, zb) = positions[a], positions[b]
            d = math.hypot(xa - xb, ya - yb)
            if d < config.min_horizontal_m and abs(za - zb) < config.min_vertical_m:
                out.append((a, b, round(d, 1)))
    return sorted(out, key=lambda t: t[2])
