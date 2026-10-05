"""
Mission 5, field survey: fly back and forth over a field ("lawnmower"),
score how green each patch is (perception/veg_index.py), keep a map of it.

Flying it, today (no brain changes):
  1. Draw the field in field.json (missions/geo.py load_polygons; "latlon" corners
     from Google Maps + "home" = your launch spot).
  2. main.py ... --mission survey --survey-area field.json --home-latlon LAT,LON
     writes survey.waypoints (lawnmower at SurveyConfig.altitude_m).
  3. Load survey.waypoints in Mission Planner / QGroundControl, upload, fly it in
     AUTO. The flight controller flies the lines itself; our brain sees the pilot /
     AUTO has control (Mode IDLE, sends nothing) and only LOGS: every frame is
     scored and dropped into the field map (--field-map field_map.json).
  Camera must look down: --fixed-camera --fixed-pitch 90 (or a gimbal at 90).

Later: flying the lines ourselves (GUIDED / our own flightcore) = a new autopilot
Task; the planner below is already what it would follow.
"""

import math
from typing import List, Optional, Sequence, Tuple

from missions.geo import XY, local_to_latlon

MAV_CMD_NAV_WAYPOINT, MAV_CMD_NAV_TAKEOFF, MAV_CMD_NAV_RTL = 16, 22, 20
FRAME_GLOBAL, FRAME_GLOBAL_RELATIVE_ALT = 0, 3


def swath_width(altitude_m: float, hfov_deg: float, overlap: float) -> float:
    """Ground width covered by one pass (camera straight down), minus the overlap."""
    return 2 * altitude_m * math.tan(math.radians(hfov_deg) / 2) * (1 - overlap)


def _rot(p: XY, a: float) -> XY:
    c, s = math.cos(a), math.sin(a)
    return p[0] * c - p[1] * s, p[0] * s + p[1] * c


def longest_edge_angle(poly: Sequence[XY]) -> float:
    best, ang = -1.0, 0.0
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        d = math.hypot(x2 - x1, y2 - y1)
        if d > best:
            best, ang = d, math.atan2(y2 - y1, x2 - x1)
    return ang


def lawnmower(poly: Sequence[XY], spacing_m: float, angle_rad: Optional[float] = None,
              inset_m: float = 0.0) -> List[XY]:
    """Back-and-forth passes `spacing_m` apart, parallel to the field's longest edge
    (or `angle_rad`). Each pass spans the field's outer width at that line: exact for
    convex fields; for an L-shaped field it also flies over the notch.
    inset_m: stop this far short of the edges (trees, fences)."""
    if spacing_m <= 0:
        raise ValueError("spacing must be positive")
    a = longest_edge_angle(poly) if angle_rad is None else angle_rad
    rp = [_rot(p, -a) for p in poly]                    # passes are now horizontal lines
    ys = [p[1] for p in rp]
    span = max(ys) - min(ys)
    n = max(1, math.ceil(span / spacing_m - 1e-9))       # passes: evenly spread, never wider apart than spacing
    out: List[XY] = []
    flip = False
    for k in range(n):
        y = min(ys) + (k + 0.5) * span / n
        xs = []
        for i in range(len(rp)):
            (x1, y1), (x2, y2) = rp[i], rp[(i + 1) % len(rp)]
            if (y1 > y) != (y2 > y):
                xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        if len(xs) >= 2:
            x0, x1 = min(xs) + inset_m, max(xs) - inset_m
            if x1 > x0:
                seg = [(x0, y), (x1, y)]
                out += [_rot(p, a) for p in (reversed(seg) if flip else seg)]
                flip = not flip
    return out


def path_length(points: Sequence[XY], start: XY = (0.0, 0.0)) -> float:
    total, prev = 0.0, start
    for p in points:
        total += math.hypot(p[0] - prev[0], p[1] - prev[1])
        prev = p
    return total


class CoverageTracker:
    """Which lawnmower waypoint is next, from the drone's pose. For our own autopilot
    later; also gives progress while ArduPilot flies it in AUTO."""

    def __init__(self, waypoints: Sequence[XY], tolerance_m: float = 1.5):
        self.wps = list(waypoints)
        self.tol = tolerance_m
        self.i = 0

    @property
    def done(self) -> bool:
        return self.i >= len(self.wps)

    def update(self, x: float, y: float) -> Optional[XY]:
        while not self.done and math.hypot(self.wps[self.i][0] - x, self.wps[self.i][1] - y) <= self.tol:
            self.i += 1
        return None if self.done else self.wps[self.i]

    def progress(self) -> float:
        return self.i / len(self.wps) if self.wps else 1.0


def write_waypoints(path: str, waypoints: Sequence[XY], home_lat: float, home_lon: float,
                    altitude_m: float, home_alt_amsl: float = 0.0) -> int:
    """QGC WPL 110 file (Mission Planner / QGroundControl "load mission"): takeoff,
    the passes, return to launch. Altitudes relative to home. Returns the item count."""
    rows = [(0, 1, FRAME_GLOBAL, MAV_CMD_NAV_WAYPOINT, home_lat, home_lon, home_alt_amsl),
            (1, 0, FRAME_GLOBAL_RELATIVE_ALT, MAV_CMD_NAV_TAKEOFF, 0.0, 0.0, altitude_m)]
    for x, y in waypoints:
        lat, lon = local_to_latlon(x, y, home_lat, home_lon)
        rows.append((len(rows), 0, FRAME_GLOBAL_RELATIVE_ALT, MAV_CMD_NAV_WAYPOINT, lat, lon, altitude_m))
    rows.append((len(rows), 0, FRAME_GLOBAL_RELATIVE_ALT, MAV_CMD_NAV_RTL, 0.0, 0.0, 0.0))
    with open(path, "w") as f:
        f.write("QGC WPL 110\n")
        for i, cur, frame, cmd, lat, lon, alt in rows:
            f.write("\t".join([str(i), str(cur), str(frame), str(cmd), "0", "0", "0", "0",
                               f"{lat:.8f}", f"{lon:.8f}", f"{alt:.2f}", "1"]) + "\n")
    return len(rows)


def parse_latlon(text: str) -> Tuple[float, float]:
    lat, lon = (float(v) for v in text.replace(" ", "").split(","))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"not a latitude,longitude: {text!r}")
    return lat, lon
