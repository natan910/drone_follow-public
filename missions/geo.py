"""
Where things are on the ground, from where they are in the picture.

Frames (same as datatypes.Pose): x = metres EAST of the launch point, y = metres
NORTH, yaw clockwise from north. Ground is assumed flat, at the launch point's
height: fine for a field or a yard, wrong on a hillside (error grows with range).

    ground_point(u, v, w, h, pose, pitch_deg, hfov_deg, aspect)
        pixel (u, v) -> (x, y) on the ground, or None when the pixel looks at
        or above the horizon (or too far away to trust).

For a person box, use the middle of the box's BOTTOM edge (their feet).
perception/geometry.py has the follow-mode camera model; this is the small
flat-ground version the missions need.
"""

import json
import math
from typing import List, Optional, Sequence, Tuple

XY = Tuple[float, float]
EARTH_R = 6_371_000.0


def ground_point(u: float, v: float, width: int, height: int, pose, pitch_deg: float,
                 hfov_deg: float, aspect: Optional[float] = None, height_m: Optional[float] = None,
                 min_down_deg: float = 5.0, max_range_m: Optional[float] = None) -> Optional[XY]:
    """Pinhole camera, square pixels. aspect = frame height / width (None: from the image).
    pitch: 0 = level, 90 = straight down. height_m: camera height above the ground
    (default pose.z, height above the launch point)."""
    h_agl = pose.z if height_m is None else height_m
    if h_agl is None or h_agl < 0.2:
        return None
    a = aspect if aspect is not None else height / width
    tan_h = math.tan(math.radians(hfov_deg) / 2)
    xn = (u - width / 2) / (width / 2) * tan_h              # right of the optical axis
    yn = (v - height / 2) / (height / 2) * tan_h * a        # below the optical axis
    th = math.radians(pitch_deg)
    fwd = math.cos(th) - yn * math.sin(th)                  # ray in the level body frame
    down = math.sin(th) + yn * math.cos(th)
    if down <= 0 or math.degrees(math.atan2(down, math.hypot(fwd, xn))) < min_down_deg:
        return None
    t = h_agl / down
    f, r = t * fwd, t * xn
    if max_range_m is not None and math.hypot(f, r) > max_range_m:
        return None
    s, c = math.sin(pose.yaw), math.cos(pose.yaw)
    return pose.x + f * s + r * c, pose.y + f * c - r * s


def foot_of(box) -> Tuple[float, float]:
    """(l, t, r, b) -> middle of the bottom edge."""
    return (box[0] + box[2]) / 2, box[3]


def point_in_polygon(x: float, y: float, poly: Sequence[XY]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xc:
                inside = not inside
    return inside


def polygon_area(poly: Sequence[XY]) -> float:
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                   for i in range(len(poly)))) / 2


def local_to_latlon(x: float, y: float, home_lat: float, home_lon: float) -> Tuple[float, float]:
    """Metres east / north of home -> latitude, longitude (fine within a few km)."""
    lat = home_lat + math.degrees(y / EARTH_R)
    lon = home_lon + math.degrees(x / (EARTH_R * math.cos(math.radians(home_lat))))
    return lat, lon


def latlon_to_local(lat: float, lon: float, home_lat: float, home_lon: float) -> XY:
    y = math.radians(lat - home_lat) * EARTH_R
    x = math.radians(lon - home_lon) * EARTH_R * math.cos(math.radians(home_lat))
    return x, y


def file_home(path: str) -> Optional[Tuple[float, float]]:
    """The "home": [lat, lon] of a zones / field file, if it has one."""
    with open(path) as f:
        data = json.load(f)
    return tuple(float(v) for v in data["home"]) if isinstance(data, dict) and "home" in data else None


def load_polygons(path: str, home: Optional[Tuple[float, float]] = None) -> List[dict]:
    """A zones / field file:

        {"home": [0.0, 0.0],                      <- optional, only needed for "latlon"
         "areas": [{"name": "gate", "polygon": [[x, y], ...]},
                   {"name": "shed", "latlon": [[lat, lon], ...]}]}

    polygon = metres east / north of the launch point. latlon = from Google Maps
    (right-click -> the coordinates), converted with "home" (the launch spot).
    Returns [{"name", "polygon"}] in metres."""
    with open(path) as f:
        data = json.load(f)
    areas = data["areas"] if isinstance(data, dict) else data
    home = home or (tuple(data["home"]) if isinstance(data, dict) and "home" in data else None)
    out = []
    for i, a in enumerate(areas):
        if "polygon" in a:
            poly = [(float(p[0]), float(p[1])) for p in a["polygon"]]
        elif "latlon" in a:
            if home is None:
                raise ValueError(f"{path}: area {a.get('name', i)!r} uses latlon but the file has no \"home\"")
            poly = [latlon_to_local(float(p[0]), float(p[1]), *home) for p in a["latlon"]]
        else:
            raise ValueError(f"{path}: area {a.get('name', i)!r} needs \"polygon\" or \"latlon\"")
        if len(poly) < 3:
            raise ValueError(f"{path}: area {a.get('name', i)!r} needs at least 3 corners")
        out.append({"name": str(a.get("name", f"area{i}")), "polygon": poly})
    return out
