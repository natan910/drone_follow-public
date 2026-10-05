"""
The drone's memory of the area: a 2-D occupancy grid plus a "last viewed" clock.

    logodds   evidence that each cell is blocked (+) or free (-); 0 = never seen
    viewed_t  the last time the camera could have recognised a face in each cell

Together they answer the two questions autonomy needs: "where is it safe to
fly?" and "where haven't I looked lately?". The map can be saved and reloaded,
so the drone starts its next flight already knowing the place.

Frame: x east, y north, metres, origin = home. Arrays are indexed [iy, ix].
"""

import math
from typing import Iterator, Optional, Tuple

import numpy as np

from config import MapConfig
from datatypes import Pose, RangeScan, wrap_angle

Cell = Tuple[int, int]


def line_cells(a: Cell, b: Cell) -> Iterator[Cell]:
    """Bresenham line from cell a to cell b, inclusive."""
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        yield x0, y0
        if (x0, y0) == (x1, y1):
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


class OccupancyGrid:
    def __init__(self, config: Optional[MapConfig] = None):
        self.cfg = config or MapConfig()
        self.res = self.cfg.resolution_m
        self.w = self.h = int(round(self.cfg.size_m / self.res))
        self.origin = -self.cfg.size_m / 2  # world coord of the grid's south-west corner
        self.logodds = np.zeros((self.h, self.w), dtype=np.float32)
        self.viewed_t = np.full((self.h, self.w), -np.inf)
        idx_y, idx_x = np.mgrid[0:self.h, 0:self.w]
        self.cx = self.origin + (idx_x + 0.5) * self.res  # world x of every cell centre
        self.cy = self.origin + (idx_y + 0.5) * self.res  # world y of every cell centre

    # ---- coordinates ------------------------------------------------------
    def to_cell(self, x: float, y: float) -> Cell:
        return (int(math.floor((x - self.origin) / self.res)),
                int(math.floor((y - self.origin) / self.res)))

    def to_world(self, cell: Cell) -> Tuple[float, float]:
        return (self.origin + (cell[0] + 0.5) * self.res,
                self.origin + (cell[1] + 0.5) * self.res)

    def inside(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.w and 0 <= cell[1] < self.h

    # ---- views of the evidence -------------------------------------------
    @property
    def free(self) -> np.ndarray:
        return self.logodds < self.cfg.free_below

    @property
    def occupied(self) -> np.ndarray:
        return self.logodds > self.cfg.occupied_above

    @property
    def unknown(self) -> np.ndarray:
        return ~self.free & ~self.occupied

    def inflated_blocked(self, radius_m: Optional[float] = None) -> np.ndarray:
        """Occupied cells grown by the drone's radius plus a margin: any cell
        whose centre is inside this is unsafe to fly through."""
        if radius_m is None:
            radius_m = self.cfg.drone_radius_m + self.cfg.safety_margin_m
        r = int(math.ceil(radius_m / self.res))
        blocked = np.zeros((self.h, self.w), dtype=bool)
        iy, ix = np.nonzero(self.occupied)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                yy, xx = iy + dy, ix + dx
                ok = (yy >= 0) & (yy < self.h) & (xx >= 0) & (xx < self.w)
                blocked[yy[ok], xx[ok]] = True
        return blocked

    def known_fraction(self, region: Optional[np.ndarray] = None) -> float:
        known = ~self.unknown
        return float(known.mean() if region is None else known[region].mean())

    # ---- updates ----------------------------------------------------------
    def _add(self, cell: Cell, delta: float) -> None:
        if self.inside(cell):
            v = self.logodds[cell[1], cell[0]] + delta
            self.logodds[cell[1], cell[0]] = min(self.cfg.l_max, max(self.cfg.l_min, v))

    def integrate(self, pose: Pose, scan: RangeScan) -> None:
        """Ray-cast each beam: cells it passes through are free, the cell where
        it ends (if it hit something) is occupied."""
        c = self.cfg
        origin = self.to_cell(pose.x, pose.y)
        for beam in scan.beams:
            hit = beam.distance is not None
            d = beam.distance if hit else scan.max_range
            a = pose.yaw + beam.bearing
            end = self.to_cell(pose.x + d * math.sin(a), pose.y + d * math.cos(a))
            cells = list(line_cells(origin, end))
            for cell in cells[:-1]:
                self._add(cell, -c.l_free)
            self._add(cells[-1], c.l_occ if hit else -c.l_free)

    def mark_viewed(self, pose: Pose, fov: float, max_dist: float, now: float) -> None:
        """Stamp every cell inside the camera's view cone (and a small disc
        around the drone) with the current time."""
        r = int(math.ceil(max_dist / self.res))
        cx, cy = self.to_cell(pose.x, pose.y)
        x0, x1 = max(cx - r, 0), min(cx + r + 1, self.w)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, self.h)
        if x0 >= x1 or y0 >= y1:
            return
        xs = self.origin + (np.arange(x0, x1) + 0.5) * self.res - pose.x
        ys = self.origin + (np.arange(y0, y1) + 0.5) * self.res - pose.y
        dx, dy = np.meshgrid(xs, ys)
        dist = np.hypot(dx, dy)
        bearing = (np.arctan2(dx, dy) - pose.yaw + math.pi) % (2 * math.pi) - math.pi
        seen = ((dist <= max_dist) & (np.abs(bearing) <= fov / 2)) | (dist <= 1.0)
        self.viewed_t[y0:y1, x0:x1][seen] = now

    def region_mask(self, radius_m: float) -> np.ndarray:
        """Cells within radius_m of home."""
        return np.hypot(self.cx, self.cy) <= radius_m

    # ---- persistence ------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez_compressed(path, logodds=self.logodds, viewed_t=self.viewed_t,
                            res=self.res, size=self.cfg.size_m)

    def load(self, path: str, keep_viewed: bool = False) -> None:
        """Load a saved map. Viewed times are dropped by default: they came
        from another flight's clock, so everything counts as stale again."""
        data = np.load(path)
        if data["logodds"].shape != self.logodds.shape or float(data["res"]) != self.res:
            raise ValueError("saved map has a different size or resolution")
        self.logodds = data["logodds"].astype(np.float32)
        if keep_viewed:
            self.viewed_t = data["viewed_t"]
