"""
Headroom map: the lowest ceiling seen above each spot, from an upward range
sensor (RangeScan.up). Indoors that is the room ceiling, a beam, a low lamp,
the underside of a staircase.

    observe(pose, up_m)       ceiling here = pose.z + up_m (metres above launch)
    ceiling_near(pose, r_m)   lowest remembered ceiling within r_m of the drone, or None

The autopilot (Autopilot._ceiling) turns that into a lower altitude limit, so
patrol descends to stay ceiling_clearance_m under it, on this pass and the next
(the avoider alone only reacts while the sensor is under the low part).

Limits, honestly:
  - one TF-Luna looking up sees a 2-degree spot above the drone's centre. The
    props stick out ~0.4 m (X500) and hit a lamp before the sensor is under it.
    The map remembers it for next time, it cannot see it coming the first time.
  - a reading only lowers a cell. A higher ceiling (something moved) replaces
    it after raise_after readings in a row agree.
Same grid as the occupancy map (launch-centred, x east / y north).
"""

import math
from typing import Optional

import numpy as np


class HeadroomMap:
    def __init__(self, size_m: float = 60.0, res_m: float = 0.5, min_up_m: float = 0.1,
                 max_up_m: float = 8.0, raise_after: int = 3, raise_by_m: float = 0.3):
        self.res = res_m
        self.n = int(round(size_m / res_m))
        self.origin = -size_m / 2
        self.min_up, self.max_up = min_up_m, max_up_m
        self.raise_after, self.raise_by = raise_after, raise_by_m
        self.ceiling = np.full((self.n, self.n), np.inf)      # metres above launch, [iy, ix]
        self._higher = np.zeros((self.n, self.n), np.int16)

    def _cell(self, x: float, y: float):
        ix, iy = int(math.floor((x - self.origin) / self.res)), int(math.floor((y - self.origin) / self.res))
        return (ix, iy) if 0 <= ix < self.n and 0 <= iy < self.n else None

    def observe(self, pose, up_m: Optional[float]) -> None:
        if up_m is None or not (self.min_up <= up_m <= self.max_up):
            return                                  # no reading, or the sensor's nonsense range
        c = self._cell(pose.x, pose.y)
        if c is None:
            return
        ix, iy = c
        z = pose.z + up_m
        cur = self.ceiling[iy, ix]
        if z < cur:
            self.ceiling[iy, ix] = z
            self._higher[iy, ix] = 0
        elif z > cur + self.raise_by:
            self._higher[iy, ix] += 1
            if self._higher[iy, ix] >= self.raise_after:
                self.ceiling[iy, ix] = z
                self._higher[iy, ix] = 0
        else:
            self._higher[iy, ix] = 0

    def ceiling_near(self, pose, radius_m: float) -> Optional[float]:
        r = max(0, int(math.ceil(radius_m / self.res)))
        c = self._cell(pose.x, pose.y)
        if c is None:
            return None
        ix, iy = c
        win = self.ceiling[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1]
        v = float(win.min()) if win.size else math.inf
        return v if math.isfinite(v) else None

    def known(self) -> int:
        return int(np.isfinite(self.ceiling).sum())
