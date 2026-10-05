"""
How green is it: vegetation indices from a camera frame, and a map of them.

    vari(frame)           normal colour camera (the Camera Module 3 Wide you bought).
                          VARI = (G - R) / (G + R - B). Rough, but needs no extra kit.
    ndvi_bluefilter(frame)  NoIR camera (no infrared filter) + a blue filter
                          (e.g. Rosco 2007 "Storaro blue") over the lens: the red
                          channel then records near-infrared, the blue channel
                          visible light. NDVI = (NIR - VIS) / (NIR + VIS).
                          The classic cheap-NDVI trick (Public Lab). Better than VARI
                          at telling healthy from stressed plants.

Both are RELATIVE: compare the same field, same camera, same time of day,
week to week. Not comparable with satellite NDVI numbers. Clouds and low sun
change them.

FieldMap: mean index per 2 m cell, when it was last surveyed, saved to JSON so
maps grow across flights (launch from the same spot: positions are
launch-relative).
"""

import json
import math
import os
import time
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from missions.geo import XY, ground_point, point_in_polygon
from missions.mission_config import SurveyConfig


def _channels(frame: np.ndarray):
    f = frame.astype(np.float32)
    return f[..., 0], f[..., 1], f[..., 2]           # OpenCV frames are BGR


def vari(frame: np.ndarray) -> np.ndarray:
    b, g, r = _channels(frame)
    den = g + r - b
    out = np.where(np.abs(den) > 1.0, (g - r) / np.where(np.abs(den) > 1.0, den, 1.0), 0.0)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def ndvi_bluefilter(frame: np.ndarray) -> np.ndarray:
    b, _, r = _channels(frame)
    den = r + b
    out = np.where(den > 1.0, (r - b) / np.where(den > 1.0, den, 1.0), 0.0)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


INDEXES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {"vari": vari, "ndvi": ndvi_bluefilter}


def colorize(index: np.ndarray, lo: float = -0.3, hi: float = 0.6) -> np.ndarray:
    """Index -> BGR picture: red (bare / stressed) -> yellow -> green (vigorous)."""
    t = np.clip((index - lo) / (hi - lo), 0.0, 1.0)
    r = np.where(t < 0.5, 255.0, 255.0 * (1 - t) * 2)
    g = np.where(t < 0.5, 255.0 * t * 2, 255.0)
    return np.stack([np.zeros_like(t), g, r], axis=-1).astype(np.uint8)


def summary(index: np.ndarray, green_above: float = 0.05) -> dict:
    v = index.ravel()
    return {"mean": round(float(v.mean()), 3), "p10": round(float(np.percentile(v, 10)), 3),
            "p90": round(float(np.percentile(v, 90)), 3), "green_frac": round(float((v > green_above).mean()), 3)}


class FieldMap:
    """Mean index per cell + when each cell was last surveyed (wall clock, so it survives restarts)."""

    def __init__(self, cell_m: float = 2.0, index_name: str = "vari"):
        self.cell_m = cell_m
        self.index_name = index_name
        self.cells: Dict[Tuple[int, int], list] = {}      # (i, j) -> [sum, n, last_wall_time]

    def key(self, x: float, y: float) -> Tuple[int, int]:
        return int(math.floor(x / self.cell_m)), int(math.floor(y / self.cell_m))

    def add(self, x: float, y: float, value: float, wall_time: float) -> None:
        c = self.cells.setdefault(self.key(x, y), [0.0, 0, 0.0])
        c[0] += float(value)
        c[1] += 1
        c[2] = max(c[2], wall_time)

    def mean_at(self, x: float, y: float) -> Optional[float]:
        c = self.cells.get(self.key(x, y))
        return None if c is None else c[0] / c[1]

    def coverage(self, poly: Sequence[XY]) -> float:
        """Share of the field's cells that have at least one reading."""
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        total = hit = 0
        for i in range(int(math.floor(min(xs) / self.cell_m)), int(math.floor(max(xs) / self.cell_m)) + 1):
            for j in range(int(math.floor(min(ys) / self.cell_m)), int(math.floor(max(ys) / self.cell_m)) + 1):
                if point_in_polygon((i + 0.5) * self.cell_m, (j + 0.5) * self.cell_m, poly):
                    total += 1
                    hit += (i, j) in self.cells
        return hit / total if total else 0.0

    def save(self, path: str) -> None:
        data = {"cell_m": self.cell_m, "index": self.index_name,
                "cells": [[i, j, round(s, 4), n, round(t, 1)] for (i, j), (s, n, t) in self.cells.items()]}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)                            # a crash mid-save can't corrupt the old map

    @classmethod
    def load(cls, path: str) -> "FieldMap":
        with open(path) as f:
            data = json.load(f)
        m = cls(data["cell_m"], data.get("index", "vari"))
        for i, j, s, n, t in data["cells"]:
            m.cells[(int(i), int(j))] = [float(s), int(n), float(t)]
        return m

    def to_csv(self, path: str) -> None:
        """One row per cell (x, y = cell centre, metres): open in a spreadsheet, or plot."""
        with open(path, "w") as f:
            f.write(f"x_m,y_m,{self.index_name},readings,last_surveyed\n")
            for (i, j), (s, n, t) in sorted(self.cells.items()):
                stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
                f.write(f"{(i + 0.5) * self.cell_m:.1f},{(j + 0.5) * self.cell_m:.1f},{s / n:.4f},{n},{stamp}\n")


class SurveyLogger:
    """Main-loop side: every `every_s`, score the middle of the frame and drop it into the map
    at the ground point the camera centre looks at."""

    def __init__(self, hfov_deg: float, aspect: float, config: Optional[SurveyConfig] = None,
                 field_map: Optional[FieldMap] = None, polygon: Optional[Sequence[XY]] = None,
                 wall_clock: Callable[[], float] = time.time):
        self.cfg = config or SurveyConfig()
        if self.cfg.index not in INDEXES:
            raise ValueError(f"unknown index {self.cfg.index!r}: {sorted(INDEXES)}")
        self.index = INDEXES[self.cfg.index]
        self.map = field_map or FieldMap(self.cfg.cell_m, self.cfg.index)
        self.polygon = polygon
        self.hfov, self.aspect = hfov_deg, aspect
        self.wall_clock = wall_clock
        self._last = -float("inf")
        self.readings = 0
        self.last: Optional[dict] = None

    def step(self, frame: Optional[np.ndarray], obs) -> Optional[dict]:
        c = self.cfg
        if frame is None or obs.now - self._last < c.every_s:
            return None
        if obs.camera_pitch_deg < c.min_pitch_deg or obs.pose.z < c.min_alt_m:
            return None
        self._last = obs.now
        import cv2
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (c.analysis_width, max(1, int(h * c.analysis_width / w))),
                           interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        dh, dw = int(sh * c.center_frac / 2), int(sw * c.center_frac / 2)
        centre = small[sh // 2 - dh: sh // 2 + dh + 1, sw // 2 - dw: sw // 2 + dw + 1]
        idx = self.index(centre)
        p = ground_point(w / 2, h / 2, w, h, obs.pose, obs.camera_pitch_deg, self.hfov, self.aspect,
                         min_down_deg=30.0)
        if p is None:
            return None
        s = summary(idx, c.green_above)
        self.map.add(p[0], p[1], s["mean"], self.wall_clock())
        self.readings += 1
        self.last = {"x": round(p[0], 1), "y": round(p[1], 1), **s}
        return self.last

    def status(self) -> dict:
        st = {"mission": "survey", "index": self.cfg.index, "readings": self.readings,
              "cells": len(self.map.cells), "last": self.last}
        if self.polygon:
            st["coverage_pct"] = round(100 * self.map.coverage(self.polygon), 1)
        return st
