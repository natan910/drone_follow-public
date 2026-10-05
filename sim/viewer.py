"""Top-down picture of the toy world and what the autopilot has learned of it."""

import math
from typing import Optional

import cv2
import numpy as np

from autonomy.autopilot import Autopilot
from sim.virtual_world import VirtualWorld


def render(world: VirtualWorld, autopilot: Autopilot, bounds=(-12.0, -9.0, 12.0, 9.0),
           scale: int = 30) -> np.ndarray:
    xmin, ymin, xmax, ymax = bounds
    W, H = int((xmax - xmin) * scale), int((ymax - ymin) * scale)
    img = np.full((H, W, 3), 255, np.uint8)

    def px(x: float, y: float):
        return int((x - xmin) * scale), int((ymax - y) * scale)

    g = autopilot.grid
    # learned map: light green = seen by the camera recently or ever, grey = free, dark = occupied
    for iy in range(g.h):
        for ix in range(g.w):
            wx, wy = g.to_world((ix, iy))
            if not (xmin <= wx < xmax and ymin <= wy < ymax):
                continue
            x0, y0 = px(wx - g.res / 2, wy + g.res / 2)
            x1, y1 = px(wx + g.res / 2, wy - g.res / 2)
            if g.occupied[iy, ix]:
                colour = (40, 40, 200)
            elif math.isfinite(g.viewed_t[iy, ix]):
                colour = (215, 245, 215)
            elif g.free[iy, ix]:
                colour = (235, 235, 235)
            else:
                continue
            cv2.rectangle(img, (x0, y0), (x1, y1), colour, -1)

    for b in world.boxes:  # ground truth outlines
        cv2.rectangle(img, px(b.x0, b.y1), px(b.x1, b.y0), (60, 60, 60), 2)

    route = autopilot.planner._route
    if route is not None and route.path:
        pts = [px(world.x, world.y)] + [px(x, y) for x, y in route.path]
        cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 160, 255), 2)

    if world.person is not None:
        cv2.circle(img, px(*world.person), 7, (255, 80, 0), -1)

    cx, cy = px(world.x, world.y)  # drone + field of view
    for sign in (-1, 1):
        a = world.heading + sign * world.half_fov
        cv2.line(img, (cx, cy), px(world.x + 3 * math.sin(a), world.y + 3 * math.cos(a)), (0, 140, 0), 1)
    tip = px(world.x + 0.5 * math.sin(world.heading), world.y + 0.5 * math.cos(world.heading))
    cv2.circle(img, (cx, cy), 8, (0, 0, 0), -1)
    cv2.line(img, (cx, cy), tip, (0, 0, 0), 3)

    cv2.putText(img, f"t={world.t:5.0f}s  {autopilot.mode.name}  {autopilot.tracker.state.name}  "
                     f"collisions={world.collisions}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2)
    return img
