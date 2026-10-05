"""
Path planning over an OccupancyGrid.

One Dijkstra run from the drone's cell gives the cost to reach every cell and
a parent pointer to walk back from any of them, so the patrol planner can score
thousands of possible goals for the price of a single search.

Costs: free space 1.0, never-seen space `unknown_cost` per metre (we will fly
into it cautiously, and the reactive avoider protects us), blocked space
(occupied cells grown by the drone's size) is impassable.
"""

import heapq
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from mapping.occupancy_grid import Cell, OccupancyGrid

SQRT2 = math.sqrt(2.0)
_NEIGHBOURS = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2))


def build_cost_grid(grid: OccupancyGrid, unknown_cost: float,
                    region: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-cell cost multiplier; inf = impassable."""
    cost = np.where(grid.free, 1.0, unknown_cost).astype(np.float64)
    cost[grid.inflated_blocked()] = np.inf
    if region is not None:
        cost[~region] = np.inf
    return cost


@dataclass
class Plan:
    grid: OccupancyGrid
    start: Cell
    dist: np.ndarray    # cost-distance from start, inf where unreachable
    parent: np.ndarray  # flat index of the previous cell, -1 at the start / unreachable

    def reachable(self, cell: Cell) -> bool:
        return self.grid.inside(cell) and math.isfinite(self.dist[cell[1], cell[0]])

    def path_to(self, cell: Cell) -> List[Tuple[float, float]]:
        """World-frame waypoints from just after the start up to `cell`."""
        w = self.grid.w
        idx = cell[1] * w + cell[0]
        out: List[Tuple[float, float]] = []
        while idx != -1 and idx != self.start[1] * w + self.start[0]:
            out.append(self.grid.to_world((idx % w, idx // w)))
            idx = int(self.parent[idx // w, idx % w])
        out.reverse()
        return out


def make_plan(grid: OccupancyGrid, start_xy: Tuple[float, float], unknown_cost: float,
              region: Optional[np.ndarray] = None, max_cost: float = 400.0) -> Optional[Plan]:
    start = grid.to_cell(*start_xy)
    if not grid.inside(start):
        return None

    cost = build_cost_grid(grid, unknown_cost, region)
    cost[start[1], start[0]] = 1.0  # we are here, so it must be passable
    h, w = cost.shape
    flat = cost.ravel().tolist()
    dist = [math.inf] * (h * w)
    parent = [-1] * (h * w)
    s = start[1] * w + start[0]
    dist[s] = 0.0
    heap = [(0.0, s)]
    res = grid.res

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if d > max_cost:
            break
        uy, ux = divmod(u, w)
        for dx, dy, step in _NEIGHBOURS:
            vx, vy = ux + dx, uy + dy
            if not (0 <= vx < w and 0 <= vy < h):
                continue
            c = flat[vy * w + vx]
            if c == math.inf:
                continue
            if dx and dy and (flat[uy * w + vx] == math.inf or flat[vy * w + ux] == math.inf):
                continue  # no cutting corners through walls
            nd = d + step * res * c
            v = vy * w + vx
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))

    return Plan(grid, start, np.array(dist).reshape(h, w), np.array(parent).reshape(h, w))
