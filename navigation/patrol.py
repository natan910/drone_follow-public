"""
Decides where to fly when there is no target to follow.

One rule covers both exploring and patrolling: go to the reachable place the
camera has gone longest without looking at. Space it has never covered counts
as maximally stale, so the drone explores first; once everything has been seen
it keeps sweeping toward whatever was seen least recently, forever. Distance is
penalised so it prefers nearby stale areas over far ones, and a hysteresis
margin stops it dithering between two similar goals.

Also handles: heading for a hint (where the target was last seen), going home,
and giving up on goals it cannot make progress toward.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import PatrolConfig
from datatypes import Mode, Pose
from mapping.occupancy_grid import Cell, OccupancyGrid
from navigation.planner import Plan, make_plan

XY = Tuple[float, float]


@dataclass
class Route:
    path: List[XY]      # waypoints from the drone toward the goal
    goal: XY
    kind: Mode          # EXPLORE, PATROL, SEARCH or RETURN
    arrived: bool = False


class PatrolPlanner:
    def __init__(self, config: Optional[PatrolConfig] = None):
        self.cfg = config or PatrolConfig()
        self._route: Optional[Route] = None
        self._goal_cell: Optional[Cell] = None
        self._last_plan = -math.inf
        self._blacklist: Dict[Cell, float] = {}
        self._anchor: Optional[Tuple[float, float, float]] = None  # x, y, time

    def reset(self) -> None:
        self._route = self._goal_cell = self._anchor = None
        self._last_plan = -math.inf

    # ---- public API -------------------------------------------------------
    def update(self, grid: OccupancyGrid, pose: Pose, now: float,
               hint: Optional[XY] = None) -> Optional[Route]:
        """Route to follow this step, or None if there is nowhere sensible to go."""
        self._check_progress(pose, now)
        if self._needs_replan(pose, now, hint):
            self._replan(grid, pose, now, hint)
        elif self._route is not None:
            self._route.arrived = self._is_arrived(pose, self._route.goal)
        return self._route

    def route_home(self, grid: OccupancyGrid, pose: Pose, now: float,
                   home: XY = (0.0, 0.0)) -> Optional[Route]:
        if self._route is None or self._route.kind != Mode.RETURN \
                or now - self._last_plan >= self.cfg.replan_interval_s:
            plan = make_plan(grid, (pose.x, pose.y), self.cfg.unknown_cost)
            goal_cell = grid.to_cell(*home)
            if plan is None or not plan.reachable(goal_cell):
                # No known way home: head straight there, the avoider protects us.
                self._route = Route([home], home, Mode.RETURN)
            else:
                self._route = Route(plan.path_to(goal_cell) or [home], home, Mode.RETURN)
            self._last_plan = now
        self._route.arrived = self._is_arrived(pose, home)
        return self._route

    # ---- internals --------------------------------------------------------
    def _is_arrived(self, pose: Pose, goal: XY) -> bool:
        return math.hypot(goal[0] - pose.x, goal[1] - pose.y) <= self.cfg.goal_tolerance_m

    def _needs_replan(self, pose: Pose, now: float, hint: Optional[XY]) -> bool:
        r = self._route
        if r is None or r.kind == Mode.RETURN:
            return True
        if (r.kind == Mode.SEARCH) != (hint is not None):
            return True  # switched between searching and exploring
        if self._is_arrived(pose, r.goal):
            return True
        return now - self._last_plan >= self.cfg.replan_interval_s

    def _check_progress(self, pose: Pose, now: float) -> None:
        """If we have barely moved for a while, the current goal is not working."""
        c = self.cfg
        if self._route is None or self._route.arrived:
            self._anchor = None
            return
        if self._anchor is None:
            self._anchor = (pose.x, pose.y, now)
            return
        ax, ay, at = self._anchor
        if math.hypot(pose.x - ax, pose.y - ay) >= c.stuck_distance_m:
            self._anchor = (pose.x, pose.y, now)
        elif now - at >= c.stuck_window_s:
            if self._goal_cell is not None:
                self._blacklist[self._goal_cell] = now + c.blacklist_s
            self._route = None
            self._anchor = None

    def _replan(self, grid: OccupancyGrid, pose: Pose, now: float,
                hint: Optional[XY]) -> None:
        c = self.cfg
        self._last_plan = now
        region = grid.region_mask(c.patrol_radius_m)
        plan = make_plan(grid, (pose.x, pose.y), c.unknown_cost, region)
        if plan is None:
            self._route = None
            return

        if hint is not None:
            goal_cell = self._nearest_reachable(plan, grid.to_cell(*hint))
            kind = Mode.SEARCH
        else:
            goal_cell, kind = self._best_stale_goal(plan, grid, pose, now)
        if goal_cell is None:
            self._route, self._goal_cell = None, None
            return

        self._goal_cell = goal_cell
        goal = grid.to_world(goal_cell)
        self._route = Route(plan.path_to(goal_cell) or [goal], goal, kind,
                            arrived=self._is_arrived(pose, goal))

    def _nearest_reachable(self, plan: Plan, cell: Cell) -> Optional[Cell]:
        if plan.reachable(cell):
            return cell
        finite = np.isfinite(plan.dist)
        if not finite.any():
            return None
        ys, xs = np.nonzero(finite)
        i = int(np.argmin((xs - cell[0]) ** 2 + (ys - cell[1]) ** 2))
        return int(xs[i]), int(ys[i])

    def _scores(self, plan: Plan, pose: Pose, now: float) -> np.ndarray:
        c = self.cfg
        stale = np.clip(now - plan.grid.viewed_t, 0.0, c.staleness_cap_s)
        score = stale - c.distance_weight_s_per_m * plan.dist
        score[~np.isfinite(plan.dist)] = -np.inf
        # not right here (that would count as already arrived), and not somewhere we gave up on
        g = plan.grid
        score[np.hypot(g.cx - pose.x, g.cy - pose.y) < 1.5 * c.goal_tolerance_m] = -np.inf
        for cell, until in list(self._blacklist.items()):
            if until < now:
                del self._blacklist[cell]
            else:
                score[cell[1], cell[0]] = -np.inf
        return score

    def _best_stale_goal(self, plan: Plan, grid: OccupancyGrid, pose: Pose,
                         now: float) -> Tuple[Optional[Cell], Mode]:
        score = self._scores(plan, pose, now)
        if not np.isfinite(score).any():
            return None, Mode.HOLD
        iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
        best = (int(ix), int(iy))

        cur = self._goal_cell
        if cur is not None and np.isfinite(score[cur[1], cur[0]]) and cur != best:
            if score[best[1], best[0]] < score[cur[1], cur[0]] + self.cfg.switch_margin_s:
                best = cur  # not clearly better: keep going where we were going

        never_viewed = not math.isfinite(grid.viewed_t[best[1], best[0]])
        return best, (Mode.EXPLORE if never_viewed else Mode.PATROL)
