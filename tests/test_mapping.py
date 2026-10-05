import math
import os
import tempfile
import unittest

import numpy as np

from datatypes import Pose, RangeBeam, RangeScan
from mapping.occupancy_grid import OccupancyGrid, line_cells


class GridTests(unittest.TestCase):
    def setUp(self):
        self.g = OccupancyGrid()

    def cell_state(self, x, y):
        ix, iy = self.g.to_cell(x, y)
        return ("occupied" if self.g.occupied[iy, ix] else "free" if self.g.free[iy, ix] else "unknown")

    def test_line_cells_includes_both_ends(self):
        cells = list(line_cells((0, 0), (5, 2)))
        self.assertEqual(cells[0], (0, 0))
        self.assertEqual(cells[-1], (5, 2))

    def test_world_cell_round_trip(self):
        cell = self.g.to_cell(3.2, -7.9)
        x, y = self.g.to_world(cell)
        self.assertEqual(self.g.to_cell(x, y), cell)

    def test_beam_carves_free_space_and_marks_the_hit(self):
        self.g.integrate(Pose(0, 0, 0), RangeScan((RangeBeam(0.0, 3.0),), 4.0))
        self.assertEqual(self.cell_state(0, 1.0), "free")
        self.assertEqual(self.cell_state(0, 3.1), "occupied")
        self.assertEqual(self.cell_state(0, 3.9), "unknown")   # beyond the hit is not free

    def test_no_return_means_free_all_the_way_and_no_obstacle(self):
        self.g.integrate(Pose(0, 0, 0), RangeScan((RangeBeam(0.0, None),), 4.0))
        self.assertEqual(self.cell_state(0, 3.5), "free")
        self.assertEqual(int(self.g.occupied.sum()), 0)

    def test_yaw_rotates_the_beam(self):
        self.g.integrate(Pose(0, 0, math.pi / 2), RangeScan((RangeBeam(0.0, 3.0),), 4.0))  # facing east
        self.assertEqual(self.cell_state(3.1, 0), "occupied")

    def test_repeated_free_observations_clear_a_false_hit(self):
        self.g.integrate(Pose(0, 0, 0), RangeScan((RangeBeam(0.0, 3.0),), 4.0))
        for _ in range(6):
            self.g.integrate(Pose(0, 0, 0), RangeScan((RangeBeam(0.0, None),), 4.0))
        self.assertEqual(self.cell_state(0, 3.1), "free")

    def test_inflation_grows_obstacles_by_the_drone_size(self):
        self.g.integrate(Pose(0, 0, 0), RangeScan((RangeBeam(0.0, 3.0),), 4.0))
        blocked = self.g.inflated_blocked()
        ix, iy = self.g.to_cell(0, 2.3)      # 0.8 m short of the obstacle: inside the halo
        self.assertTrue(blocked[iy, ix])
        ix, iy = self.g.to_cell(0, 1.0)      # 2 m short: fine
        self.assertFalse(blocked[iy, ix])

    def test_mark_viewed_stamps_only_the_view_cone(self):
        self.g.mark_viewed(Pose(0, 0, 0), math.radians(60), 4.0, now=7.0)
        ahead = self.g.to_cell(0, 3.0)
        behind = self.g.to_cell(0, -3.0)
        beyond = self.g.to_cell(0, 5.5)
        self.assertEqual(self.g.viewed_t[ahead[1], ahead[0]], 7.0)
        self.assertFalse(np.isfinite(self.g.viewed_t[behind[1], behind[0]]))
        self.assertFalse(np.isfinite(self.g.viewed_t[beyond[1], beyond[0]]))

    def test_save_and_load_round_trip(self):
        self.g.integrate(Pose(0, 0, 0), RangeScan((RangeBeam(0.0, 3.0),), 4.0))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            self.g.save(path)
            other = OccupancyGrid()
            other.load(path)
        self.assertTrue(np.array_equal(self.g.logodds, other.logodds))
        self.assertFalse(np.isfinite(other.viewed_t).any())  # other flight's clock: dropped

    def test_load_rejects_a_map_of_a_different_size(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            self.g.save(path)
            from config import MapConfig
            with self.assertRaises(ValueError):
                OccupancyGrid(MapConfig(size_m=40)).load(path)

    def test_range_scan_clearance_handles_blind_spots_and_wraparound(self):
        scan = RangeScan((RangeBeam(math.radians(179), 2.0), RangeBeam(0.0, None)), 4.0)
        self.assertEqual(scan.clearance_at(math.pi, 0.3), 2.0)       # behind: wraps around +-180
        self.assertEqual(scan.clearance_at(0.0, 0.3), 4.0)           # nothing seen = max range
        self.assertIsNone(scan.clearance_at(math.pi / 2, 0.3))       # no beam there: blind


if __name__ == "__main__":
    unittest.main()
