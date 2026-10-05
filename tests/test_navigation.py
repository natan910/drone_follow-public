import math
import unittest

from config import AvoidConfig, MapConfig, PatrolConfig
from datatypes import DriveCommand, Mode, Pose, RangeBeam, RangeScan
from mapping.occupancy_grid import OccupancyGrid
from navigation.avoidance import ObstacleAvoider
from navigation.follower import WaypointFollower
from navigation.patrol import PatrolPlanner
from navigation.planner import make_plan


def open_grid(size=20.0):
    g = OccupancyGrid(MapConfig(size_m=size))
    g.logodds[:] = -1.0  # everything known free
    return g


def block(g, x0, y0, x1, y1):
    for iy in range(g.h):
        for ix in range(g.w):
            wx, wy = g.to_world((ix, iy))
            if x0 <= wx <= x1 and y0 <= wy <= y1:
                g.logodds[iy, ix] = 3.0


class PlannerTests(unittest.TestCase):
    def test_path_goes_around_a_wall(self):
        g = open_grid()
        block(g, -0.5, -9.9, 0.5, 5.0)
        plan = make_plan(g, (-4, 0), 2.0)
        goal = g.to_cell(4, 0)
        self.assertTrue(plan.reachable(goal))
        path = plan.path_to(goal)
        self.assertGreater(sum(math.dist(a, b) for a, b in zip(path, path[1:])), 12.0)  # not the 8 m straight line
        self.assertTrue(all(abs(x) > 1.0 or y > 5.0 for x, y in path))                 # never through the wall

    def test_fully_walled_goal_is_unreachable(self):
        g = open_grid()
        block(g, -0.5, -9.9, 0.5, 9.9)
        self.assertFalse(make_plan(g, (-4, 0), 2.0).reachable(g.to_cell(4, 0)))

    def test_unknown_space_is_passable_but_costs_more(self):
        known, unknown = open_grid(), OccupancyGrid(MapConfig(size_m=20.0))
        cost_known = make_plan(known, (0, 0), 2.0).dist[known.to_cell(5, 0)[1], known.to_cell(5, 0)[0]]
        cost_unknown = make_plan(unknown, (0, 0), 2.0).dist[unknown.to_cell(5, 0)[1], unknown.to_cell(5, 0)[0]]
        self.assertTrue(math.isfinite(cost_unknown))
        self.assertGreater(cost_unknown, 1.5 * cost_known)


class PatrolTests(unittest.TestCase):
    def test_explores_space_never_viewed_first(self):
        route = PatrolPlanner().update(open_grid(), Pose(0, 0, 0), now=10.0)
        self.assertEqual(route.kind, Mode.EXPLORE)

    def test_once_everything_is_seen_it_patrols_the_stalest_place(self):
        g = open_grid()
        g.viewed_t[:] = 90.0
        ix, iy = g.to_cell(-6, 5)
        g.viewed_t[iy - 2:iy + 3, ix - 2:ix + 3] = 5.0   # one patch nobody has looked at for ages
        route = PatrolPlanner().update(g, Pose(0, 0, 0), now=100.0)
        self.assertEqual(route.kind, Mode.PATROL)
        self.assertLess(math.dist(route.goal, (-6, 5)), 3.0)

    def test_a_hint_sends_it_to_where_the_target_was_last_seen(self):
        route = PatrolPlanner().update(open_grid(), Pose(0, 0, 0), now=1.0, hint=(6.0, 3.0))
        self.assertEqual(route.kind, Mode.SEARCH)
        self.assertLess(math.dist(route.goal, (6, 3)), 1.0)

    def test_gives_up_on_a_goal_it_cannot_make_progress_toward(self):
        p, g = PatrolPlanner(PatrolConfig(stuck_window_s=5.0)), open_grid()
        pose = Pose(0, 0, 0)
        first = p.update(g, pose, 0.0).goal
        p.update(g, pose, 1.0)                        # the clock for "no progress" starts here
        self.assertEqual(p.update(g, pose, 4.0).goal, first)   # only 3 s: keep trying
        self.assertNotEqual(p.update(g, pose, 7.0).goal, first)  # 6 s without moving: pick another

    def test_route_home_ends_at_home(self):
        route = PatrolPlanner().route_home(open_grid(), Pose(6, 4, 0), now=0.0)
        self.assertEqual(route.kind, Mode.RETURN)
        self.assertLess(math.dist(route.path[-1], (0, 0)), 0.6)
        self.assertFalse(route.arrived)
        self.assertTrue(PatrolPlanner().route_home(open_grid(), Pose(0.2, 0.1, 0), 0.0).arrived)


class FollowerTests(unittest.TestCase):
    def setUp(self):
        self.f = WaypointFollower()

    def test_straight_ahead_drives_forward_without_turning(self):
        cmd = self.f.command(Pose(0, 0, 0), [(0, 3), (0, 6), (0, 9)])
        self.assertAlmostEqual(cmd.yaw_rate_dps, 0.0, places=3)
        self.assertGreater(cmd.forward_mps, 0.4)

    def test_turns_in_place_when_badly_misaligned(self):
        cmd = self.f.command(Pose(0, 0, 0), [(5, 0)])   # due east while facing north
        self.assertGreater(cmd.yaw_rate_dps, 0)
        self.assertEqual(cmd.forward_mps, 0.0)

    def test_turns_left_for_a_waypoint_on_the_left(self):
        self.assertLess(self.f.command(Pose(0, 0, 0), [(-5, 0)]).yaw_rate_dps, 0)

    def test_slows_down_near_the_end_of_the_path(self):
        far = self.f.command(Pose(0, 0, 0), [(0, 6)]).forward_mps
        near = self.f.command(Pose(0, 0, 0), [(0, 0.6)]).forward_mps
        self.assertLess(near, far)

    def test_empty_path_means_stand_still(self):
        self.assertTrue(self.f.command(Pose(0, 0, 0), []).is_zero)


def scan(front=None, l45=None, r45=None, l90=None, r90=None, rear=None, max_range=4.0):
    beams = [(0, front), (-45, l45), (45, r45), (-90, l90), (90, r90), (180, rear)]
    return RangeScan(tuple(RangeBeam(math.radians(b), d) for b, d in beams), max_range)


class AvoiderTests(unittest.TestCase):
    def setUp(self):
        self.a, self.c = ObstacleAvoider(), AvoidConfig()

    def test_open_space_leaves_the_command_alone(self):
        cmd = DriveCommand(10.0, 0.6)
        self.assertEqual(self.a.filter(cmd, scan()), cmd)

    def test_slows_down_as_something_gets_closer(self):
        half = self.a.filter(DriveCommand(0, 0.6), scan(front=2.0)).forward_mps
        self.assertAlmostEqual(half, 0.3, places=2)
        self.assertEqual(self.a.filter(DriveCommand(0, 0.6), scan(front=self.c.stop_distance_m)).forward_mps, 0.0)

    def test_no_scan_passes_through_for_the_supervisor_to_judge(self):
        cmd = DriveCommand(5.0, 0.6)
        self.assertEqual(self.a.filter(cmd, None), cmd)

    def test_blind_ahead_means_no_forward_motion(self):
        sideways_only = RangeScan((RangeBeam(math.radians(90), None), RangeBeam(math.radians(-90), None)), 4.0)
        self.assertEqual(self.a.filter(DriveCommand(0, 0.6), sideways_only).forward_mps, 0.0)

    def test_reversing_needs_a_clear_rear_sensor(self):
        no_rear = RangeScan((RangeBeam(0.0, None),), 4.0)
        self.assertEqual(self.a.filter(DriveCommand(0, -0.4), no_rear).forward_mps, 0.0)
        self.assertEqual(self.a.filter(DriveCommand(0, -0.4), scan(rear=0.5)).forward_mps, 0.0)
        self.assertEqual(self.a.filter(DriveCommand(0, -0.4), scan(rear=3.0)).forward_mps, -0.4)

    def test_steers_away_from_the_closer_side_while_driving(self):
        toward_right_obstacle = self.a.filter(DriveCommand(0, 0.6), scan(r45=1.5))
        toward_left_obstacle = self.a.filter(DriveCommand(0, 0.6), scan(l45=1.5))
        self.assertLess(toward_right_obstacle.yaw_rate_dps, 0)   # obstacle right -> turn left
        self.assertGreater(toward_left_obstacle.yaw_rate_dps, 0)

    def test_never_fights_a_turn_in_place(self):
        cmd = DriveCommand(-40.0, 0.0)
        self.assertEqual(self.a.filter(cmd, scan(front=1.2, l45=1.5, r45=1.5)), cmd)

    def test_head_on_picks_the_more_open_side_and_commits(self):
        first = self.a.filter(DriveCommand(0, 0.6), scan(front=1.4, l45=2.0, r45=2.0, l90=3.2, r90=3.8))
        self.assertGreater(first.yaw_rate_dps, 0)                # right side has more room
        flipped = scan(front=1.4, l45=2.0, r45=2.0, l90=3.8, r90=3.2)
        second = self.a.filter(DriveCommand(0, 0.6), flipped)    # readings flip, decision holds
        self.assertGreater(second.yaw_rate_dps, 0)
        self.a.filter(DriveCommand(0, 0.6), scan())              # way ahead clear: commitment released
        third = self.a.filter(DriveCommand(0, 0.6), flipped)
        self.assertLess(third.yaw_rate_dps, 0)                   # now the left really is more open


if __name__ == "__main__":
    unittest.main()
