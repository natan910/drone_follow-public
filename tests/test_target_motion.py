import math
import unittest

from control.target_motion import TargetVelocity
from datatypes import Pose
from perception.geometry import RelativePosition


class TargetVelocityTests(unittest.TestCase):
    def test_relative_position_is_rotated_into_the_world(self):
        # facing east (yaw 90 deg): "forward" is +x, "right" is -y (south)
        x, y = TargetVelocity.world(Pose(1.0, 2.0, math.radians(90)), RelativePosition(3.0, 1.0, 0.0))
        self.assertAlmostEqual(x, 4.0)
        self.assertAlmostEqual(y, 1.0)

    def test_a_walking_target_is_measured_even_while_the_drone_moves_and_turns(self):
        tv = TargetVelocity(smoothing=0.5)
        for i in range(40):
            t = i * 0.1
            target = (0.4 * t, 1.0)                                   # walking east at 0.4 m/s
            pose = Pose(-0.2 * t, 0.0, math.radians(10 * i))          # drone drifting and spinning
            s, c = math.sin(pose.yaw), math.cos(pose.yaw)
            dx, dy = target[0] - pose.x, target[1] - pose.y
            tv.sighting(pose, RelativePosition(dx * s + dy * c, dx * c - dy * s, -1.0), t)
        vx, vy = tv.velocity(3.9)
        self.assertAlmostEqual(vx, 0.4, places=2)
        self.assertAlmostEqual(vy, 0.0, places=2)

    def test_body_frame_matches_the_heading(self):
        tv = TargetVelocity(smoothing=1.0)
        tv.sighting(Pose(0, 0, 0), RelativePosition(1.0, 0.0, 0.0), 0.0)
        tv.sighting(Pose(0, 0, 0), RelativePosition(1.5, 0.0, 0.0), 0.5)   # moving north at 1 m/s
        fwd, right = tv.body_frame(Pose(0, 0, math.radians(90)), 0.5)     # while we face east
        self.assertAlmostEqual(fwd, 0.0, places=6)
        self.assertAlmostEqual(right, -1.0, places=6)                     # north is to our left

    def test_fades_out_when_sightings_stop_and_ignores_long_gaps(self):
        tv = TargetVelocity(smoothing=1.0, max_gap_s=0.5, fade_s=1.0)
        tv.sighting(Pose(0, 0, 0), RelativePosition(0.0, 0.0, 0.0), 0.0)
        tv.sighting(Pose(0, 0, 0), RelativePosition(0.5, 0.0, 0.0), 0.5)
        self.assertAlmostEqual(tv.velocity(0.5)[1], 1.0)
        self.assertAlmostEqual(tv.velocity(1.5)[1], 0.5)                  # half faded
        self.assertEqual(tv.velocity(3.0), (0.0, 0.0))
        tv.sighting(Pose(0, 0, 0), RelativePosition(9.0, 0.0, 0.0), 10.0)  # after a long gap: no jump
        self.assertEqual(tv.velocity(10.0), (0.0, 0.0))

    def test_capped_and_deadbanded(self):
        tv = TargetVelocity(smoothing=1.0, max_speed=1.5, deadband=0.1)
        tv.sighting(Pose(0, 0, 0), RelativePosition(0.0, 0.0, 0.0), 0.0)
        tv.sighting(Pose(0, 0, 0), RelativePosition(0.0, 0.02, 0.0), 0.5)  # 4 cm/s of jitter
        self.assertEqual(tv.velocity(0.5), (0.0, 0.0))
        tv.sighting(Pose(0, 0, 0), RelativePosition(2.0, 0.02, 0.0), 0.6)  # a 20 m/s glitch
        self.assertAlmostEqual(math.hypot(*tv.velocity(0.6)), 1.5)


if __name__ == "__main__":
    unittest.main()
