import unittest

from config import CameraConfig, ControlConfig, ShapingConfig
from control.command_shaper import CommandShaper
from control.follow_controller import FollowController
from datatypes import DriveCommand, TargetEstimate, TrackerOutput, TrackState

C = ControlConfig()
CAM = CameraConfig()


def out(state=TrackState.TRACKING, offset_x=0.0, offset_y=0.0, size=0.10):
    return TrackerOutput(state, TargetEstimate(offset_x, offset_y, size))


class FollowControllerTests(unittest.TestCase):
    def setUp(self):
        self.f = FollowController()

    def size_for_forward(self, forward_m):
        """The `size` that puts the target `forward_m` away on the camera axis
        (offset_x = offset_y = 0, camera level)."""
        return self.f.model.size_at_1m / forward_m

    def test_centred_and_at_hover_height_means_no_motion(self):
        size = self.size_for_forward(0.05)  # well inside every deadband
        r = self.f.desired(out(size=size), current_pitch_deg=0.0,
                           down_range_m=C.hover_height_above_target_m)
        self.assertTrue(r.cmd.is_zero)

    def test_turns_toward_target_when_far_off_to_the_side(self):
        far = self.size_for_forward(5.0)  # horiz well past yaw_release_m
        right = self.f.desired(out(offset_x=0.4, size=far), 0.0)
        left = self.f.desired(out(offset_x=-0.4, size=far), 0.0)
        self.assertGreater(right.cmd.yaw_rate_dps, 0)
        self.assertLess(left.cmd.yaw_rate_dps, 0)

    def test_yaw_deadband_and_clamp(self):
        far = self.size_for_forward(5.0)
        tiny = self.f.desired(out(offset_x=0.01, size=far), 0.0)
        self.assertEqual(tiny.cmd.yaw_rate_dps, 0.0)
        huge = self.f.desired(out(offset_x=0.99, size=far), 0.0)
        self.assertLessEqual(abs(huge.cmd.yaw_rate_dps), C.max_yaw_rate_dps)

    def test_yaw_fades_out_as_the_target_gets_close(self):
        from perception.geometry import RelativePosition
        def yaw_at(forward, right):
            return self.f._track(RelativePosition(forward, right, -1.0), height=0.3).cmd.yaw_rate_dps
        far, near = yaw_at(2.0, 0.6), yaw_at(0.2, 0.06)   # same bearing (~17 deg right)
        self.assertGreater(far, 0)
        self.assertGreater(near, 0)
        self.assertLess(near, far / 2)

    def test_close_but_behind_still_turns_toward_them(self):
        from perception.geometry import RelativePosition
        # Walked off behind-left of the drone, 0.4 m away: well inside yaw_release_m.
        r = self.f._track(RelativePosition(-0.3, -0.25, -0.8), height=0.3)
        self.assertLess(r.cmd.yaw_rate_dps, -10)

    def test_lost_close_and_behind_turns_instead_of_holding(self):
        from perception.geometry import RelativePosition
        behind = self.f._lost(RelativePosition(-0.4, 0.1, -0.8), height=0.3)
        in_front = self.f._lost(RelativePosition(0.4, 0.1, -0.8), height=0.3)
        self.assertGreater(behind.cmd.yaw_rate_dps, 0)
        self.assertTrue(in_front.cmd.is_zero)
        self.assertEqual(behind.camera_pitch_deg, self.f.aim_for_search())   # look out, not down

    def test_moves_toward_target_when_far_and_aligned(self):
        far = self.size_for_forward(5.0)
        r = self.f.desired(out(size=far), 0.0)
        self.assertGreater(r.cmd.forward_mps, 0)
        self.assertLessEqual(r.cmd.forward_mps, C.max_horizontal_mps + 1e-9)

    def test_no_horizontal_motion_within_the_deadband(self):
        size = self.size_for_forward(0.05)
        r = self.f.desired(out(size=size), 0.0, down_range_m=C.hover_height_above_target_m)
        self.assertEqual(r.cmd.horizontal_speed, 0.0)

    def test_climbs_when_below_the_desired_hover_height(self):
        size = self.size_for_forward(0.05)
        r = self.f.desired(out(size=size), 0.0, down_range_m=0.05)  # far below target height
        self.assertGreater(r.cmd.up_mps, 0)
        self.assertLessEqual(r.cmd.up_mps, C.max_climb_mps + 1e-9)

    def test_descends_when_above_the_desired_hover_height(self):
        size = self.size_for_forward(0.05)
        r = self.f.desired(out(size=size), 0.0, down_range_m=5.0)  # far above target height
        self.assertLess(r.cmd.up_mps, 0)
        self.assertGreaterEqual(r.cmd.up_mps, -C.max_descend_mps - 1e-9)

    def test_no_vertical_motion_within_the_height_deadband(self):
        size = self.size_for_forward(0.05)
        r = self.f.desired(out(size=size), 0.0, down_range_m=C.hover_height_above_target_m)
        self.assertEqual(r.cmd.up_mps, 0.0)

    def test_approach_slows_down_while_still_high_above_the_target(self):
        # Same horizontal distance (10m forward), two different vision-estimated
        # heights above the target -- built via the camera model's own inverse
        # projection so the offsets fed to the controller are self-consistent.
        from perception.geometry import RelativePosition
        low_up = -(C.hover_height_above_target_m + CAM.face_to_head_top_m)
        high_up = -(C.hover_height_above_target_m + 1.0 + CAM.face_to_head_top_m)
        low = self.f.desired(TrackerOutput(TrackState.TRACKING,
                             self.f.model.project(RelativePosition(10.0, 0.0, low_up), 0.0)), 0.0)
        high = self.f.desired(TrackerOutput(TrackState.TRACKING,
                              self.f.model.project(RelativePosition(10.0, 0.0, high_up), 0.0)), 0.0)
        self.assertLess(high.cmd.horizontal_speed, low.cmd.horizontal_speed)

    def test_down_rangefinder_overrides_vision_once_nearly_overhead(self):
        size = self.size_for_forward(0.05)  # inside down_valid_radius_m
        r = self.f.desired(out(size=size), 0.0, down_range_m=1.23)
        self.assertAlmostEqual(r.height_above_target_m, 1.23)

    def test_vision_height_used_when_far_away_even_with_a_down_reading(self):
        far = self.size_for_forward(5.0)  # well outside down_valid_radius_m
        r = self.f.desired(out(size=far), 0.0, down_range_m=1.23)
        self.assertNotAlmostEqual(r.height_above_target_m, 1.23, places=2)

    def test_lost_holds_position_if_probably_right_underneath(self):
        close = self.size_for_forward(0.1)  # inside lost_hold_within_m
        r = self.f.desired(out(TrackState.LOST, size=close), 0.0)
        self.assertTrue(r.cmd.is_zero)

    def test_lost_turns_toward_last_known_side_and_never_advances(self):
        far = self.size_for_forward(5.0)
        right = self.f.desired(out(TrackState.LOST, offset_x=0.8, size=far), 0.0)
        left = self.f.desired(out(TrackState.LOST, offset_x=-0.8, size=far), 0.0)
        self.assertEqual((right.cmd.yaw_rate_dps, right.cmd.forward_mps),
                         (C.lost_search_yaw_dps, 0.0))
        self.assertEqual(left.cmd.yaw_rate_dps, -C.lost_search_yaw_dps)

    def test_searching_with_no_target_holds_still_and_aims_the_search_pitch(self):
        r = self.f.desired(TrackerOutput(TrackState.SEARCHING, None), 0.0)
        self.assertTrue(r.cmd.is_zero)
        self.assertEqual(r.camera_pitch_deg, self.f.aim_for_search())

    def test_gimbal_aims_more_level_for_a_target_further_away_at_the_same_height(self):
        # Same "how far below us" (height_above_target), more horizontal distance:
        # the pitch-down angle should shrink toward level, not grow toward straight down.
        near = self.f.desired(out(size=self.size_for_forward(1.0)), 0.0)
        far = self.f.desired(out(size=self.size_for_forward(5.0)), 0.0)
        self.assertGreater(near.camera_pitch_deg, far.camera_pitch_deg)

    def test_gimbal_points_straight_down_when_directly_overhead(self):
        from perception.geometry import RelativePosition
        # Straight down (horizontal = 0): the internal aim helper should ask
        # for max_pitch_deg regardless of how far above the target we are.
        # (Exactly-overhead can't be expressed as a Detection at all -- a
        # point on the camera's own optical axis has no image position -- so
        # this checks the aiming geometry directly rather than round-tripping
        # through the camera model.)
        pitch = self.f._aim_pitch(RelativePosition(0.0, 0.0, -1.0), height=0.88)
        self.assertAlmostEqual(pitch, CAM.max_pitch_deg, places=1)

    def test_fixed_camera_never_reports_a_new_pitch_request(self):
        f = FollowController(camera=CameraConfig(gimbal=False))
        r = f.desired(out(size=self.size_for_forward(1.0)), f.camera_cfg.fixed_pitch_deg)
        self.assertEqual(r.camera_pitch_deg, CameraConfig(gimbal=False).fixed_pitch_deg)


class CommandShaperTests(unittest.TestCase):
    def test_acceleration_is_limited(self):
        s, c = CommandShaper(), ShapingConfig()
        first = s.shape(DriveCommand(0, 1.0), 0.0).forward_mps
        second = s.shape(DriveCommand(0, 1.0), 0.1).forward_mps
        self.assertLessEqual(first, c.max_forward_accel_mps2 * 0.1 + 1e-9)
        self.assertLessEqual(second - first, c.max_forward_accel_mps2 * 0.1 + 1e-9)

    def test_braking_is_faster_than_accelerating(self):
        s, c = CommandShaper(), ShapingConfig()
        cruising = 0.0
        for i in range(30):
            cruising = s.shape(DriveCommand(0, 1.0), i * 0.1).forward_mps
        after = s.shape(DriveCommand(0, 0.0), 3.0).forward_mps
        self.assertGreater(cruising - after, c.max_forward_accel_mps2 * 0.1)

    def test_hard_caps_all_four_axes(self):
        s, c = CommandShaper(), ShapingConfig()
        cmd = DriveCommand()
        for i in range(200):
            cmd = s.shape(DriveCommand(999, 999, 999, 999), i * 0.1)
        self.assertEqual((cmd.yaw_rate_dps, cmd.forward_mps, cmd.right_mps, cmd.up_mps),
                         (c.max_yaw_rate_dps, c.max_forward_mps, c.max_right_mps, c.max_up_mps))
        for i in range(200, 400):
            cmd = s.shape(DriveCommand(-999, -999, -999, -999), i * 0.1)
        self.assertEqual((cmd.yaw_rate_dps, cmd.forward_mps, cmd.right_mps, cmd.up_mps),
                         (-c.max_yaw_rate_dps, -c.max_backward_mps, -c.max_right_mps, -c.max_down_mps))

    def test_reset_returns_to_zero(self):
        s = CommandShaper()
        for i in range(20):
            s.shape(DriveCommand(30, 1.0, 1.0, 1.0), i * 0.1)
        s.reset()
        cmd = s.shape(DriveCommand(30, 1.0, 1.0, 1.0), 5.0)
        self.assertLess(cmd.forward_mps, 0.11)
        self.assertLess(cmd.right_mps, 0.11)
        self.assertLess(cmd.up_mps, 0.11)


if __name__ == "__main__":
    unittest.main()
