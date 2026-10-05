import unittest

from config import SafetyConfig
from datatypes import Observation, Pose, RangeBeam, RangeScan
from safety.supervisor import SafetyAction, SafetySupervisor

SCAN = RangeScan((RangeBeam(0.0, None),), 4.0)


def obs(now=1.0, x=0.0, y=0.0, battery=90.0, scan=SCAN, frame_age=0.0, scan_age=0.0):
    return Observation(now=now, pose=Pose(x, y, 0.0), scan=scan, battery_pct=battery,
                       frame_age=frame_age, scan_age=scan_age)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.s = SafetySupervisor(SafetyConfig())

    def act(self, **kw):
        return self.s.check(obs(**kw))[0]

    def test_healthy_flight_is_ok(self):
        self.assertEqual(self.act(), SafetyAction.OK)

    def test_stalled_camera_holds_then_sends_home(self):
        self.assertEqual(self.act(frame_age=2.0), SafetyAction.HOLD)
        self.assertEqual(self.act(frame_age=11.0), SafetyAction.RETURN)

    def test_missing_or_stale_range_data_holds_then_lands(self):
        self.assertEqual(self.act(scan=None, scan_age=0.0), SafetyAction.HOLD)
        self.assertEqual(self.act(scan_age=1.0), SafetyAction.HOLD)
        self.assertEqual(self.act(scan_age=4.0), SafetyAction.LAND)

    def test_range_data_is_not_required_for_the_dry_run(self):
        s = SafetySupervisor(SafetyConfig(require_scan=False))
        self.assertEqual(s.check(obs(scan=None, scan_age=99))[0], SafetyAction.OK)

    def test_battery_thresholds(self):
        self.assertEqual(self.act(battery=30.0), SafetyAction.RETURN)
        self.assertEqual(SafetySupervisor().check(obs(battery=15.0))[0], SafetyAction.LAND)

    def test_unknown_battery_is_not_a_reason_to_act(self):
        self.assertEqual(self.act(battery=None), SafetyAction.OK)

    def test_geofence_and_flight_time(self):
        self.assertEqual(self.act(x=31.0), SafetyAction.RETURN)
        s = SafetySupervisor(SafetyConfig(max_flight_s=100))
        s.check(obs(now=0.0))
        self.assertEqual(s.check(obs(now=101.0))[0], SafetyAction.RETURN)

    def test_return_is_latched_but_hold_is_not(self):
        self.assertEqual(self.act(frame_age=2.0), SafetyAction.HOLD)
        self.assertEqual(self.act(), SafetyAction.OK)                    # hold clears
        self.assertEqual(self.act(battery=20.0), SafetyAction.RETURN)
        action, why = self.s.check(obs(battery=95.0))                    # ...but return sticks
        self.assertEqual((action, why), (SafetyAction.RETURN, "battery low"))

    def test_the_worst_problem_wins(self):
        self.assertEqual(self.act(battery=10.0, frame_age=2.0), SafetyAction.LAND)


if __name__ == "__main__":
    unittest.main()
