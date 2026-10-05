import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace

from control.mavlink_driver import COPTER_GUIDED, VELOCITY_AND_YAW_RATE, MavlinkDriver
from datatypes import Decision, DriveCommand, Mode, Observation, Pose, RangeBeam, RangeScan
from telemetry.flight_log import FlightLog


class FakeMavlink:
    """Stands in for pymavlink's `mavlink` constants module."""
    MAV_TYPE_GCS = 6
    MAV_MODE_FLAG_SAFETY_ARMED = 128
    MAVLINK_MSG_ID_LOCAL_POSITION_NED = 32
    MAVLINK_MSG_ID_ATTITUDE = 30
    MAVLINK_MSG_ID_SYS_STATUS = 1
    MAV_CMD_SET_MESSAGE_INTERVAL = 511
    MAV_CMD_COMPONENT_ARM_DISARM = 400
    MAV_CMD_NAV_TAKEOFF = 22
    MAV_FRAME_BODY_OFFSET_NED = 9


def msg(kind, **fields):
    return SimpleNamespace(get_type=lambda: kind, **fields)


def heartbeat(mode=COPTER_GUIDED, armed=True, type_=2):
    return msg("HEARTBEAT", custom_mode=mode, base_mode=128 if armed else 0, type=type_)


class FakeConn:
    target_system, target_component = 1, 1

    def __init__(self):
        self.inbox, self.sent, self.modes, self.closed = [], [], [], False
        outer = self
        self.mav = SimpleNamespace(
            command_long_send=lambda *a: outer.sent.append(("command_long", a)),
            set_position_target_local_ned_send=lambda *a: outer.sent.append(("velocity", a)))

    def wait_heartbeat(self, timeout=None):
        pass

    def recv_match(self, blocking=False):
        return self.inbox.pop(0) if self.inbox else None

    def set_mode(self, name):
        self.modes.append(name)

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def make(**kw):
    conn, clock = FakeConn(), Clock()
    drv = MavlinkDriver(conn, FakeMavlink, clock=clock, sleep=clock.sleep, **kw)
    return drv, conn, clock


class MavlinkDriverTests(unittest.TestCase):
    def test_airborne_needs_armed_and_off_the_ground(self):
        drv, conn, _ = make()
        conn.inbox = [heartbeat(armed=True), msg("LOCAL_POSITION_NED", x=0, y=0, z=-0.1)]
        self.assertFalse(drv.airborne())                 # armed, but still on the pad
        conn.inbox = [msg("LOCAL_POSITION_NED", x=0, y=0, z=-2.0)]
        self.assertTrue(drv.airborne())
        conn.inbox = [heartbeat(armed=False)]
        self.assertFalse(drv.airborne())                 # disarmed (landed / never took off)

    def test_type_mask_uses_only_velocity_and_yaw_rate(self):
        self.assertEqual(VELOCITY_AND_YAW_RATE, 0x7C7)
        for bit in (3, 4, 5, 11):                      # vx, vy, vz, yaw_rate must NOT be ignored
            self.assertEqual(VELOCITY_AND_YAW_RATE & (1 << bit), 0)

    def test_send_builds_a_body_frame_velocity_and_yaw_rate_message(self):
        drv, conn, _ = make()
        conn.inbox = [heartbeat()]
        drv.send(DriveCommand(yaw_rate_dps=90.0, forward_mps=0.5))
        kind, a = conn.sent[-1]
        self.assertEqual(kind, "velocity")
        (_, sys_id, comp, frame, mask, *rest) = a
        self.assertEqual((sys_id, comp, frame, mask), (1, 1, 9, 0x7C7))
        x, y, z, vx, vy, vz, ax, ay, az, yaw, yaw_rate = rest
        self.assertEqual((vx, vy, vz), (0.5, 0.0, 0.0))            # forward only, hold altitude
        self.assertAlmostEqual(yaw_rate, math.pi / 2)               # rad/s, clockwise positive

    def test_nothing_is_sent_when_the_pilot_has_taken_over(self):
        drv, conn, _ = make()
        conn.inbox = [heartbeat(mode=5)]                            # LOITER
        drv.send(DriveCommand(0, 0.5))
        drv.stop()
        self.assertEqual(conn.sent, [])
        self.assertFalse(drv.autonomy_permitted())

    def test_a_silent_flight_controller_means_no_autonomy(self):
        drv, conn, clock = make(heartbeat_timeout_s=2.0)
        conn.inbox = [heartbeat()]
        self.assertTrue(drv.autonomy_permitted())
        clock.t += 3.0
        self.assertFalse(drv.autonomy_permitted())

    def test_ground_station_heartbeats_are_ignored(self):
        drv, conn, _ = make()
        conn.inbox = [heartbeat(mode=5, type_=6)]                   # a GCS claiming LOITER
        drv._pump()
        self.assertIsNone(drv._mode)

    def test_pose_converts_from_mavlink_ned_to_east_north(self):
        drv, conn, _ = make()
        conn.inbox = [msg("LOCAL_POSITION_NED", x=10.0, y=3.0, z=-1.5), msg("ATTITUDE", yaw=0.5)]
        p = drv.pose()
        self.assertEqual((p.x, p.y, p.yaw), (3.0, 10.0, 0.5))       # east = MAVLink y, north = MAVLink x

    def test_battery_is_none_when_unknown(self):
        drv, conn, _ = make()
        conn.inbox = [msg("SYS_STATUS", battery_remaining=-1)]
        self.assertIsNone(drv.battery_pct())
        conn.inbox = [msg("SYS_STATUS", battery_remaining=62)]
        self.assertEqual(drv.battery_pct(), 62.0)

    def test_arm_switches_to_guided_first_and_reports_refusal_helpfully(self):
        drv, conn, _ = make()
        conn.inbox = [heartbeat(armed=False)]
        with self.assertRaises(RuntimeError) as ctx:
            drv.arm()
        self.assertEqual(conn.modes, ["GUIDED"])
        self.assertIn("PreArm", str(ctx.exception))

    def test_arm_and_takeoff_happy_path(self):
        drv, conn, _ = make(takeoff_alt_m=2.0)
        conn.inbox = [heartbeat(armed=True)]
        drv.arm()
        conn.inbox = [msg("LOCAL_POSITION_NED", x=0, y=0, z=-1.9)]
        drv.takeoff()
        commands = [a for k, a in conn.sent if k == "command_long"]
        self.assertTrue(any(a[2] == 400 and a[4] == 1 for a in commands))     # arm
        self.assertTrue(any(a[2] == 22 and a[-1] == 2.0 for a in commands))   # takeoff to 2 m

    def test_land_rtl_and_shutdown(self):
        drv, conn, _ = make()
        conn.inbox = [heartbeat()]
        with drv:
            pass                                                     # __exit__: stop -> land -> disconnect
        self.assertEqual(conn.modes, ["LAND"])
        self.assertTrue(conn.closed)
        drv.return_to_launch()
        self.assertEqual(conn.modes[-1], "RTL")


class FlightLogTests(unittest.TestCase):
    def test_one_json_line_per_step_with_the_essentials(self):
        obs = Observation(now=1.234, pose=Pose(1.0, 2.0, 0.5), battery_pct=80.0,
                          scan=RangeScan((RangeBeam(0.0, 2.5), RangeBeam(1.0, None)), 4.0))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.jsonl")
            log = FlightLog(path)
            log.log(obs, Decision(DriveCommand(10.0, 0.5), Mode.PATROL, "x"))
            log.log(obs, Decision(DriveCommand(), Mode.HOLD))
            log.close()
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[0]["mode"], rows[0]["fwd_cmd"], rows[0]["ranges"]), ("PATROL", 0.5, [2.5, None]))
        self.assertIsNone(rows[0]["seen"])


if __name__ == "__main__":
    unittest.main()
