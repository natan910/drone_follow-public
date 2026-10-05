"""
MavlinkDriver: camera-tilt (DO_MOUNT_CONTROL) behaviour, and the arm request.

Found in SITL: with no gimbal configured, ArduPilot answered every tilt command
with COMMAND_ACK ... FAILED, and the driver sent a new one on every control step,
forever. These tests pin the fix. Hand-written fakes only, no pymavlink, no hardware.
"""

import unittest

from control.mavlink_driver import MavlinkDriver


class FakeMavlink:
    """Stands in for pymavlink's `mavlink` module: only the constants the driver uses
    (values are the real MAVLink ones)."""
    MAV_TYPE_GCS = 6
    MAV_MODE_FLAG_SAFETY_ARMED = 128
    MAV_FRAME_BODY_OFFSET_NED = 9
    MAV_CMD_NAV_TAKEOFF = 22
    MAV_CMD_DO_MOUNT_CONTROL = 205
    MAV_CMD_COMPONENT_ARM_DISARM = 400
    MAV_CMD_SET_MESSAGE_INTERVAL = 511
    MAVLINK_MSG_ID_LOCAL_POSITION_NED = 32
    MAVLINK_MSG_ID_ATTITUDE = 30
    MAVLINK_MSG_ID_SYS_STATUS = 1
    MAV_RESULT_ACCEPTED = 0
    MAV_RESULT_TEMPORARILY_REJECTED = 1
    MAV_RESULT_DENIED = 2
    MAV_RESULT_UNSUPPORTED = 3
    MAV_RESULT_FAILED = 4
    MAV_RESULT_IN_PROGRESS = 5


M = FakeMavlink
COPTER_QUAD = 2
GUIDED = 4


class FakeMsg:
    def __init__(self, kind, **fields):
        self._kind = kind
        self.__dict__.update(fields)

    def get_type(self):
        return self._kind


def heartbeat(mode=GUIDED, armed=False):
    return FakeMsg("HEARTBEAT", type=COPTER_QUAD, custom_mode=mode,
                   base_mode=M.MAV_MODE_FLAG_SAFETY_ARMED if armed else 0)


def ack(command, result):
    return FakeMsg("COMMAND_ACK", command=command, result=result)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeSender:
    """conn.mav: records every command_long_send as a tuple
    (target_system, target_component, command, confirmation, p1, ..., p7)."""

    def __init__(self, on_send=None):
        self.sent = []
        self.on_send = on_send

    def command_long_send(self, *args):
        self.sent.append(args)
        if self.on_send:
            self.on_send(args)


class FakeConn:
    target_system = 1
    target_component = 1

    def __init__(self):
        self.incoming = []
        self.modes = []
        self.mav = FakeSender(on_send=self._react)

    def _react(self, args):
        # A flight controller that obeys an arm request.
        if args[2] == M.MAV_CMD_COMPONENT_ARM_DISARM and args[4] == 1:
            self.incoming.append(heartbeat(GUIDED, armed=True))

    def recv_match(self, blocking=False):
        return self.incoming.pop(0) if self.incoming else None

    def set_mode(self, name):
        self.modes.append(name)
        if name == "GUIDED":
            self.incoming.append(heartbeat(GUIDED))

    def wait_heartbeat(self, timeout=None):
        pass

    def close(self):
        pass


def make_driver(**kwargs):
    clock, conn = FakeClock(), FakeConn()
    drv = MavlinkDriver(conn, FakeMavlink, clock=clock, sleep=clock.sleep, **kwargs)
    return drv, conn, clock


def mount_commands(conn):
    return [a for a in conn.mav.sent if a[2] == M.MAV_CMD_DO_MOUNT_CONTROL]


class CameraTiltSendingTests(unittest.TestCase):
    def test_same_angle_every_step_is_sent_only_once(self):
        drv, conn, clock = make_driver()
        for _ in range(50):              # 50 steps in one second, angle never changes
            drv.set_camera_pitch(0.0)
            clock.now += 0.02
        self.assertEqual(len(mount_commands(conn)), 1)

    def test_same_angle_is_repeated_after_the_resend_interval(self):
        drv, conn, clock = make_driver(gimbal_resend_s=2.0)
        drv.set_camera_pitch(25.0)
        clock.now += 1.9
        drv.set_camera_pitch(25.0)
        self.assertEqual(len(mount_commands(conn)), 1)
        clock.now += 0.2                 # 2.1 s since the last send
        drv.set_camera_pitch(25.0)
        self.assertEqual(len(mount_commands(conn)), 2)

    def test_a_new_angle_is_sent_at_once(self):
        drv, conn, _ = make_driver()
        drv.set_camera_pitch(0.0)
        drv.set_camera_pitch(10.0)
        self.assertEqual(len(mount_commands(conn)), 2)

    def test_tiny_changes_are_skipped_but_a_slow_creep_still_gets_through(self):
        drv, conn, _ = make_driver(gimbal_min_step_deg=0.5)
        for angle in (0.0, 0.2, 0.4):    # each within 0.5 deg of the last one sent
            drv.set_camera_pitch(angle)
        self.assertEqual(len(mount_commands(conn)), 1)
        drv.set_camera_pitch(0.6)        # 0.6 from the last SENT angle (0.0), not from 0.4
        self.assertEqual(len(mount_commands(conn)), 2)

    def test_command_format_is_unchanged(self):
        drv, conn, _ = make_driver()
        drv.set_camera_pitch(30.0)       # 30 deg down
        (cmd,) = mount_commands(conn)
        self.assertEqual(cmd[:4], (1, 1, M.MAV_CMD_DO_MOUNT_CONTROL, 0))
        self.assertEqual(cmd[4:], (-30.0, 0.0, 0.0, 0, 0, 0, 2))  # pitch (negative = down), roll, yaw, ..., MAVLINK_TARGETING

    def test_gimbal_false_never_sends(self):
        drv, conn, clock = make_driver(gimbal=False)
        for angle in (0.0, 10.0, 20.0):
            drv.set_camera_pitch(angle)
            clock.now += 5.0
        self.assertEqual(mount_commands(conn), [])


class CameraTiltRefusalTests(unittest.TestCase):
    def refuse(self, conn, result=M.MAV_RESULT_FAILED):
        conn.incoming.append(ack(M.MAV_CMD_DO_MOUNT_CONTROL, result))

    def test_gives_up_after_three_refusals_in_a_row_and_warns_once(self):
        drv, conn, clock = make_driver(gimbal_max_refusals=3)
        drv.set_camera_pitch(0.0)
        with self.assertLogs("control.mavlink_driver", level="WARNING") as logs:
            for angle in (10.0, 20.0):   # refusals 1 and 2: keep trying
                self.refuse(conn)
                drv.set_camera_pitch(angle)
            self.assertEqual(len(mount_commands(conn)), 3)
            self.refuse(conn)            # refusal 3: stop
            for angle in (30.0, 40.0, 50.0):
                drv.set_camera_pitch(angle)
                clock.now += 10.0        # not even after the resend interval
        self.assertEqual(len(mount_commands(conn)), 3)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("no gimbal", logs.records[0].getMessage())

    def test_an_accepted_answer_resets_the_count(self):
        drv, conn, _ = make_driver(gimbal_max_refusals=3)
        drv.set_camera_pitch(0.0)
        angle = 0.0
        for result in (M.MAV_RESULT_FAILED, M.MAV_RESULT_FAILED, M.MAV_RESULT_ACCEPTED,
                       M.MAV_RESULT_FAILED, M.MAV_RESULT_FAILED):
            self.refuse(conn, result)
            angle += 10.0
            drv.set_camera_pitch(angle)
        self.assertEqual(len(mount_commands(conn)), 6)   # never three failures in a row: still trying
        self.refuse(conn)                                # third in a row after the reset
        with self.assertLogs("control.mavlink_driver", level="WARNING"):
            drv.set_camera_pitch(angle + 10.0)
        self.assertEqual(len(mount_commands(conn)), 6)   # now it has given up

    def test_denied_and_unsupported_count_as_refusals_too(self):
        drv, conn, _ = make_driver(gimbal_max_refusals=2)
        drv.set_camera_pitch(0.0)
        self.refuse(conn, M.MAV_RESULT_DENIED)
        drv.set_camera_pitch(10.0)
        self.refuse(conn, M.MAV_RESULT_UNSUPPORTED)
        with self.assertLogs("control.mavlink_driver", level="WARNING"):
            drv.set_camera_pitch(20.0)
        self.assertEqual(len(mount_commands(conn)), 2)

    def test_temporary_rejection_is_not_a_refusal(self):
        drv, conn, _ = make_driver(gimbal_max_refusals=3)
        drv.set_camera_pitch(0.0)
        for i in range(1, 8):
            self.refuse(conn, M.MAV_RESULT_TEMPORARILY_REJECTED)
            drv.set_camera_pitch(10.0 * i)
        self.assertEqual(len(mount_commands(conn)), 8)

    def test_refusals_of_other_commands_are_ignored(self):
        drv, conn, _ = make_driver(gimbal_max_refusals=3)
        drv.set_camera_pitch(0.0)
        for i in range(1, 8):
            conn.incoming.append(ack(M.MAV_CMD_NAV_TAKEOFF, M.MAV_RESULT_FAILED))
            drv.set_camera_pitch(10.0 * i)
        self.assertEqual(len(mount_commands(conn)), 8)

    def test_the_answer_is_also_picked_up_by_the_normal_polling(self):
        # The main loop calls pose()/autonomy_permitted() every step; a refusal
        # read there must count even if set_camera_pitch has not run since.
        drv, conn, _ = make_driver(gimbal_max_refusals=1)
        drv.set_camera_pitch(0.0)
        self.refuse(conn)
        with self.assertLogs("control.mavlink_driver", level="WARNING"):
            drv.pose()
        drv.set_camera_pitch(10.0)
        self.assertEqual(len(mount_commands(conn)), 1)


class ArmRequestTests(unittest.TestCase):
    def test_arm_is_a_plain_request_never_the_force_arm_magic_number(self):
        drv, conn, _ = make_driver()
        drv.arm()
        self.assertEqual(conn.modes, ["GUIDED"])
        arms = [a for a in conn.mav.sent if a[2] == M.MAV_CMD_COMPONENT_ARM_DISARM]
        self.assertEqual(len(arms), 1)
        arm = arms[0]
        self.assertEqual(arm[4], 1)              # param1: 1 = arm
        self.assertEqual(arm[5], 0)              # param2: 0 = normal; 2989 would force past the pre-arm checks
        self.assertNotIn(2989, arm)
        self.assertNotIn(21196, arm)


if __name__ == "__main__":
    unittest.main()
