"""
Px4Driver (control/px4_driver.py) and open_flight_controller
(control/flight_controller.py) against a fake PX4. No pymavlink, no mocks.

The fake answers like PX4 v1.17 SITL did in the cloud sandbox: DO_SET_MODE
switches the heartbeat's custom_mode (main << 16 | sub << 24), NAV_TAKEOFF's
param7 is an AMSL altitude, and the local origin is NOT where it arms.
"""

import math
import threading
import time
import unittest
from types import SimpleNamespace

from control.flight_controller import detect_autopilot, open_flight_controller
from control.mavlink_driver import MavlinkDriver
from control.px4_driver import (MODE_HOLD, MODE_LAND, MODE_OFFBOARD, MODE_RTL, MODE_TAKEOFF,
                                PX4_VELOCITY_AND_YAW_RATE, Px4Driver, px4_custom_mode,
                                px4_mode_name, px4_mode_parts)
from datatypes import DriveCommand

# MAVLink numbers (checked against pymavlink's generated dialect)
MAV = SimpleNamespace(
    MAV_TYPE_GCS=6, MAV_TYPE_QUADROTOR=2, MAV_TYPE_ONBOARD_CONTROLLER=18,
    MAV_AUTOPILOT_ARDUPILOTMEGA=3, MAV_AUTOPILOT_INVALID=8, MAV_AUTOPILOT_PX4=12,
    MAV_MODE_FLAG_SAFETY_ARMED=128, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED=1,
    MAV_CMD_NAV_TAKEOFF=22, MAV_CMD_DO_SET_MODE=176, MAV_CMD_DO_MOUNT_CONTROL=205,
    MAV_CMD_COMPONENT_ARM_DISARM=400, MAV_CMD_SET_MESSAGE_INTERVAL=511, MAV_CMD_REQUEST_MESSAGE=512,
    MAV_FRAME_LOCAL_NED=1, MAV_FRAME_BODY_NED=8, MAV_FRAME_BODY_OFFSET_NED=9,
    MAVLINK_MSG_ID_SYS_STATUS=1, MAVLINK_MSG_ID_ATTITUDE=30, MAVLINK_MSG_ID_LOCAL_POSITION_NED=32,
    MAVLINK_MSG_ID_GLOBAL_POSITION_INT=33, MAVLINK_MSG_ID_EXTENDED_SYS_STATE=245,
    MAV_RESULT_ACCEPTED=0, MAV_RESULT_TEMPORARILY_REJECTED=1, MAV_RESULT_DENIED=2,
    MAV_RESULT_UNSUPPORTED=3, MAV_RESULT_FAILED=4, MAV_RESULT_IN_PROGRESS=5,
)


class Msg:
    def __init__(self, kind, **fields):
        self._kind = kind
        self.__dict__.update(fields)

    def get_type(self):
        return self._kind


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class FakePx4:
    """Both the pymavlink connection and its .mav sender, plus a toy vehicle."""

    def __init__(self, autopilot=12, refuse_modes=(), refuse_arm=False, ground_amsl=120.0,
                 publish_global=True, start_mode=MODE_HOLD):
        self.mav = self
        self.target_system, self.target_component = 1, 1
        self.autopilot = autopilot
        self.refuse_modes, self.refuse_arm = set(refuse_modes), refuse_arm
        self.ground_amsl, self.publish_global = ground_amsl, publish_global
        self.mode = px4_custom_mode(*start_mode)
        self.armed = False
        self.north, self.east, self.down = 12.0, -7.0, -0.4    # EKF origin is NOT home
        self.ground_down = self.down
        self.yaw = 0.0
        self.landed_state = 1
        self.queue, self.commands, self.setpoints = [], [], []
        self.heartbeats_sent = 0
        self.takeoff_param7 = None
        self.closed = False
        self.mode_changes_ignored = 0         # drop this many DO_SET_MODEs silently (lost packets)
        self.hb()
        self.publish()

    # ---- vehicle -> us --------------------------------------------------
    def hb(self):
        base = 1 | (128 if self.armed else 0)
        self.queue.append(Msg("HEARTBEAT", type=2, autopilot=self.autopilot,
                              custom_mode=self.mode, base_mode=base))

    def publish(self):
        self.queue.append(Msg("LOCAL_POSITION_NED", x=self.north, y=self.east, z=self.down))
        self.queue.append(Msg("ATTITUDE", yaw=self.yaw))
        self.queue.append(Msg("SYS_STATUS", battery_remaining=87))
        self.queue.append(Msg("EXTENDED_SYS_STATE", landed_state=self.landed_state))
        if self.publish_global:
            amsl = self.ground_amsl + (self.ground_down - self.down)
            self.queue.append(Msg("GLOBAL_POSITION_INT", alt=int(amsl * 1000), relative_alt=0))

    def ack(self, command, result):
        self.queue.append(Msg("COMMAND_ACK", command=command, result=result))

    def text(self, severity, text):
        self.queue.append(Msg("STATUSTEXT", severity=severity, text=text))

    # ---- pymavlink connection API --------------------------------------
    def recv_match(self, blocking=False, type=None, timeout=None):
        for i, msg in enumerate(self.queue):
            if type is None or msg.get_type() == type:
                return self.queue.pop(i)
        return None

    def wait_heartbeat(self, timeout=None):
        self.hb()
        return self.queue[-1]

    def close(self):
        self.closed = True

    # ---- pymavlink .mav sender API --------------------------------------
    def heartbeat_send(self, type_, autopilot, base_mode, custom_mode, status):
        self.heartbeats_sent += 1
        self.last_heartbeat_sent = (type_, autopilot)

    def set_position_target_local_ned_send(self, t, ts, tc, frame, mask, x, y, z, vx, vy, vz,
                                           ax, ay, az, yaw, yaw_rate):
        self.setpoints.append(dict(frame=frame, mask=mask, vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate,
                                   mode=self.mode))

    def command_long_send(self, ts, tc, command, confirmation, *p):
        self.commands.append((command, p))
        if command == MAV.MAV_CMD_DO_SET_MODE:
            if self.mode_changes_ignored > 0:
                self.mode_changes_ignored -= 1
                return
            want = (int(p[1]), int(p[2]))
            if want in self.refuse_modes:
                self.text(3, f"Switching to mode {want} is currently not possible")
                self.ack(command, MAV.MAV_RESULT_DENIED)
                return
            self.mode = px4_custom_mode(*want)
            self.ack(command, MAV.MAV_RESULT_ACCEPTED)
            self.hb()
        elif command == MAV.MAV_CMD_COMPONENT_ARM_DISARM:
            if self.refuse_arm:
                self.text(2, "Preflight Fail: No valid data from Compass")
                self.ack(command, MAV.MAV_RESULT_DENIED)
                return
            self.armed = p[0] == 1
            self.ack(command, MAV.MAV_RESULT_ACCEPTED)
            self.hb()
        elif command == MAV.MAV_CMD_NAV_TAKEOFF:
            self.takeoff_param7 = p[6]
            climb = 2.5 if math.isnan(p[6]) else p[6] - self.ground_amsl
            self.mode = px4_custom_mode(*MODE_TAKEOFF)
            self.down = self.ground_down - climb
            self.landed_state = 2
            self.ack(command, MAV.MAV_RESULT_ACCEPTED)
            self.hb()
            self.publish()
        elif command == MAV.MAV_CMD_DO_MOUNT_CONTROL:
            self.ack(command, MAV.MAV_RESULT_UNSUPPORTED)
        else:
            self.ack(command, MAV.MAV_RESULT_ACCEPTED)

    # ---- helpers ---------------------------------------------------------
    def commands_of(self, command):
        return [p for c, p in self.commands if c == command]


def make(fake=None, **kwargs):
    fake = fake or FakePx4()
    clock = FakeClock()
    kwargs.setdefault("start_thread", False)
    d = Px4Driver(fake, MAV, clock=clock, sleep=clock.sleep, **kwargs)
    return d, fake, clock


def flying(**kwargs):
    d, fake, clock = make(**kwargs)
    d.connect()
    d.arm()
    d.takeoff()
    return d, fake, clock


class TestModeNumbers(unittest.TestCase):
    def test_custom_mode_matches_what_px4_sitl_reported(self):
        self.assertEqual(px4_custom_mode(*MODE_HOLD), 50593792)      # seen in PX4 v1.17 SITL heartbeat
        self.assertEqual(px4_custom_mode(*MODE_OFFBOARD), 6 << 16)
        self.assertEqual(px4_mode_parts(px4_custom_mode(*MODE_LAND)), MODE_LAND)

    def test_names(self):
        self.assertEqual(px4_mode_name(px4_custom_mode(*MODE_OFFBOARD)), "OFFBOARD")
        self.assertEqual(px4_mode_name(px4_custom_mode(*MODE_RTL)), "AUTO.RTL")
        self.assertEqual(px4_mode_name(px4_custom_mode(3)), "POSCTL")
        self.assertEqual(px4_mode_name(None), "?")


class TestMission(unittest.TestCase):
    def test_connect_arm_takeoff_ends_in_offboard_with_control(self):
        d, fake, _ = flying()
        self.assertTrue(d.autonomy_permitted())
        self.assertEqual(d.mode_name(), "OFFBOARD")
        modes = [(int(p[1]), int(p[2])) for p in fake.commands_of(MAV.MAV_CMD_DO_SET_MODE)]
        self.assertEqual(modes, [MODE_HOLD, MODE_OFFBOARD])

    def test_arm_is_a_plain_arm_request(self):
        d, fake, _ = flying()
        (p,) = fake.commands_of(MAV.MAV_CMD_COMPONENT_ARM_DISARM)
        self.assertEqual(p[0], 1)
        self.assertEqual(p[1], 0)                 # never the 21196 force-arm number

    def test_takeoff_altitude_is_amsl(self):
        d, fake, _ = flying(fake=FakePx4(ground_amsl=250.0), takeoff_alt_m=1.5)
        self.assertAlmostEqual(fake.takeoff_param7, 251.5)
        self.assertAlmostEqual(d.pose().z, 1.5, places=3)

    def test_takeoff_without_global_position_lets_px4_pick(self):
        d, fake, _ = flying(fake=FakePx4(publish_global=False))
        self.assertTrue(math.isnan(fake.takeoff_param7))
        self.assertTrue(d.autonomy_permitted())

    def test_setpoints_are_streamed_before_asking_for_offboard(self):
        d, fake, _ = flying(offboard_prime_s=1.0, stream_hz=20)
        before = [s for s in fake.setpoints if s["mode"] != px4_custom_mode(*MODE_OFFBOARD)]
        self.assertGreaterEqual(len(before), 15)
        self.assertTrue(all(s["vx"] == s["vy"] == s["vz"] == s["yaw_rate"] == 0 for s in before))

    def test_extra_telemetry_requested(self):
        d, fake, _ = flying()
        ids = {int(p[0]) for p in fake.commands_of(MAV.MAV_CMD_SET_MESSAGE_INTERVAL)}
        self.assertTrue({30, 32, 1, 33, 245} <= ids)


class TestRefusals(unittest.TestCase):
    def test_arm_refused_fails_fast_and_says_why(self):
        d, fake, clock = make(FakePx4(refuse_arm=True))
        d.connect()
        start = clock()
        with self.assertRaises(RuntimeError) as cm:
            d.arm()
        self.assertIn("refused to arm", str(cm.exception))
        self.assertIn("Compass", str(cm.exception))
        self.assertLess(clock() - start, 1.0)

    def test_offboard_refused(self):
        d, fake, _ = make(FakePx4(refuse_modes=[MODE_OFFBOARD]))
        d.connect()
        d.arm()
        with self.assertRaises(RuntimeError) as cm:
            d.takeoff()
        self.assertIn("OFFBOARD", str(cm.exception))
        self.assertFalse(d.autonomy_permitted())

    def test_hold_refused_before_arming(self):
        d, fake, _ = make(FakePx4(refuse_modes=[MODE_HOLD], start_mode=MODE_LAND))
        d.connect()
        with self.assertRaises(RuntimeError) as cm:
            d.arm()
        self.assertIn("AUTO.LOITER", str(cm.exception))
        self.assertEqual(fake.commands_of(MAV.MAV_CMD_COMPONENT_ARM_DISARM), [])

    def test_connect_refuses_ardupilot(self):
        d, fake, _ = make(FakePx4(autopilot=3))
        with self.assertRaises(RuntimeError) as cm:
            d.connect()
        self.assertIn("--autopilot ardupilot", str(cm.exception))


class TestSend(unittest.TestCase):
    def test_body_frame_velocity_and_yaw_rate(self):
        d, fake, _ = flying()
        d.send(DriveCommand(yaw_rate_dps=30.0, forward_mps=1.2, right_mps=-0.4, up_mps=0.3))
        s = fake.setpoints[-1]
        self.assertEqual(s["frame"], MAV.MAV_FRAME_BODY_NED)      # PX4 rejects BODY_OFFSET_NED (9)
        self.assertEqual(s["mask"], PX4_VELOCITY_AND_YAW_RATE)
        self.assertEqual(s["mask"] & (1 << 9), 0)                 # FORCE_SET must be clear
        self.assertEqual((s["vx"], s["vy"], s["vz"]), (1.2, -0.4, -0.3))
        self.assertAlmostEqual(s["yaw_rate"], math.radians(30.0))

    def test_local_frame_rotates_by_heading(self):
        fake = FakePx4()
        fake.yaw = math.radians(90)                               # facing east
        d, fake, _ = flying(fake=fake, frame="local")
        d.send(DriveCommand(forward_mps=1.0, right_mps=0.5))
        s = fake.setpoints[-1]
        self.assertEqual(s["frame"], MAV.MAV_FRAME_LOCAL_NED)
        self.assertAlmostEqual(s["vx"], -0.5)                     # right of east-facing = south
        self.assertAlmostEqual(s["vy"], 1.0)                      # forward = east

    def test_dropped_when_pilot_took_over(self):
        d, fake, _ = flying()
        fake.mode = px4_custom_mode(3)                            # pilot flips to POSCTL
        fake.hb()
        n = len(fake.setpoints)
        d.send(DriveCommand(forward_mps=1.0))
        self.assertFalse(d.autonomy_permitted())
        self.assertEqual(len(fake.setpoints), n)

    def test_not_permitted_when_heartbeat_is_stale(self):
        d, fake, clock = flying()
        clock.t += 5.0
        self.assertFalse(d.autonomy_permitted())

    def test_not_permitted_disarmed_even_in_offboard(self):
        d, fake, _ = flying()
        fake.armed = False                                        # PX4 goes back to OFFBOARD after auto-disarm
        fake.hb()
        self.assertFalse(d.autonomy_permitted())


class TestStream(unittest.TestCase):
    def test_resends_latest_command_while_fresh_then_holds(self):
        d, fake, clock = flying(command_timeout_s=1.0)
        d.send(DriveCommand(forward_mps=0.8))
        clock.t += 0.5
        d._stream_tick()
        self.assertEqual(fake.setpoints[-1]["vx"], 0.8)
        clock.t += 0.6                                            # 1.1 s since the last send
        d._stream_tick()
        self.assertEqual(fake.setpoints[-1]["vx"], 0.0)
        d.send(DriveCommand(forward_mps=0.5))                     # the brain is back
        d._stream_tick()
        self.assertEqual(fake.setpoints[-1]["vx"], 0.5)

    def test_no_setpoints_before_takeoff_or_after_land(self):
        d, fake, clock = make()
        d.connect()
        d._stream_tick()
        self.assertEqual(fake.setpoints, [])
        d.arm()
        d.takeoff()
        d.land()
        n = len(fake.setpoints)
        clock.t += 0.1
        d._stream_tick()
        self.assertEqual(len(fake.setpoints), n)

    def test_heartbeat_once_a_second_as_onboard_controller(self):
        d, fake, clock = make()
        d.connect()
        for _ in range(40):                                       # 2 s at 20 Hz
            d._stream_tick()
            clock.t += 0.05
        self.assertEqual(fake.heartbeats_sent, 2)
        self.assertEqual(fake.last_heartbeat_sent, (18, 8))

    def test_pilot_takeover_is_never_replayed(self):
        d, fake, clock = flying()
        d.send(DriveCommand(forward_mps=1.0))
        fake.mode = px4_custom_mode(3)
        fake.hb()
        d.send(DriveCommand(forward_mps=1.0))                     # dropped, and forgets the old command
        fake.mode = px4_custom_mode(*MODE_OFFBOARD)               # pilot hands back
        fake.hb()
        d._stream_tick()
        self.assertEqual(fake.setpoints[-1]["vx"], 0.0)

    def test_real_thread_streams_and_stops(self):
        fake = FakePx4()
        d = Px4Driver(fake, MAV, stream_hz=50, offboard_prime_s=0.05)
        d.connect()
        d.arm()
        d.takeoff()
        d.send(DriveCommand(forward_mps=0.3))
        n = len(fake.setpoints)
        time.sleep(0.3)
        self.assertGreater(len(fake.setpoints), n + 5)
        d.disconnect()
        self.assertTrue(fake.closed)
        self.assertFalse(any(t.name == "px4-setpoints" for t in threading.enumerate()))


class TestLandAndState(unittest.TestCase):
    def test_land_and_rtl_set_px4_modes(self):
        d, fake, _ = flying()
        d.land()
        self.assertEqual(d.mode_name(), "AUTO.LAND")
        self.assertFalse(d.autonomy_permitted())
        d2, fake2, _ = flying()
        d2.return_to_launch()
        self.assertEqual(d2.mode_name(), "AUTO.RTL")

    def test_land_resends_a_lost_mode_command(self):
        d, fake, _ = flying()
        fake.mode_changes_ignored = 2
        d.land()
        self.assertEqual(d.mode_name(), "AUTO.LAND")
        self.assertEqual(len(fake.commands_of(MAV.MAV_CMD_DO_SET_MODE)), 2 + 3)   # Hold, OFFBOARD, 3x LAND

    def test_land_on_the_ground_does_not_wait(self):
        d, fake, clock = make()
        d.connect()
        start = clock()
        fake.mode_changes_ignored = 99
        d.land()
        self.assertLess(clock() - start, 0.5)

    def test_shutdown_sequence(self):
        d, fake, _ = flying()
        d.__exit__(None, None, None)
        self.assertEqual(px4_mode_parts(fake.mode), MODE_LAND)
        self.assertTrue(fake.closed)

    def test_pose_is_relative_to_where_it_armed(self):
        d, fake, _ = make()
        d.connect()
        d.arm()
        p = d.pose()
        self.assertAlmostEqual((p.x, p.y, p.z), (0.0, 0.0, 0.0))
        fake.north += 3.0
        fake.east += 4.0
        fake.down -= 2.0
        fake.publish()
        p = d.pose()
        self.assertAlmostEqual(p.x, 4.0)                          # x = east
        self.assertAlmostEqual(p.y, 3.0)                          # y = north
        self.assertAlmostEqual(p.z, 2.0)                          # z = up

    def test_airborne_uses_landed_state(self):
        d, fake, _ = flying()
        self.assertTrue(d.airborne())
        fake.landed_state = 1
        fake.publish()
        self.assertFalse(d.airborne())
        fake.armed = False
        fake.landed_state = 2
        fake.hb()
        fake.publish()
        self.assertFalse(d.airborne())

    def test_battery(self):
        d, _, _ = flying()
        self.assertEqual(d.battery_pct(), 87.0)

    def test_gimbal_gives_up_after_refusals(self):
        d, fake, clock = flying()
        for i in range(6):
            d.set_camera_pitch(10.0 + 5 * i)
        self.assertEqual(len(fake.commands_of(MAV.MAV_CMD_DO_MOUNT_CONTROL)), 3)

    def test_statustext_bytes_or_str(self):
        d, fake, _ = make()
        fake.text(4, b"Low battery\x00\x00")
        d._pump()
        self.assertEqual(d._last_text, "Low battery")


class TestOpenFlightController(unittest.TestCase):
    def open(self, fake, autopilot="auto", **kw):
        clock = FakeClock()
        return open_flight_controller("udpin:x", 57600, autopilot, connect_fn=lambda: fake, mav=MAV,
                                      timeout_s=2, clock=clock, **kw)

    def test_auto_picks_px4(self):
        d = self.open(FakePx4(autopilot=12), gimbal=False, start_thread=False)
        self.assertIsInstance(d, Px4Driver)
        self.assertFalse(d.gimbal)

    def test_auto_picks_ardupilot(self):
        d = self.open(FakePx4(autopilot=3))
        self.assertIs(type(d), MavlinkDriver)

    def test_mismatch_refuses_and_closes(self):
        fake = FakePx4(autopilot=3)
        with self.assertRaises(RuntimeError) as cm:
            self.open(fake, "px4")
        self.assertIn("ardupilot", str(cm.exception))
        self.assertTrue(fake.closed)

    def test_explicit_match(self):
        self.assertIsInstance(self.open(FakePx4(autopilot=12), "px4", start_thread=False), Px4Driver)

    def test_bad_choice(self):
        with self.assertRaises(ValueError):
            self.open(FakePx4(), "betaflight")

    def test_skips_gcs_and_companion_heartbeats(self):
        fake = FakePx4(autopilot=12)
        fake.queue.insert(0, Msg("HEARTBEAT", type=6, autopilot=8, custom_mode=0, base_mode=0))
        fake.queue.insert(0, Msg("HEARTBEAT", type=18, autopilot=8, custom_mode=0, base_mode=0))
        self.assertEqual(detect_autopilot(fake, MAV, 2, FakeClock()), 12)

    def test_no_heartbeat(self):
        class Silent:
            def __init__(self):
                self.clock, self.closed = FakeClock(), False

            def recv_match(self, **kw):
                self.clock.t += kw.get("timeout") or 0
                return None

            def close(self):
                self.closed = True

        link = Silent()
        with self.assertRaises(RuntimeError) as cm:
            open_flight_controller("udpin:x", 1, "auto", connect_fn=lambda: link, mav=MAV,
                                   timeout_s=3, clock=link.clock)
        self.assertIn("no heartbeat", str(cm.exception))
        self.assertTrue(link.closed)

    def test_unknown_firmware(self):
        with self.assertRaises(RuntimeError):
            self.open(FakePx4(autopilot=0))


if __name__ == "__main__":
    unittest.main()
