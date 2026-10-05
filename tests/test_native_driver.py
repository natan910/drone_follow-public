"""NativeDriver: the BaseDriver that flies flightcore instead of ArduPilot.

Two kinds of test:
  * frame/sign mapping against a recording fake core (no physics), where a wrong sign would hide;
  * whole flights in flightcore's simulator, in virtual time driven by a fake clock (no sleeping, no threads).
No unittest.mock. The simulator is deterministic (fixed seed).
"""
import math
import unittest

from control.base_driver import BaseDriver
from control.native_driver import NativeDriver
from datatypes import DriveCommand, Pose
from flightcore.autopilot import FlightCore
from flightcore.mathutil import quat_from_euler
from flightcore.sim import SensorSpec
from flightcore.state import NavEstimate
from flightcore.supervisor import Mode


class Clock:
    """Fake time source: only moves when the test says so."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class Brain:
    """Stands in for the main loop: every `period` seconds send a command and let the driver sync."""

    def __init__(self, driver, clock, period=0.1):
        self.d, self.clock, self.period = driver, clock, period

    def send(self, cmd, seconds):
        for _ in range(int(round(seconds / self.period))):
            self.clock.advance(self.period)
            self.d.send(cmd)

    def idle(self, seconds):
        """Time passes, the brain sends nothing (it only looks at the pose)."""
        for _ in range(int(round(seconds / self.period))):
            self.clock.advance(self.period)
            self.d.pose()


def flown(**kw):
    """A simulated driver that is connected, armed and hovering at its takeoff height."""
    clock = Clock()
    d = NativeDriver.simulated(clock=clock, **kw)
    d.connect()
    d.arm()
    d.takeoff()
    return d, clock, Brain(d, clock)


# ---------------------------------------------------------------------------------- mapping, no physics
class FakeRunner:
    error = None

    def __init__(self):
        self.started = self.stopped = 0
        self.syncs = 0

    def start(self): self.started += 1
    def stop(self): self.stopped += 1
    def sync(self): self.syncs += 1
    def wait(self, seconds): self.syncs += 1


class RecordingCore(FlightCore):
    """Real FlightCore with the command entry point recorded and the mode pinned."""

    def __init__(self, mode=Mode.HOLD):
        super().__init__()
        self._fake_mode = mode
        self.velocity_calls = []

    @property
    def mode(self):
        return self._fake_mode

    def set_velocity_body(self, vx, vy, vz, yaw_rate):
        self.velocity_calls.append((vx, vy, vz, yaw_rate))
        return True


class Mapping(unittest.TestCase):
    def driver(self, mode=Mode.HOLD):
        core = RecordingCore(mode)
        return NativeDriver(core, FakeRunner()), core

    def test_velocity_command_frames_and_signs(self):
        d, core = self.driver()
        d.send(DriveCommand(yaw_rate_dps=30.0, forward_mps=1.0, right_mps=0.5, up_mps=0.25))
        (vx, vy, vz, wz), = core.velocity_calls
        self.assertEqual((vx, vy), (1.0, 0.5))                 # forward, right unchanged
        self.assertEqual(vz, -0.25)                            # NED: up is negative down
        self.assertAlmostEqual(wz, math.radians(30.0))         # deg/s -> rad/s, positive = clockwise

    def test_pose_is_east_north_up_and_yaw_clockwise_from_north(self):
        d, core = self.driver()
        core.nav = NavEstimate(p=[3.0, 5.0, -2.0], q=quat_from_euler(0.0, 0.0, 0.5))   # N=3 E=5 D=-2
        p = d.pose()
        self.assertIsInstance(p, Pose)
        self.assertEqual((p.x, p.y, p.z), (5.0, 3.0, 2.0))     # x east, y north, z up
        self.assertAlmostEqual(p.yaw, 0.5)

    def test_battery_is_a_percentage_or_none(self):
        d, core = self.driver()
        self.assertIsNone(d.battery_pct())
        core.batt = 0.42
        self.assertAlmostEqual(d.battery_pct(), 42.0)

    def test_autonomy_only_in_hold_or_guided_and_without_a_latched_failsafe(self):
        for mode, expect in ((Mode.HOLD, True), (Mode.GUIDED, True), (Mode.DISARMED, False), (Mode.ARMED, False),
                             (Mode.TAKEOFF, False), (Mode.GOTO, False), (Mode.LAND, False), (Mode.RTL, False),
                             (Mode.KILLED, False)):
            d, core = self.driver(mode)
            self.assertEqual(d.autonomy_permitted(), expect, mode)
        d, core = self.driver(Mode.HOLD)
        core.sup.latched = "land"
        self.assertFalse(d.autonomy_permitted())

    def test_send_and_stop_are_dropped_when_autonomy_is_not_permitted(self):
        d, core = self.driver(Mode.LAND)
        d.send(DriveCommand(forward_mps=1.0))
        d.stop()
        self.assertEqual(core.velocity_calls, [])

    def test_stop_sends_zero_velocity(self):
        d, core = self.driver()
        d.stop()
        self.assertEqual(core.velocity_calls, [(0.0, 0.0, -0.0, 0.0)])

    def test_every_call_syncs_the_runner_first(self):
        d, core = self.driver()
        n = d.runner.syncs
        d.pose(); d.battery_pct(); d.send(DriveCommand()); d.land()
        self.assertGreaterEqual(d.runner.syncs - n, 4)

    def test_a_dead_runner_is_not_swallowed(self):
        class Dead(FakeRunner):
            def sync(self):
                raise RuntimeError("the flight loop died")

        d = NativeDriver(RecordingCore(), Dead())
        for call in (d.pose, d.battery_pct, d.autonomy_permitted, lambda: d.send(DriveCommand())):
            with self.assertRaises(RuntimeError):
                call()

    def test_is_a_base_driver_and_camera_pitch_is_a_noop(self):
        d, core = self.driver()
        self.assertIsInstance(d, BaseDriver)
        d.set_camera_pitch(45.0)


# ---------------------------------------------------------------------------------- whole flights
class Startup(unittest.TestCase):
    def test_connect_waits_for_the_estimator_and_gives_up_if_it_never_aligns(self):
        d = NativeDriver.simulated(clock=Clock(), ready_timeout_s=0.3)          # alignment needs ~1 s
        with self.assertRaisesRegex(RuntimeError, "estimator"):
            d.connect()

    def test_arm_refusal_carries_the_flight_cores_reasons(self):
        d = NativeDriver.simulated(clock=Clock(), spec=SensorSpec(battery_start=0.1), ready_timeout_s=2.0)
        d.connect()
        with self.assertRaisesRegex(RuntimeError, "battery_low"):
            d.arm()
        self.assertFalse(d.core.armed)

    def test_takeoff_needs_arming_first(self):
        d = NativeDriver.simulated(clock=Clock())
        d.connect()
        with self.assertRaisesRegex(RuntimeError, "takeoff refused"):
            d.takeoff()

    def test_connect_arm_takeoff_reach_a_hover(self):
        d, clock, brain = flown()
        self.assertEqual(d.core.mode, Mode.HOLD)
        p = d.pose()
        self.assertAlmostEqual(p.z, d.takeoff_alt_m, delta=0.4)
        self.assertGreater(d.battery_pct(), 90.0)
        self.assertTrue(d.autonomy_permitted())
        self.assertFalse(d.airborne())                                   # simulated: never a real airborne vehicle
        d.real_vehicle = True
        self.assertTrue(d.airborne())                                    # same vehicle, declared real


class Motion(unittest.TestCase):
    def test_forward_right_up_move_the_pose_the_right_way(self):
        d, clock, brain = flown()
        p0 = d.pose()
        brain.send(DriveCommand(forward_mps=1.0), 4.0)                   # heading north at start
        p1 = d.pose()
        self.assertGreater(p1.y - p0.y, 2.0)                             # y = north
        self.assertLess(abs(p1.x - p0.x), 0.5)
        brain.send(DriveCommand(right_mps=1.0), 4.0)
        p2 = d.pose()
        self.assertGreater(p2.x - p1.x, 2.0)                             # right of north = east = x
        self.assertLess(abs(p2.y - p1.y), 0.7)
        brain.send(DriveCommand(up_mps=0.4), 2.0)
        self.assertGreater(d.pose().z - p2.z, 0.5)

    def test_yaw_rate_is_clockwise_and_forward_follows_the_heading(self):
        d, clock, brain = flown()
        brain.send(DriveCommand(yaw_rate_dps=30.0), 3.0)                 # ~90 degrees clockwise: facing east
        p1 = d.pose()
        self.assertAlmostEqual(p1.yaw, math.radians(90.0), delta=math.radians(20.0))
        brain.send(DriveCommand(), 1.0)                                  # let the turn finish
        p2 = d.pose()
        brain.send(DriveCommand(forward_mps=1.0), 3.0)
        p3 = d.pose()
        self.assertGreater(p3.x - p2.x, 1.5)                             # forward while facing east = +x
        self.assertLess(abs(p3.y - p2.y), 0.8)

    def test_stop_holds_position(self):
        d, clock, brain = flown()
        brain.send(DriveCommand(forward_mps=1.5), 4.0)
        d.stop()
        brain.idle(1.5)
        a = d.pose()
        brain.idle(2.0)
        b = d.pose()
        self.assertLess(math.hypot(b.x - a.x, b.y - a.y), 0.25)
        self.assertLess(abs(b.z - a.z), 0.2)


class Failsafes(unittest.TestCase):
    def test_a_silent_brain_gets_hold_then_land_and_no_longer_permission(self):
        d, clock, brain = flown()
        brain.send(DriveCommand(forward_mps=0.5), 1.0)
        brain.idle(1.5)                                                  # setpoint_timeout_s = 0.5: now holding
        a = d.pose()
        brain.idle(1.0)
        b = d.pose()
        self.assertLess(math.hypot(b.x - a.x, b.y - a.y), 0.25)          # link-loss hold: not drifting away
        self.assertTrue(d.autonomy_permitted())                          # a new command would resume
        brain.idle(5.0)                                                  # hold_before_land_s = 5
        self.assertEqual(d.core.mode, Mode.LAND)
        self.assertFalse(d.autonomy_permitted())
        d.send(DriveCommand(forward_mps=1.0))                            # dropped: the failsafe is latched
        self.assertEqual(d.core.mode, Mode.LAND)

    def test_return_to_launch_takes_the_controls_from_the_brain(self):
        d, clock, brain = flown()
        brain.send(DriveCommand(forward_mps=1.0), 2.0)
        d.return_to_launch()
        self.assertEqual(d.core.mode, Mode.RTL)
        self.assertFalse(d.autonomy_permitted())

    def test_context_manager_lands_before_the_flight_loop_stops(self):
        clock = Clock()
        d = NativeDriver.simulated(clock=clock)
        with d:                                                          # connect ... stop, land, disconnect
            d.arm()
            d.takeoff()
            Brain(d, clock).send(DriveCommand(forward_mps=1.0), 3.0)
            self.assertTrue(d.core.armed)
        self.assertFalse(d.core.armed)                                   # disconnect waited for touchdown
        self.assertEqual(d.core.mode, Mode.DISARMED)
        self.assertFalse(d.world.plant.crashed)
        self.assertLess(d.world.plant.max_impact, 1.5)                   # m/s at touchdown

    def test_disconnect_gives_up_if_it_cannot_land(self):
        d, clock, brain = flown(land_timeout_s=0.5)
        with self.assertLogs("control.native_driver", level="ERROR") as logs:
            d.disconnect()                                               # never asked to land: still hovering
        self.assertIn("still armed", logs.output[0])
        self.assertTrue(d.core.armed)


if __name__ == "__main__":
    unittest.main()
