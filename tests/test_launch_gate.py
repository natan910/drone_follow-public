"""HANDOFF "Must fix" 1-4: launch gate, battery unknown/low, altitude ceiling,
return home with no route. Fakes only (no hardware, no mocks)."""

import unittest

import numpy as np

from autonomy.autopilot import Autopilot
from autonomy.launch import LaunchGate, launch_problems, link_ok
from comms.phone_server import CommandBox, CommandError, parse_command
from config import AppConfig, SafetyConfig
from control.base_driver import BaseDriver
from datatypes import DriveCommand, Mode, Observation, Pose, Task
from navigation.patrol import Route
from perception.stream_input import FrameSource
from platforms.real import RealPlatform
from safety.supervisor import SafetyAction, SafetySupervisor


# ---- fakes ------------------------------------------------------------------

class FakeCamera(FrameSource):
    def read(self):
        return np.zeros((48, 64, 3), np.uint8)

    def release(self):
        pass


class NoOneMatcher:
    """Never sees anybody. Enough of the matcher interface for RealPlatform."""
    has_target = False

    def find(self, frame):
        return None


class FakeDriver(BaseDriver):
    """A vehicle on the ground. battery / pose settable; counts arm calls."""

    def __init__(self, battery=80.0, pose=Pose(0.0, 0.0, 0.0, 0.0)):
        self.battery, self._pose, self.armed = battery, pose, 0

    def connect(self): pass
    def arm(self): self.armed += 1
    def takeoff(self): pass
    def send(self, cmd): pass
    def stop(self): pass
    def land(self): pass
    def disconnect(self): pass
    def pose(self): return self._pose
    def battery_pct(self): return self.battery


class HeartbeatDriver(FakeDriver):
    """Looks like MavlinkDriver/Px4Driver to link_ok(): keeps the last heartbeat time."""

    def __init__(self, now=100.0, last=99.5, **kw):
        super().__init__(**kw)
        self.now, self._last_heartbeat, self.heartbeat_timeout_s = now, last, 2.0
        self._clock = lambda: self.now


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def platform_with(driver=None, commands=None):
    return RealPlatform(FakeCamera(), NoOneMatcher(), driver or FakeDriver(), commands=commands)


def obs(now=0.0, z=0.0, x=0.0, battery=80.0, task=None, permitted=True):
    return Observation(now=now, pose=Pose(x, 0.0, 0.0, z), battery_pct=battery, task=task,
                       autonomy_permitted=permitted)


def dry_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.safety.require_scan = False     # no range sensors in these tests
    return cfg


# ---- the command ------------------------------------------------------------

class LaunchCommandParsingTests(unittest.TestCase):
    def test_launch_true_is_a_command(self):
        self.assertEqual(parse_command({"launch": True}), {"launch": True})
        self.assertEqual(parse_command({"launch": True, "task": "patrol"}), {"task": "PATROL", "launch": True})

    def test_anything_but_true_is_refused(self):
        for bad in (False, "yes", 1, None):
            with self.assertRaises(CommandError) as e:
                parse_command({"launch": bad})
            self.assertEqual(e.exception.code, 422)


class PreflightTests(unittest.TestCase):
    def test_unknown_battery_blocks_only_when_required(self):
        required = SafetyConfig(require_battery=True)
        self.assertIn("unknown", " ".join(launch_problems(None, True, required)))
        self.assertEqual(launch_problems(None, True, SafetyConfig()), [])

    def test_battery_at_or_below_the_return_level_blocks(self):
        cfg = SafetyConfig()                           # battery_return_pct 30
        self.assertEqual(len(launch_problems(30.0, True, cfg)), 1)
        self.assertEqual(len(launch_problems(0.0, True, cfg)), 1)
        self.assertEqual(launch_problems(31.0, True, cfg), [])

    def test_no_heartbeat_blocks(self):
        self.assertIn("heartbeat", launch_problems(80.0, False, SafetyConfig())[0])

    def test_link_ok_reads_the_mavlink_heartbeat_age(self):
        self.assertTrue(link_ok(FakeDriver()))                       # print / native-sim: no link to lose
        self.assertTrue(link_ok(HeartbeatDriver(now=100.0, last=99.5)))
        self.assertFalse(link_ok(HeartbeatDriver(now=100.0, last=90.0)))


# ---- RealPlatform: launch command and launch-relative pose -----------------

class RealPlatformLaunchTests(unittest.TestCase):
    def setUp(self):
        self.commands = CommandBox()
        self.driver = FakeDriver()
        self.platform = platform_with(self.driver, self.commands)
        self.problems = []
        self.platform.launch_check = lambda: list(self.problems)

    def test_an_accepted_launch_is_answered_and_picked_up_once(self):
        reply = self.commands.submit({"launch": True})
        self.platform.observe()
        self.assertEqual(reply.result(timeout=0), {"ok": True, "launch": True})
        self.assertTrue(self.platform.take_launch_request())
        self.assertFalse(self.platform.take_launch_request())

    def test_a_refused_launch_says_why_and_changes_nothing(self):
        self.problems = ["battery level unknown"]
        reply = self.commands.submit({"launch": True, "task": "PATROL"})
        o = self.platform.observe()
        with self.assertRaises(ValueError) as e:
            reply.result(timeout=0)
        self.assertIn("battery level unknown", str(e.exception))
        self.assertIsNone(o.task)                     # the task in the same command was not applied
        self.assertFalse(self.platform.take_launch_request())

    def test_no_gate_or_already_flying_refuses(self):
        self.platform.launch_check = None
        reply = self.commands.submit({"launch": True})
        self.platform.observe()
        self.assertRaises(ValueError, reply.result, 0)
        self.platform.launch_check = lambda: []
        self.platform.launched = True
        reply = self.commands.submit({"launch": True})
        self.platform.observe()
        with self.assertRaises(ValueError) as e:
            reply.result(timeout=0)
        self.assertIn("already", str(e.exception))

    def test_pose_is_measured_from_the_launch_point(self):
        # the flight controller's local origin is 4 m away from where this flight starts
        self.driver._pose = Pose(4.0, 3.0, 0.5, 0.2)
        self.platform.set_home(self.driver.pose())
        self.assertEqual(self.platform.observe().pose, Pose(0.0, 0.0, 0.5, 0.0))
        self.driver._pose = Pose(5.0, 1.0, 0.7, 12.2)
        p = self.platform.observe().pose
        for got, want in zip((p.x, p.y, p.yaw, p.z), (1.0, -2.0, 0.7, 12.0)):
            self.assertAlmostEqual(got, want)


# ---- the gate loop ----------------------------------------------------------

class LaunchGateTests(unittest.TestCase):
    def make(self, auto=False, battery=80.0, require_battery=False, clock=None):
        cfg = dry_cfg()
        cfg.safety.require_battery = require_battery
        self.commands = CommandBox()
        self.driver = FakeDriver(battery=battery)
        self.platform = platform_with(self.driver, self.commands)
        self.autopilot = Autopilot(cfg)
        self.said = []
        self.gate = LaunchGate(cfg.safety, self.driver, auto=auto, clock=clock or Clock(),
                               say=self.said.append)
        self.statuses = []

    def run_gate(self, max_loops=50, each=None):
        """wait() with an on_step that records statuses and stops after max_loops."""
        def on_step(o, status):
            self.statuses.append(status)
            if each is not None:
                each(len(self.statuses))
            return None if len(self.statuses) < max_loops else False
        return self.gate.wait(self.platform, self.autopilot, on_step)

    def test_waits_on_the_ground_until_launch_is_tapped(self):
        self.make()
        replies = []

        def tap(n):
            if n == 5:
                replies.append(self.commands.submit({"launch": True}))
        self.assertTrue(self.run_gate(each=tap))
        self.assertEqual(replies[0].result(timeout=0)["launch"], True)
        self.assertEqual({s["state"] for s in self.statuses[:5]}, {"waiting"})
        self.assertEqual(self.statuses[-1]["state"], "launching")
        self.assertEqual(self.driver.armed, 0)                 # the gate never arms; main.py does, after it

    def test_stopping_before_launch_returns_false(self):
        self.make()
        self.assertFalse(self.run_gate(max_loops=10))

    def test_launch_is_refused_with_unknown_battery_on_a_real_flight_controller(self):
        self.make(battery=None, require_battery=True)
        replies = []
        self.assertFalse(self.run_gate(max_loops=10, each=lambda n: n == 2 and replies.append(
            self.commands.submit({"launch": True}))))
        self.assertRaises(ValueError, replies[0].result, 0)
        self.assertFalse(self.statuses[-1]["ready"])
        self.assertIn("unknown", " ".join(self.statuses[-1]["problems"]))

    def test_low_battery_refuses_launch(self):
        self.make(battery=25.0)
        replies = []
        self.assertFalse(self.run_gate(max_loops=6, each=lambda n: n == 1 and replies.append(
            self.commands.submit({"launch": True}))))
        with self.assertRaises(ValueError) as e:
            replies[0].result(timeout=0)
        self.assertIn("25 %", str(e.exception))

    def test_auto_launch_waits_for_preflight_then_goes(self):
        self.make(auto=True, battery=None, require_battery=True)
        self.assertFalse(self.run_gate(max_loops=5))           # battery unknown: never launches
        self.driver.battery = 90.0
        self.statuses = []
        self.assertTrue(self.run_gate(max_loops=5))
        self.assertEqual(len(self.statuses), 1)

    def test_after_a_refused_arming_auto_launch_retries_later(self):
        clock = Clock(100.0)
        self.make(auto=True, clock=clock)
        self.gate.failed("the flight controller refused to arm")
        self.assertFalse(self.run_gate(max_loops=5))           # inside retry_s: no new attempt
        self.assertIn("refused to arm", self.statuses[-1]["last_error"])
        clock.t += 6.0
        self.statuses = []
        self.assertTrue(self.run_gate(max_loops=5))

    def test_tasks_sent_before_launch_are_not_lost(self):
        self.make()

        def send(n):
            if n == 1:
                self.commands.submit({"task": "PATROL", "hover_height_m": 0.8})
            if n == 3:
                self.commands.submit({"launch": True})
        self.assertTrue(self.run_gate(each=send))
        self.assertEqual(self.autopilot.task, Task.PATROL)
        self.assertEqual(self.autopilot.cfg.control.hover_height_above_target_m, 0.8)


# ---- supervisor: battery unknown, above the ceiling ---------------------------

class SupervisorLimitTests(unittest.TestCase):
    def test_battery_unknown_in_flight_returns_after_a_while_when_required(self):
        sup = SafetySupervisor(SafetyConfig(require_scan=False, require_battery=True,
                                            battery_unknown_return_s=10.0))
        self.assertEqual(sup.check(obs(0.0, battery=None))[0], SafetyAction.OK)
        self.assertEqual(sup.check(obs(9.0, battery=None))[0], SafetyAction.OK)
        action, why = sup.check(obs(10.5, battery=None))
        self.assertEqual((action, why), (SafetyAction.RETURN, "battery level unknown"))
        self.assertEqual(sup.check(obs(11.0, battery=80.0))[0], SafetyAction.RETURN)   # latched

    def test_a_known_reading_restarts_the_count(self):
        sup = SafetySupervisor(SafetyConfig(require_scan=False, require_battery=True,
                                            battery_unknown_return_s=10.0))
        sup.check(obs(0.0, battery=None))
        sup.check(obs(8.0, battery=70.0))
        self.assertEqual(sup.check(obs(15.0, battery=None))[0], SafetyAction.OK)

    def test_unknown_battery_is_fine_for_drivers_that_never_report_it(self):
        sup = SafetySupervisor(SafetyConfig(require_scan=False))       # require_battery False
        self.assertEqual(sup.check(obs(0.0, battery=None))[0], SafetyAction.OK)
        self.assertEqual(sup.check(obs(500.0, battery=None))[0], SafetyAction.OK)

    def test_well_above_the_ceiling_returns(self):
        cfg = SafetyConfig(require_scan=False)                         # 15 m, margin 3 m
        self.assertEqual(SafetySupervisor(cfg).check(obs(z=17.9))[0], SafetyAction.OK)
        self.assertEqual(SafetySupervisor(cfg).check(obs(z=18.1)),
                         (SafetyAction.RETURN, "above altitude ceiling"))


# ---- autopilot: the ceiling on every command ---------------------------------

class ClimbingAutopilot(Autopilot):
    """Whatever it 'decides', it wants to climb at full rate (like FOLLOW with the
    open-loop webcam in SITL)."""

    def _behave(self, obs, out, action, why):
        return Mode.TRACK, DriveCommand(up_mps=0.5), "", None


def climbs(z, steps=30):
    ap = ClimbingAutopilot(dry_cfg())
    return [ap.step(obs(now=0.1 * k, z=z)).cmd.up_mps for k in range(steps)]


class CeilingTests(unittest.TestCase):
    def test_climbs_normally_well_below_the_ceiling(self):
        self.assertAlmostEqual(climbs(5.0)[-1], 0.5, places=2)

    def test_climb_slows_near_the_ceiling(self):
        up = climbs(14.8)
        self.assertLessEqual(max(up), 0.2 + 1e-9)
        self.assertGreater(up[-1], 0.0)

    def test_never_climbs_at_the_ceiling(self):
        self.assertLessEqual(max(climbs(15.0)), 0.0)

    def test_comes_back_down_when_above_it(self):
        up = climbs(16.0)
        self.assertLessEqual(max(up), 0.0)
        self.assertAlmostEqual(up[-1], -AppConfig().safety.ceiling_descend_mps, places=2)

    def test_no_climb_even_when_the_last_command_was_climbing(self):
        ap = ClimbingAutopilot(dry_cfg())
        for k in range(20):                          # climbing hard at 10 m...
            ap.step(obs(now=0.1 * k, z=10.0))
        cmd = ap.step(obs(now=2.0, z=15.2)).cmd      # ...then a jump in height reading
        self.assertLessEqual(cmd.up_mps, 0.0)

    def test_the_ceiling_is_a_setting(self):
        cfg = dry_cfg()
        cfg.safety.max_altitude_m = 30.0
        ap = ClimbingAutopilot(cfg)
        up = [ap.step(obs(now=0.1 * k, z=16.0)).cmd.up_mps for k in range(30)]
        self.assertAlmostEqual(up[-1], 0.5, places=2)


# ---- return home ------------------------------------------------------------

class FakePlanner:
    def __init__(self, route):
        self.route = route

    def route_home(self, grid, pose, now, home=(0.0, 0.0)):
        return self.route

    def update(self, *a, **k):
        return None

    def reset(self):
        pass


def returning(route, x):
    ap = Autopilot(dry_cfg())
    ap.planner = FakePlanner(route)
    return ap.step(obs(x=x, z=2.0, task=Task.RETURN))


class ReturnHomeTests(unittest.TestCase):
    def test_no_route_home_flies_straight_home_instead_of_landing_here(self):
        d = returning(None, x=4.1)
        self.assertEqual(d.mode, Mode.RETURN)
        self.assertIn("no route home", d.note)
        self.assertIn("4.1 m", d.note)

    def test_no_route_but_already_home_lands(self):
        d = returning(None, x=0.3)
        self.assertEqual(d.mode, Mode.LAND)
        self.assertIn("home reached (0.3 m)", d.note)

    def test_arrived_lands_and_says_how_far_from_home(self):
        d = returning(Route([(0.0, 0.0)], (0.0, 0.0), Mode.RETURN, arrived=True), x=0.4)
        self.assertEqual(d.mode, Mode.LAND)
        self.assertEqual(d.note, "operator requested return: home reached (0.4 m)")

    def test_on_the_way_it_returns(self):
        d = returning(Route([(0.0, 0.0)], (0.0, 0.0), Mode.RETURN), x=6.0)
        self.assertEqual((d.mode, d.note), (Mode.RETURN, "operator requested return"))


class MainFlagTests(unittest.TestCase):
    def test_auto_launch_flag(self):
        import main
        self.assertFalse(main.parse_args([]).auto_launch)
        self.assertTrue(main.parse_args(["--auto-launch"]).auto_launch)


if __name__ == "__main__":
    unittest.main()
