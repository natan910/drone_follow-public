"""
tools/mav_watch.py: the read-only flight monitor that replaces MAVProxy's
`watch` in the SITL window. Hand-written fakes only: no pymavlink, no SITL.
"""

import csv
import io
import math
import os
import re
import tempfile
import unittest

from tools import mav_watch
from tools.mav_watch import Monitor, format_row, header

QUAD, ARDUPILOT, GCS, NOT_AN_AUTOPILOT = 2, 3, 6, 8   # MAVLink MAV_TYPE / MAV_AUTOPILOT values
MODES = {0: "STABILIZE", 4: "GUIDED", 6: "RTL", 9: "LAND"}

# SITL's default home, in MAVLink units (degrees * 1e7). Reference for the distance
# tests, independent of the code: 1e-7 deg of latitude is 1.113195 cm.
HOME_LAT, HOME_LON = -353632610, 1491652300
TEN_M_NORTH = 898   # 898 * 1.113195 cm = 9.9965 m
TEN_M_EAST = 1102   # the same, times cos(35.3633 deg) = 0.81548 -> 10.004 m


def mode_name(heartbeat):
    return MODES.get(heartbeat.custom_mode, f"Mode({heartbeat.custom_mode})")


class FakeMsg:
    def __init__(self, kind, src=1, comp=1, **fields):
        self._kind, self._src, self._comp = kind, src, comp
        self.__dict__.update(fields)

    def get_type(self):
        return self._kind

    def get_srcSystem(self):
        return self._src

    def get_srcComponent(self):
        return self._comp


def heartbeat(mode=4, armed=False, src=1, comp=1, vehicle_type=QUAD, autopilot=ARDUPILOT):
    return FakeMsg("HEARTBEAT", src, comp, type=vehicle_type, autopilot=autopilot, custom_mode=mode,
                   base_mode=1 | (128 if armed else 0))


def position(lat=HOME_LAT, lon=HOME_LON, alt_m=2.0, vn=0.0, ve=0.0, vd=0.0, hdg_deg=None, src=1, comp=1):
    """GLOBAL_POSITION_INT in its own units: 1e-7 deg, mm, cm/s (north, east, DOWN), centidegrees."""
    return FakeMsg("GLOBAL_POSITION_INT", src, comp, lat=lat, lon=lon, relative_alt=round(alt_m * 1000),
                   vx=round(vn * 100), vy=round(ve * 100), vz=round(vd * 100),
                   hdg=65535 if hdg_deg is None else round(hdg_deg * 100))


def attitude(yaw_deg=0.0, yaw_rate_dps=0.0):
    return FakeMsg("ATTITUDE", yaw=math.radians(yaw_deg), yawspeed=math.radians(yaw_rate_dps))


class FakeClock:
    def __init__(self, now=1_790_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def fed(*msgs):
    mon = Monitor(mode_name, FakeClock())
    for m in msgs:
        mon.feed(m)
    return mon


class SourceTests(unittest.TestCase):
    def test_nothing_counts_before_the_flight_controller_sends_a_heartbeat(self):
        mon = fed(position())
        self.assertIsNone(mon.vehicle)
        self.assertIsNone(mon.position)

    def test_ground_station_and_gimbal_heartbeats_are_not_the_flight_controller(self):
        mon = fed(heartbeat(src=255, vehicle_type=GCS),                     # MAVProxy itself
                  heartbeat(src=1, comp=154, autopilot=NOT_AN_AUTOPILOT))   # a gimbal
        self.assertIsNone(mon.vehicle)
        mon.feed(heartbeat())
        self.assertEqual(mon.vehicle, (1, 1))

    def test_messages_from_other_senders_are_ignored(self):
        mon = fed(heartbeat(), position(alt_m=2.0))
        mon.feed(position(alt_m=50.0, src=255))
        mon.feed(position(alt_m=60.0, comp=154))
        self.assertEqual(mon.alt_m, 2.0)


class HomeTests(unittest.TestCase):
    def test_home_is_where_it_armed(self):
        mon = fed(heartbeat(armed=False), position())            # sitting at HOME
        mon.feed(position(lon=HOME_LON + TEN_M_EAST))            # carried 10 m east, still disarmed
        mon.feed(heartbeat(armed=True))                          # arms here: this is home now
        mon.feed(position(lon=HOME_LON + TEN_M_EAST))
        self.assertAlmostEqual(mon.row()["home"], 0.0, places=3)
        mon.feed(position(lat=HOME_LAT + TEN_M_NORTH, lon=HOME_LON + TEN_M_EAST))
        self.assertAlmostEqual(mon.row()["home"], 10.0, delta=0.05)

    def test_distance_east_allows_for_the_latitude(self):
        mon = fed(heartbeat(armed=True), position(), position(lon=HOME_LON + TEN_M_EAST))
        self.assertAlmostEqual(mon.row()["home"], 10.0, delta=0.05)

    def test_home_position_from_the_flight_controller_overrides_the_guess(self):
        # monitor started mid-flight: its first guess is wherever the drone was
        mon = fed(heartbeat(armed=True), position(lat=HOME_LAT + TEN_M_NORTH))
        mon.feed(FakeMsg("HOME_POSITION", latitude=HOME_LAT, longitude=HOME_LON))
        self.assertAlmostEqual(mon.row()["home"], 10.0, delta=0.05)

    def test_no_gps_fix_yet_is_not_a_position(self):
        row = fed(heartbeat(), position(lat=0, lon=0)).row()
        self.assertIsNone(row["home"])
        self.assertIsNone(row["alt"])


class MotionSignTests(unittest.TestCase):
    """fwd / right / yawr must use DriveCommand's signs, so they compare
    directly with what main.py commanded."""

    def row_for(self, heading_deg, vn, ve):
        return fed(heartbeat(), position(vn=vn, ve=ve, hdg_deg=heading_deg)).row()

    def test_facing_east_and_moving_east_is_forward(self):
        row = self.row_for(90, 0.0, 1.0)
        self.assertAlmostEqual(row["fwd"], 1.0, places=6)
        self.assertAlmostEqual(row["right"], 0.0, places=6)

    def test_facing_east_and_moving_south_is_to_the_right(self):
        row = self.row_for(90, -1.0, 0.0)
        self.assertAlmostEqual(row["fwd"], 0.0, places=6)
        self.assertAlmostEqual(row["right"], 1.0, places=6)

    def test_facing_north_and_moving_west_is_to_the_left(self):
        self.assertAlmostEqual(self.row_for(0, 0.0, -1.0)["right"], -1.0, places=6)

    def test_climb_is_positive_going_up(self):
        # MAVLink's vz is positive DOWN; the column flips it
        self.assertAlmostEqual(fed(heartbeat(), position(vd=-0.5)).row()["climb"], 0.5, places=6)

    def test_yaw_rate_is_positive_clockwise_and_heading_runs_0_to_360(self):
        row = fed(heartbeat(), attitude(yaw_deg=-90.0, yaw_rate_dps=30.0)).row()
        self.assertAlmostEqual(row["yawr"], 30.0, places=6)
        self.assertAlmostEqual(row["hdg"], 270.0, places=6)

    def test_attitude_heading_wins_over_the_slower_position_heading(self):
        mon = fed(heartbeat(), attitude(yaw_deg=10.0), position(hdg_deg=90))
        self.assertAlmostEqual(mon.row()["hdg"], 10.0, places=6)


class EventTests(unittest.TestCase):
    def test_mode_and_arming_changes_are_reported_once(self):
        mon = Monitor(mode_name, FakeClock())
        self.assertEqual(mon.feed(heartbeat(mode=0)), ["mode STABILIZE"])
        self.assertEqual(mon.feed(heartbeat(mode=0)), [])
        self.assertEqual(mon.feed(heartbeat(mode=4, armed=True)), ["mode STABILIZE -> GUIDED", "ARMED"])
        self.assertEqual(mon.feed(heartbeat(mode=9, armed=True)), ["mode GUIDED -> LAND"])
        self.assertEqual(mon.feed(heartbeat(mode=9, armed=False)), ["DISARMED"])

    def test_flight_controller_text_is_reported(self):
        mon = fed(heartbeat())
        self.assertEqual(mon.feed(FakeMsg("STATUSTEXT", severity=4, text="PreArm: Need Position Estimate")),
                         ["FC: PreArm: Need Position Estimate"])
        self.assertEqual(mon.feed(FakeMsg("STATUSTEXT", severity=6, text=b"Fence breach\x00\x00")),
                         ["FC: Fence breach"])

    def test_events_wait_for_the_csv_until_taken(self):
        mon = fed(heartbeat(mode=4))
        mon.feed(heartbeat(mode=9))
        self.assertEqual(mon.take_events(), ["mode GUIDED", "mode GUIDED -> LAND"])
        self.assertEqual(mon.take_events(), [])

    def test_unknown_battery_is_none(self):
        mon = fed(heartbeat(), FakeMsg("SYS_STATUS", battery_remaining=-1))
        self.assertIsNone(mon.row()["bat"])
        mon.feed(FakeMsg("SYS_STATUS", battery_remaining=57))
        self.assertEqual(mon.row()["bat"], 57.0)


class FormatTests(unittest.TestCase):
    def test_unknown_values_print_as_a_dash(self):
        tokens = format_row(fed(heartbeat()).row()).split()
        self.assertRegex(tokens[0], r"^\d\d:\d\d:\d\d$")
        self.assertEqual(tokens[1], "GUIDED")
        self.assertEqual(tokens[2:], ["-"] * 9)   # disarmed, then 8 unknown numbers

    def test_zero_never_prints_as_minus_zero(self):
        line = format_row({"climb": -0.0, "fwd": -1e-17, "yawr": -0.3, "alt": -0.01, "right": -0.05})
        self.assertIn(" +0.00  +0.00  -0.05    +0 ", line)
        self.assertNotIn("-0.0 ", line)

    def test_columns_line_up_with_the_header(self):
        mon = fed(heartbeat(armed=True), position(),
                  position(lat=HOME_LAT + TEN_M_NORTH, alt_m=12.3, vn=0.6, vd=-0.4, hdg_deg=87),
                  attitude(yaw_deg=87, yaw_rate_dps=-12), FakeMsg("SYS_STATUS", battery_remaining=57))
        line = format_row(mon.row())
        self.assertEqual(len(line), len(header()))
        self.assertRegex(line, r"^\d\d:\d\d:\d\d GUIDED {4}ARMED ")


class FakeLink:
    """Plays a script of messages (None = nothing arrived before the timeout),
    moving the fake clock on by `step_s` per call; when the script runs out it
    behaves like the owner pressing Ctrl+C."""

    def __init__(self, clock, script, step_s=0.25):
        self.clock, self.script, self.step_s = clock, list(script), step_s
        self.closed = False

    def recv_match(self, blocking=True, timeout=None):
        self.clock.now += self.step_s
        if not self.script:
            raise KeyboardInterrupt
        return self.script.pop(0)

    def close(self):
        self.closed = True


class MainTests(unittest.TestCase):
    def run_main(self, script, argv=()):
        clock = FakeClock()
        link = FakeLink(clock, script)
        out = io.StringIO()
        code = mav_watch.main(list(argv), connect=lambda url, baud: (link, mode_name), out=out, clock=clock)
        return code, out.getvalue(), link

    @staticmethod
    def flight():
        """Arms in GUIDED, flies north at 0.6 m/s for 3 s (12 steps of 0.25 s), then LAND."""
        script = [heartbeat(mode=4, armed=True), position()]
        for i in range(1, 13):
            script.append(position(lat=HOME_LAT + round(i * 0.015 * TEN_M_NORTH), vn=0.6, hdg_deg=0))
        script.append(heartbeat(mode=9, armed=True))
        return script

    def test_prints_lines_and_events_and_saves_the_same_lines_to_csv(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.csv")
            code, text, link = self.run_main(self.flight(), ["--csv", path])
            with open(path, newline="") as f:
                records = list(csv.DictReader(f))
        self.assertEqual(code, 0)
        self.assertTrue(link.closed)
        self.assertEqual(text.count(header()), 1)
        for event in (">> mode GUIDED", ">> ARMED", ">> mode GUIDED -> LAND"):
            self.assertIn(event, text)
        lines = [line for line in text.splitlines() if re.match(r"^\d\d:\d\d:\d\d (GUIDED|LAND)", line)]
        self.assertGreaterEqual(len(lines), 3)
        self.assertEqual(len(records), len(lines))
        self.assertEqual(records[0]["event"], "mode GUIDED | ARMED")
        self.assertEqual(records[0]["mode"], "GUIDED")
        self.assertAlmostEqual(float(records[-1]["fwd"]), 0.6, places=3)
        self.assertGreater(float(records[-1]["home"]), 1.0)   # it flew away from home
        self.assertIn("Stopped. Saved", text)

    def test_says_once_when_no_flight_controller_is_heard(self):
        code, text, _ = self.run_main([None] * 40)   # 10 s of silence
        self.assertEqual(code, 0)
        self.assertEqual(text.count("No heartbeat"), 1)
        self.assertNotIn(header(), text)

    def test_busy_port_is_a_clear_message_not_a_traceback(self):
        def busy(url, baud):
            raise OSError(48, "Address already in use")
        out = io.StringIO()
        self.assertEqual(mav_watch.main([], connect=busy, out=out, clock=FakeClock()), 1)
        self.assertIn("udpin:127.0.0.1:14550", out.getvalue())
        self.assertIn("Address already in use", out.getvalue())

    def test_missing_pymavlink_is_a_clear_message(self):
        def missing(url, baud):
            raise ImportError("No module named 'pymavlink'")
        out = io.StringIO()
        self.assertEqual(mav_watch.main([], connect=missing, out=out, clock=FakeClock()), 1)
        self.assertIn("pip install pymavlink", out.getvalue())


if __name__ == "__main__":
    unittest.main()
