"""
tools/flight_summary.py: the one-screen summary of a main.py --log file.
Logs are written to a temp dir; nothing on the owner's disk is read.
"""

import io
import json
import os
import tempfile
import unittest

from tools import flight_summary


def step(t, mode, x=0.0, y=0.0, z=2.0, note="", battery=None):
    return {"t": t, "mode": mode, "note": note, "x": x, "y": y, "z": z, "yaw": 0.0,
            "yaw_cmd": 0.0, "fwd_cmd": 0.0, "right_cmd": 0.0, "up_cmd": 0.0,
            "gimbal": 0.0, "battery": battery, "seen": None, "ranges": None, "down": None}


FLIGHT = ([step(100.0 + i * 0.1, "EXPLORE", z=1.4 + 0.006 * i) for i in range(100)]      # 10 s patrol
          + [step(110.0 + i * 0.1, "TRACK", y=-0.3 * i, z=2.0 + 0.04 * i) for i in range(100)]
          + [step(120.0, "RETURN", y=-30.1, z=6.0, note="outside geofence")]
          + [step(125.0, "RETURN", y=-20.0, z=4.0, note="outside geofence")]
          + [step(160.0, "LAND", x=0.3, y=-0.4, z=2.0, note="outside geofence: home reached")])


class SummaryTests(unittest.TestCase):
    def test_one_line_per_mode_change_with_time_height_distance_and_note(self):
        lines = flight_summary.summarize(FLIGHT, "run.jsonl")
        table = [line for line in lines if line.lstrip()[:1].isdigit()]
        self.assertEqual([line.split()[1] for line in table], ["EXPLORE", "TRACK", "RETURN", "LAND"])
        self.assertEqual(table[2].split()[:4], ["0:20", "RETURN", "6.0", "30.1"])
        self.assertTrue(table[2].endswith("outside geofence"))
        self.assertEqual(lines[-1], "end: LAND, 2.0 m up, 0.5 m from home")

    def test_headline_numbers(self):
        lines = flight_summary.summarize(FLIGHT, "run.jsonl")
        self.assertTrue(lines[0].startswith("run.jsonl: 203 steps, 1:00 long, loop 3.4 steps/s"))
        self.assertIn("highest 6.0 m, farthest 30.1 m from home", lines)

    def test_missing_battery_is_called_out(self):
        lines = flight_summary.summarize(FLIGHT)
        self.assertTrue(any("battery: NEVER reported" in line for line in lines))

    def test_battery_when_reported(self):
        rows = [step(0.0, "EXPLORE", battery=97.0), step(1.0, "EXPLORE", battery=None),
                step(2.0, "RETURN", battery=29.0)]
        self.assertIn("battery 97 % -> 29 % (lowest 29 %)", flight_summary.summarize(rows))

    def test_empty_log(self):
        self.assertEqual(flight_summary.summarize([], "x.jsonl"), ["x.jsonl: no flight steps in it"])


class FileTests(unittest.TestCase):
    def test_a_line_cut_in_half_by_ctrl_c_is_skipped_and_counted(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.jsonl")
            with open(path, "w") as f:
                for row in FLIGHT[:3]:
                    f.write(json.dumps(row) + "\n")
                f.write('{"t": 100.3, "mode": "EXPL')          # the half line
            out = io.StringIO()
            self.assertEqual(flight_summary.main([path], out=out), 0)
        self.assertIn("3 steps", out.getvalue())
        self.assertIn("(1 unreadable line(s) skipped)", out.getvalue())

    def test_missing_file_is_a_message_not_a_traceback(self):
        out = io.StringIO()
        self.assertEqual(flight_summary.main(["/no/such/run.jsonl"], out=out), 1)
        self.assertIn("cannot read", out.getvalue())

    def test_no_arguments_prints_usage(self):
        out = io.StringIO()
        self.assertEqual(flight_summary.main([], out=out), 2)
        self.assertIn("usage", out.getvalue())


if __name__ == "__main__":
    unittest.main()
