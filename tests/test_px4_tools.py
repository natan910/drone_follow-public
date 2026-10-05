"""
PX4 additions to tools/mav_watch.py (height from LOCAL_POSITION_NED) and
tools/preflight_check.py (firmware check on the MAVLink link). Fakes only.
"""

import unittest

from tools import mav_watch, preflight_check


class Msg:
    def __init__(self, kind, src=(1, 1), **fields):
        self._kind, self._src = kind, src
        self.__dict__.update(fields)

    def get_type(self):
        return self._kind

    def get_srcSystem(self):
        return self._src[0]

    def get_srcComponent(self):
        return self._src[1]


def heartbeat(autopilot, armed=False):
    return Msg("HEARTBEAT", type=2, autopilot=autopilot, base_mode=129 if armed else 1, custom_mode=0)


def global_pos(relative_alt_m):
    return Msg("GLOBAL_POSITION_INT", lat=450000000, lon=90000000, relative_alt=int(relative_alt_m * 1000),
               vx=0, vy=0, vz=0, hdg=0)


class TestMavWatchPx4Height(unittest.TestCase):
    def monitor(self):
        return mav_watch.Monitor(lambda msg: "MODE", clock=lambda: 0.0)

    def test_px4_height_is_local_z_relative_to_arming(self):
        m = self.monitor()
        m.feed(heartbeat(12))
        m.feed(Msg("LOCAL_POSITION_NED", z=-0.5))
        m.feed(heartbeat(12, armed=True))
        m.feed(Msg("LOCAL_POSITION_NED", z=-0.4))        # where it armed
        m.feed(Msg("LOCAL_POSITION_NED", z=-1.9))
        m.feed(global_pos(2.9))                          # PX4 relative_alt after a home correction jump
        self.assertAlmostEqual(m.row()["alt"], 1.5)

    def test_px4_before_arming_uses_first_local_z(self):
        m = self.monitor()
        m.feed(heartbeat(12))
        m.feed(Msg("LOCAL_POSITION_NED", z=-0.5))
        m.feed(Msg("LOCAL_POSITION_NED", z=-0.7))
        self.assertAlmostEqual(m.row()["alt"], 0.2)

    def test_ardupilot_keeps_relative_alt(self):
        m = self.monitor()
        m.feed(heartbeat(3, armed=True))
        m.feed(Msg("LOCAL_POSITION_NED", z=-9.0))
        m.feed(global_pos(2.0))
        self.assertAlmostEqual(m.row()["alt"], 2.0)

    def test_px4_without_local_position_falls_back(self):
        m = self.monitor()
        m.feed(heartbeat(12, armed=True))
        m.feed(global_pos(1.2))
        self.assertAlmostEqual(m.row()["alt"], 1.2)


class TestPreflightFirmware(unittest.TestCase):
    def check(self, seen, autopilot="auto"):
        return preflight_check.check_mavlink("udpin:x", 1, 1, connector=lambda d, b, t: seen, autopilot=autopilot)

    def test_reports_firmware(self):
        r = self.check(12)
        self.assertTrue(r.ok)
        self.assertIn("px4", r.detail)
        self.assertIn("ardupilot", self.check(3).detail)

    def test_mismatch_fails(self):
        r = self.check(3, "px4")
        self.assertFalse(r.ok)
        self.assertFalse(r.warning)
        self.assertIn("expected px4", r.detail)

    def test_match_passes(self):
        self.assertTrue(self.check(3, "ardupilot").ok)

    def test_unknown_firmware_fails(self):
        self.assertFalse(self.check(0).ok)

    def test_old_bool_connectors_still_work(self):
        self.assertTrue(self.check(True).ok)
        self.assertFalse(self.check(False).ok)
        self.assertFalse(self.check(None).ok)

    def test_cli_flag_reaches_the_check(self):
        args = preflight_check.parse_args(["--driver", "mavlink", "--autopilot", "px4", "--skip-camera"])
        self.assertEqual(args.autopilot, "px4")
