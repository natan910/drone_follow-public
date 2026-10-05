#!/usr/bin/env python3
"""
Read-only flight monitor: one short line per second from a MAVLink stream,
optionally saved to a CSV to attach later.

Replaces MAVProxy's `watch GLOBAL_POSITION_INT`, which prints four lines a
second into the SITL window, where the only way out was Ctrl+C, and that
kills SITL. This only listens, on the spare output MAVProxy always sends to
(127.0.0.1:14550), so Ctrl+C here stops this monitor and nothing else.

    python tools/mav_watch.py                        # SITL (ArduPilot or PX4)
    python tools/mav_watch.py --csv ~/sitl_run1.csv  # and save every line

PX4 SITL also sends to 127.0.0.1:14550 (its ground-station link), so the
default works there too, as long as QGroundControl is not holding that port.

Start it before main.py, so it sees the arming and knows where home is.

Columns. fwd / right / yawr use DriveCommand's signs, so they compare directly
with what drone_follow asked for:
    time   wall clock
    mode   the flight controller's mode (ArduPilot: GUIDED, LAND, RTL, ...;
           PX4: OFFBOARD, LOITER (= Hold), TAKEOFF, LAND, RTL, POSCTL, ...)
    arm    ARMED, or - when disarmed
    alt    height above home, m. PX4: from LOCAL_POSITION_NED, relative to
           where it armed, because PX4's GLOBAL_POSITION_INT.relative_alt
           jumps by a metre or more when PX4 corrects home for baro drift
    climb  vertical speed, m/s (+ = up)
    fwd    speed along the nose, m/s (+ = forward)
    right  sideways speed, m/s (+ = toward the drone's right)
    yawr   turn rate, deg/s (+ = clockwise seen from above)
    hdg    heading, deg (0 = north, 90 = east)
    home   horizontal distance from home, m (home = where it last armed)
    bat    battery %, as the flight controller reports it
Mode changes, arming, and the flight controller's own messages (PreArm
failures, fence breaches, ...) print the moment they happen, marked ">>".
"""

import argparse
import csv
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple

# MAVLink numbers used below, so this module (and its tests) need no pymavlink.
MAV_TYPE_GCS = 6                  # heartbeat from a ground station (MAVProxy, QGroundControl)
MAV_AUTOPILOT_INVALID = 8         # heartbeat from a component that is not a flight controller
MAV_AUTOPILOT_PX4 = 12
MAV_MODE_FLAG_SAFETY_ARMED = 128
NO_HEADING = 65535                # GLOBAL_POSITION_INT.hdg when unknown

EARTH_RADIUS_M = 6_378_137.0

# (column, format, width); a negative width means left-aligned
FIELDS = (("time", "{}", -8), ("mode", "{}", -9), ("arm", "{}", -5), ("alt", "{:.1f}", 6),
          ("climb", "{:+.2f}", 6), ("fwd", "{:+.2f}", 6), ("right", "{:+.2f}", 6),
          ("yawr", "{:+.0f}", 5), ("hdg", "{:.0f}", 4), ("home", "{:.1f}", 6), ("bat", "{:.0f}", 4))
CSV_COLUMNS = ["time", "t"] + [name for name, _, _ in FIELDS[1:]] + ["event"]
HEADER_EVERY = 20                 # repeat the column names every this many lines


def distance_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Horizontal distance between two (lat, lon) points in degrees. Flat-earth:
    well under 1 % error within a few km, which is all a drone flight covers."""
    north = math.radians(b[0] - a[0]) * EARTH_RADIUS_M
    east = math.radians(b[1] - a[1]) * EARTH_RADIUS_M * math.cos(math.radians(a[0]))
    return math.hypot(north, east)


def _cell(value: Any, fmt: str, width: int) -> str:
    if value is None:
        text = "-"
    else:
        text = fmt.format(value)
        if isinstance(value, float) and text.startswith("-") and not text.strip("-0."):  # "-0.00" -> "+0.00"
            text = ("+" if fmt.startswith("{:+") else "") + text[1:]
    return text.ljust(-width) if width < 0 else text.rjust(width)


def header() -> str:
    return " ".join(_cell(name, "{}", width) for name, _, width in FIELDS)


def format_row(row: Dict[str, Any]) -> str:
    return " ".join(_cell(row.get(name), fmt, width) for name, fmt, width in FIELDS)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return round(value, 3) if isinstance(value, float) else value


class Monitor:
    """Latest state of one flight controller, from the MAVLink messages fed to
    it. No I/O; the clock is injected."""

    def __init__(self, mode_name: Callable[[Any], str], clock: Callable[[], float] = time.time):
        self._mode_name = mode_name              # HEARTBEAT -> "GUIDED" etc. (pymavlink's mode_string_v10)
        self._clock = clock
        self.t0 = clock()
        self.vehicle: Optional[Tuple[int, int]] = None   # (system, component) of the flight controller
        self.autopilot: Optional[int] = None     # MAV_AUTOPILOT_*: 3 ArduPilot, 12 PX4
        self.mode: Optional[str] = None
        self.armed: Optional[bool] = None
        self.position: Optional[Tuple[float, float]] = None   # lat, lon in degrees
        self.home: Optional[Tuple[float, float]] = None
        self._home_pending = False               # just armed: the next position is home
        self.alt_m: Optional[float] = None
        self._local_down: Optional[float] = None   # LOCAL_POSITION_NED z (PX4 height)
        self._down_ref: Optional[float] = None     # z where it armed (or first seen)
        self._down_ref_pending = False
        self.vel_ned: Optional[Tuple[float, float, float]] = None   # m/s north, east, down
        self.heading_deg: Optional[float] = None
        self._heading_from_attitude = False      # ATTITUDE is faster than GLOBAL_POSITION_INT; prefer it
        self.yaw_rate_dps: Optional[float] = None
        self.battery_pct: Optional[float] = None
        self._events: List[str] = []             # since the last take_events(), for the CSV

    def now(self) -> float:
        return self._clock()

    def clock_text(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self._clock()))

    # ---- input ---------------------------------------------------------------
    def feed(self, msg: Any) -> List[str]:
        """Take one MAVLink message. Returns the events to show right away."""
        kind = msg.get_type()
        source = (msg.get_srcSystem(), msg.get_srcComponent())
        if kind == "HEARTBEAT" and self.vehicle is None:
            if msg.type != MAV_TYPE_GCS and msg.autopilot != MAV_AUTOPILOT_INVALID:
                self.vehicle = source
                self.autopilot = msg.autopilot
        if source != self.vehicle:
            return []  # a ground station, a gimbal, ... or the flight controller not heard from yet

        events: List[str] = []
        if kind == "HEARTBEAT":
            events = self._heartbeat(msg)
        elif kind == "GLOBAL_POSITION_INT":
            self._position(msg)
        elif kind == "LOCAL_POSITION_NED":
            self._local_down = msg.z
            if self._down_ref is None or self._down_ref_pending:
                self._down_ref = msg.z
                self._down_ref_pending = False
        elif kind == "ATTITUDE":
            self.heading_deg = math.degrees(msg.yaw) % 360.0
            self._heading_from_attitude = True
            self.yaw_rate_dps = math.degrees(msg.yawspeed)
        elif kind == "SYS_STATUS":
            self.battery_pct = None if msg.battery_remaining < 0 else float(msg.battery_remaining)
        elif kind == "HOME_POSITION":
            self.home = (msg.latitude / 1e7, msg.longitude / 1e7)
            self._home_pending = False
        elif kind == "STATUSTEXT":
            text = msg.text.decode("utf-8", "replace") if isinstance(msg.text, bytes) else str(msg.text)
            events = ["FC: " + text.rstrip("\x00").strip()]
        self._events.extend(events)
        return events

    def _heartbeat(self, msg: Any) -> List[str]:
        events = []
        mode = self._mode_name(msg)
        if mode != self.mode:
            events.append(f"mode {mode}" if self.mode is None else f"mode {self.mode} -> {mode}")
            self.mode = mode
        armed = bool(msg.base_mode & MAV_MODE_FLAG_SAFETY_ARMED)
        if armed != self.armed:
            if armed or self.armed is not None:  # a drone sitting disarmed at start-up is not news
                events.append("ARMED" if armed else "DISARMED")
            if armed:
                self._home_pending = True        # the flight controller sets home where it arms
                self._down_ref_pending = True
            self.armed = armed
        return events

    def _position(self, msg: Any) -> None:
        if msg.lat == 0 and msg.lon == 0:
            return                               # no position fix yet
        self.position = (msg.lat / 1e7, msg.lon / 1e7)
        self.alt_m = msg.relative_alt / 1000.0
        self.vel_ned = (msg.vx / 100.0, msg.vy / 100.0, msg.vz / 100.0)
        if not self._heading_from_attitude and msg.hdg != NO_HEADING:
            self.heading_deg = msg.hdg / 100.0
        if self.home is None or self._home_pending:
            self.home = self.position
            self._home_pending = False

    # ---- output --------------------------------------------------------------
    def row(self) -> Dict[str, Any]:
        climb = fwd = right = None
        if self.vel_ned is not None:
            north, east, down = self.vel_ned
            climb = -down                        # MAVLink counts down as positive
            if self.heading_deg is not None:
                h = math.radians(self.heading_deg)
                fwd = north * math.cos(h) + east * math.sin(h)
                right = -north * math.sin(h) + east * math.cos(h)
        home = None
        if self.home is not None and self.position is not None:
            home = distance_m(self.home, self.position)
        alt = self.alt_m
        if self.autopilot == MAV_AUTOPILOT_PX4 and self._local_down is not None and self._down_ref is not None:
            alt = -(self._local_down - self._down_ref)
        now = self._clock()
        return {"time": time.strftime("%H:%M:%S", time.localtime(now)), "t": now - self.t0,
                "mode": self.mode, "arm": None if self.armed is None else ("ARMED" if self.armed else "-"),
                "alt": alt, "climb": climb, "fwd": fwd, "right": right,
                "yawr": self.yaw_rate_dps, "hdg": self.heading_deg, "home": home, "bat": self.battery_pct}

    def take_events(self) -> List[str]:
        events, self._events = self._events, []
        return events


def watch(link: Any, monitor: Monitor, every_s: float, out: TextIO,
          csv_file: Optional[TextIO] = None, url: str = "", hint_after_s: float = 5.0) -> None:
    """Print a line every `every_s` seconds, events as they come. Runs until
    Ctrl+C (KeyboardInterrupt), which the caller handles."""
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS) if csv_file is not None else None
    if writer is not None:
        writer.writeheader()
    rows, hinted, next_row = 0, False, monitor.now()
    while True:
        msg = link.recv_match(blocking=True, timeout=0.2)
        if msg is not None and msg.get_type() != "BAD_DATA":
            for event in monitor.feed(msg):
                print(f"{monitor.clock_text()}  >> {event}", file=out, flush=True)
        now = monitor.now()
        if monitor.vehicle is None:
            if not hinted and now - monitor.t0 >= hint_after_s:
                print(f"No heartbeat from a flight controller on {url} yet. Is SITL running? "
                      "(sim_vehicle.py and PX4 SITL both send to 127.0.0.1:14550; "
                      "QGroundControl open on the same port takes it.)", file=out, flush=True)
                hinted = True
            continue
        if now < next_row:
            continue
        next_row = now + every_s
        row = monitor.row()
        if rows % HEADER_EVERY == 0:
            print(header(), file=out)
        print(format_row(row), file=out, flush=True)
        rows += 1
        record = {name: _csv_value(row.get(name)) for name in CSV_COLUMNS[:-1]}
        record["event"] = " | ".join(monitor.take_events())
        if writer is not None:
            writer.writerow(record)
            csv_file.flush()


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only flight monitor: one line per second from MAVLink.")
    p.add_argument("--mavlink", default="udpin:127.0.0.1:14550",
                   help="where to listen (default: the spare output sim_vehicle.py always sends to). "
                        "Not 14551: main.py listens there")
    p.add_argument("--baud", type=int, default=57600, help="serial links only")
    p.add_argument("--every", type=float, default=1.0, help="seconds between lines (default 1)")
    p.add_argument("--csv", help="also save every line to this CSV file")
    return p.parse_args(argv)


def open_link(url: str, baud: int) -> Tuple[Any, Callable[[Any], str]]:
    from pymavlink import mavutil  # imported here, so the tests never need pymavlink
    return mavutil.mavlink_connection(url, baud=baud), mavutil.mode_string_v10


def main(argv: Optional[List[str]] = None, connect: Callable = open_link, out: TextIO = sys.stdout,
         clock: Callable[[], float] = time.time) -> int:
    args = parse_args(argv)
    try:
        link, mode_name = connect(args.mavlink, args.baud)
    except ImportError:
        print("pymavlink is missing: pip install pymavlink (or ./setup.sh --full).", file=out)
        return 1
    except OSError as e:
        print(f"Cannot listen on {args.mavlink}: {e}\n"
              "Another program (a ground station such as QGroundControl?) may be using that port.", file=out)
        return 1
    monitor = Monitor(mode_name, clock)
    csv_path = os.path.expanduser(args.csv) if args.csv else None
    csv_file = open(csv_path, "w", newline="") if csv_path else None
    print(f"Listening on {args.mavlink}. Read-only: Ctrl+C here stops only this monitor.", file=out, flush=True)
    try:
        watch(link, monitor, args.every, out, csv_file, url=args.mavlink)
    except KeyboardInterrupt:
        pass
    finally:
        if csv_file is not None:
            csv_file.close()
        close = getattr(link, "close", None)
        if close is not None:
            close()
    print("\nStopped." + (f" Saved {csv_path}" if csv_path else ""), file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
