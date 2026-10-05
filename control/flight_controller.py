"""
Open the right MAVLink driver for whatever flight controller is on the link.

    driver = open_flight_controller("/dev/serial0", 921600, "auto", gimbal=False)

autopilot:
  "auto"       read the flight controller's HEARTBEAT and pick the driver
  "ardupilot"  MavlinkDriver (GUIDED); refuses to start if the FC is PX4
  "px4"        Px4Driver (OFFBOARD); refuses to start if the FC is ArduPilot

Either way it waits (up to timeout_s) for a heartbeat, so a wrong port fails
here with a clear message instead of later. The connection is injectable:
tests pass connect_fn and mav, and never need pymavlink.
"""

import time
from typing import Any, Callable, Optional

from control.mavlink_driver import MavlinkDriver
from control.px4_driver import Px4Driver

AUTOPILOTS = ("auto", "ardupilot", "px4")
MAV_AUTOPILOT_ARDUPILOTMEGA = 3
MAV_AUTOPILOT_PX4 = 12
_NAMES = {MAV_AUTOPILOT_ARDUPILOTMEGA: "ardupilot", MAV_AUTOPILOT_PX4: "px4"}
_DRIVERS = {"ardupilot": MavlinkDriver, "px4": Px4Driver}


def detect_autopilot(conn: Any, mav: Any, timeout_s: float = 10.0,
                     clock: Callable[[], float] = time.monotonic) -> int:
    """MAV_AUTOPILOT number from the first flight-controller heartbeat.
    Skips heartbeats from ground stations and other non-autopilot components."""
    deadline = clock() + timeout_s
    while True:
        left = deadline - clock()
        if left <= 0:
            raise RuntimeError(f"no heartbeat from a flight controller within {timeout_s:.0f} s: "
                               "wrong --mavlink port or baud, or the flight controller is off")
        msg = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=min(left, 1.0))
        if msg is None:
            continue
        if msg.type == mav.MAV_TYPE_GCS or msg.autopilot == mav.MAV_AUTOPILOT_INVALID:
            continue
        return msg.autopilot


def open_flight_controller(url: str = "/dev/serial0", baud: int = 921600, autopilot: str = "auto",
                           connect_fn: Optional[Callable[[], Any]] = None, mav: Any = None,
                           timeout_s: float = 10.0, clock: Callable[[], float] = time.monotonic,
                           **driver_kwargs) -> MavlinkDriver:
    """url: "/dev/serial0" on the drone; "udpin:127.0.0.1:14551" for ArduPilot SITL
    (sim_vehicle.py --out); "udpin:127.0.0.1:14540" for PX4 SITL.
    driver_kwargs go to the driver (gimbal=..., takeoff_alt_m=..., ...)."""
    if autopilot not in AUTOPILOTS:
        raise ValueError(f"autopilot must be one of {AUTOPILOTS}")
    if connect_fn is None:
        from pymavlink import mavutil
        mav = mavutil.mavlink
        connect_fn = lambda: mavutil.mavlink_connection(url, baud=baud)   # noqa: E731
    conn = connect_fn()
    try:
        found_id = detect_autopilot(conn, mav, timeout_s, clock)
        found = _NAMES.get(found_id)
        if found is None:
            raise RuntimeError(f"unsupported flight controller firmware (MAV_AUTOPILOT {found_id}): "
                               "only ArduPilot and PX4 are supported")
        if autopilot != "auto" and autopilot != found:
            raise RuntimeError(f"--autopilot {autopilot}, but the flight controller on {url} is {found}")
    except Exception:
        conn.close()
        raise
    print(f"Flight controller: {found} ({'detected' if autopilot == 'auto' else 'confirmed'})")
    return _DRIVERS[found](conn, mav, **driver_kwargs)
