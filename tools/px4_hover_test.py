"""
First PX4 test, SITL or bench (props OFF on a bench): connect, arm, take off,
hover, optionally check the direction of every command, land, wait for disarm.
Uses the same Px4Driver main.py uses, so a pass here means the driver works.

    # PX4 SITL (headless SIH quad), terminal 1, in the PX4-Autopilot checkout:
    make px4_sitl sihsim_quadx
    # terminal 2, in drone_follow with .venv active:
    python tools/px4_hover_test.py --mavlink udpin:127.0.0.1:14540 --signs

--signs flies 2 m forward, right, up and yaws right, one at a time, and prints
PASS/FAIL for each direction (world-frame change measured by PX4 itself).
"""

import argparse
import logging
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.flight_controller import open_flight_controller  # noqa: E402
from datatypes import DriveCommand  # noqa: E402


def fly(driver, cmd: DriveCommand, seconds: float, hz: float = 10.0) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not driver.autonomy_permitted():
            raise RuntimeError(f"left OFFBOARD (now {driver.mode_name()}): a failsafe or the pilot took over")
        driver.send(cmd)
        time.sleep(1.0 / hz)


def status(driver) -> str:
    p = driver.pose()
    batt = driver.battery_pct()
    return (f"{driver.mode_name():<12} armed={driver.armed()!s:<5} x(east)={p.x:+6.2f} y(north)={p.y:+6.2f} "
            f"z(up)={p.z:5.2f} yaw={math.degrees(p.yaw):+7.1f} batt={batt if batt is not None else '?'}")


def check_signs(driver) -> bool:
    ok = True
    tests = [("forward 0.5 m/s", DriveCommand(forward_mps=0.5), "fwd"),
             ("right 0.5 m/s", DriveCommand(right_mps=0.5), "right"),
             ("up 0.3 m/s", DriveCommand(up_mps=0.3), "up"),
             ("yaw +20 deg/s (clockwise, seen from above)", DriveCommand(yaw_rate_dps=20.0), "yaw")]
    for label, cmd, kind in tests:
        fly(driver, DriveCommand(), 2.0)                 # settle
        a = driver.pose()
        fly(driver, cmd, 4.0)
        b = driver.pose()
        fly(driver, DriveCommand(), 2.0)
        dx, dy, dz = b.x - a.x, b.y - a.y, b.z - a.z
        # expected world displacement from the heading at the start (x east, y north, yaw 0 = north)
        fwd = (math.sin(a.yaw), math.cos(a.yaw))
        right = (math.cos(a.yaw), -math.sin(a.yaw))
        if kind == "fwd":
            got = dx * fwd[0] + dy * fwd[1]
        elif kind == "right":
            got = dx * right[0] + dy * right[1]
        elif kind == "up":
            got = dz
        else:
            got = math.degrees(math.atan2(math.sin(b.yaw - a.yaw), math.cos(b.yaw - a.yaw)))
        passed = got > (30.0 if kind == "yaw" else 0.5)
        ok &= passed
        unit = "deg" if kind == "yaw" else "m"
        print(f"  {'PASS' if passed else 'FAIL'}  {label:<44} moved {got:+.2f} {unit}")
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mavlink", default="udpin:127.0.0.1:14540")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--alt", type=float, default=1.5, help="takeoff height above home, m")
    p.add_argument("--hover", type=float, default=10.0, help="seconds to hover")
    p.add_argument("--signs", action="store_true", help="also check forward/right/up/yaw directions")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    driver = open_flight_controller(args.mavlink, args.baud, "px4", takeoff_alt_m=args.alt, gimbal=False)
    ok = True
    with driver:                                         # stop -> land -> disconnect however we leave
        print("connected: ", status(driver))
        driver.arm()
        print("armed:     ", status(driver))
        driver.takeoff()
        print("offboard:  ", status(driver))
        end = time.monotonic() + args.hover
        while time.monotonic() < end:
            fly(driver, DriveCommand(), 1.0)
            print("hover:     ", status(driver))
        if args.signs:
            print("direction checks:")
            ok = check_signs(driver)
        driver.land()
        print("landing... ", status(driver))
        deadline = time.monotonic() + 60
        while driver.armed() and time.monotonic() < deadline:
            time.sleep(1.0)
            print("           ", status(driver))
        if driver.armed():
            print("FAIL: still armed 60 s after LAND")
            ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
