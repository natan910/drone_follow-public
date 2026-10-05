"""
First autonomous flight test: arm in GUIDED, take off to a low hover, hold for
a few seconds sending zero velocity, then land. No camera, no navigation.
If this is not rock solid, nothing else should fly.

    python tools/hover_test.py --mavlink /dev/serial0
    python tools/hover_test.py --mavlink udpin:127.0.0.1:14551      # ArduPilot SITL

Keep the RC transmitter in your hand. Flipping its mode switch out of GUIDED
takes control back instantly.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.mavlink_driver import MavlinkDriver  # noqa: E402
from datatypes import DriveCommand  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mavlink", default="/dev/serial0")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--altitude", type=float, default=1.5)
    p.add_argument("--hold", type=float, default=10.0, help="seconds to hover")
    args = p.parse_args()

    with MavlinkDriver.open(args.mavlink, args.baud, takeoff_alt_m=args.altitude) as drone:
        print("Connected. Arming in GUIDED...")
        drone.arm()
        print(f"Armed. Taking off to {args.altitude} m...")
        drone.takeoff()
        end = time.monotonic() + args.hold
        while time.monotonic() < end:
            if not drone.autonomy_permitted():
                print("Pilot took control: stopping.")
                return
            drone.send(DriveCommand())
            p = drone.pose()
            print(f"hover  x={p.x:+.2f} y={p.y:+.2f} battery={drone.battery_pct()}")
            time.sleep(0.1)
        print("Landing.")
    # leaving the `with` block stops, lands, and disconnects


if __name__ == "__main__":
    main()
