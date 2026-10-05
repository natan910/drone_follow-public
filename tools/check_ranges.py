"""
Bench check for the TF-Luna range sensors: prints what each one sees, live.
Point them at a wall, then at open sky, then wave a hand in front.

    python tools/check_ranges.py "0:/dev/ttyUSB0,-45:/dev/ttyUSB1,45:/dev/ttyUSB2"

Plug them in one at a time and note which /dev/ttyUSB* each becomes (or use the
stable names in /dev/serial/by-id/).
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.range_sensors import TFLunaRing  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ports = {float(b): p for b, p in (item.split(":", 1) for item in sys.argv[1].split(","))}
    ring = TFLunaRing(ports)
    try:
        while True:
            scan, age = ring.read(time.monotonic())
            if scan is None:
                print("no fresh readings")
            else:
                cells = [f"{math.degrees(b.bearing):+4.0f}deg: " +
                         ("  --  " if b.distance is None else f"{b.distance:5.2f}m") for b in scan.beams]
                print("  |  ".join(cells), f"   (age {age * 1000:.0f} ms)")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        ring.close()


if __name__ == "__main__":
    main()
