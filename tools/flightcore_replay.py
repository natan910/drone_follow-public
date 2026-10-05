#!/usr/bin/env python3
"""Replay a recorded sensor log through the flightcore estimator, offline.

    python tools/flightcore_replay.py sensors.csv                 # summary
    python tools/flightcore_replay.py sensors.csv --out est.csv   # full estimate, one row per IMU tick
    python tools/flightcore_replay.py other_log.csv --airborne-from 41.5

Logs written by tools/flightcore_sim.py --log (or flightcore.log.CsvRecorder) carry an `airborne` column and
replay exactly as flown.  For a log without it, pass --airborne-from <log seconds> or the whole log is treated
as sitting on the ground.  Use this after every filter change: same flight, new estimate.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flightcore.log import read_frames, replay, write_estimates  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="sensor log CSV")
    ap.add_argument("--out", help="write the estimate to this CSV")
    ap.add_argument("--airborne-from", type=float, default=None, help="log time (s) when flight started")
    a = ap.parse_args()

    with open(a.log, newline="") as f:
        rows = replay(read_frames(f), airborne_from=a.airborne_from)
    if not rows:
        print("no estimate: the log ended before the estimator finished aligning (needs ~1-2 s of stillness)")
        return 1
    last = rows[-1]
    print(f"{len(rows)} ticks, {rows[0]['t']:.2f}s .. {last['t']:.2f}s")
    print(f"final position N/E/D  {last['pn']:.2f} {last['pe']:.2f} {last['pd']:.2f} m")
    print(f"final velocity N/E/D  {last['vn']:.2f} {last['ve']:.2f} {last['vd']:.2f} m/s")
    print(f"final attitude r/p/y  {math.degrees(last['roll']):.1f} {math.degrees(last['pitch']):.1f} "
          f"{math.degrees(last['yaw']):.1f} deg")
    print(f"final sigma pos {last['sig_pos_h']:.2f} m, tilt {math.degrees(last['sig_tilt']):.2f} deg, "
          f"yaw {math.degrees(last['sig_yaw']):.2f} deg")
    lost = {k: sum(1 for r in rows if not r[k]) for k in ("att_ok", "vel_ok", "pos_ok", "alt_ok")}
    print("ticks with an invalid estimate: " + ", ".join(f"{k[:-3]} {v}" for k, v in lost.items()))
    if a.out:
        with open(a.out, "w", newline="") as f:
            write_estimates(rows, f)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
