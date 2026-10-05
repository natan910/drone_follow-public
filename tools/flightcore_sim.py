#!/usr/bin/env python3
"""Fly the flightcore stack against the simulated quad and print what happened.

    python tools/flightcore_sim.py                    # list scenarios
    python tools/flightcore_sim.py hover              # one scenario
    python tools/flightcore_sim.py all                # every scenario
    python tools/flightcore_sim.py square --wind 4,0,0 --seed 3
    python tools/flightcore_sim.py hover --csv trace.csv --log sensors.csv

--csv  : truth vs estimate every 50 ms (open in a spreadsheet / plot)
--log  : raw sensor log, replay it with tools/flightcore_replay.py

No hardware, no network.  A scenario passes when the vehicle lands and disarms without crashing.
"""
import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from flightcore.log import CsvRecorder  # noqa: E402
from flightcore.sim import SensorSpec  # noqa: E402
from flightcore.sim.harness import ClosedLoop  # noqa: E402


def near(target, tol=0.3):
    t = np.asarray(target, dtype=float)
    return lambda s: float(np.linalg.norm(s.truth.p - t)) < tol


def scen_hover(s):
    s.run(10.0)


def scen_square(s):
    for n, e in ((5, 0), (5, 5), (0, 5), (0, 0)):
        s.core.goto(n, e, -2.0)
        s.run_until(near([n, e, -2.0]), 20.0)
        s.run(1.0)


def scen_follow(s):
    """What the person-following brain does: body-frame velocity commands, re-sent at 10 Hz."""
    s.fly_velocity(1.5, 0.0, 0.0, 0.0, 4.0)
    s.fly_velocity(1.5, 0.0, 0.0, 0.3, 4.0)        # forward while turning right
    s.fly_velocity(0.0, 0.0, 0.0, 0.0, 2.0)


def scen_rtl(s):
    s.fly_velocity(2.0, 0.0, 0.0, 0.0, 5.0)
    s.core.rtl()
    s.run_until(lambda c: not c.core.armed, 60.0)


def scen_link_loss(s):
    s.fly_velocity(1.0, 0.0, 0.0, 0.0, 3.0)
    s.run(20.0)                                     # brain goes silent: hold, then land


SCENARIOS = {
    "hover": (scen_hover, {}),
    "square": (scen_square, {}),
    "follow": (scen_follow, {}),
    "rtl": (scen_rtl, {}),
    "link_loss": (scen_link_loss, {}),
    "gps_outage": (scen_hover, {"gps_outages": [(9.0, 14.0)]}),
    "windy": (scen_square, {"wind": (4.0, 2.0, 0.0), "gust_std": 1.2}),
}


def run_scenario(name, seed, wind, csv_path, log_path):
    fn, opts = SCENARIOS[name]
    spec = SensorSpec(gps_outages=opts.get("gps_outages", []))
    if wind is None:
        wind = opts.get("wind", (0.0, 0.0, 0.0))
    log_f = open(log_path, "w", newline="") if log_path else None
    try:
        rec = CsvRecorder(log_f).record if log_f else None
        s = ClosedLoop(spec=spec, seed=seed, wind=wind, gust_std=opts.get("gust_std", 0.0), record_every=25,
                       recorder=rec)
        wall0 = time.perf_counter()
        if not s.arm_and_takeoff(2.0):
            print(f"{name}: FAIL could not arm/take off: {s.core.prearm_check() or s.core.status()['reason']}")
            return False
        t_fly = s.t
        s.run(1.0)
        fn(s)
        if s.core.armed and s.core.mode.value not in ("land", "rtl"):
            s.core.land()
        s.run_until(lambda c: not c.core.armed, 60.0)
        wall = time.perf_counter() - wall0
    finally:
        if log_f:
            log_f.close()

    tr = s.trace
    t = tr.arr("t")
    fly = t >= t_fly
    err = np.linalg.norm(tr.arr("p_est") - tr.arr("p"), axis=1)[fly]
    speed = np.linalg.norm(tr.arr("v")[:, :2], axis=1)
    landed = not s.core.armed
    crashed = s.truth.crashed
    ok = landed and not crashed
    print(f"{name:11s} {'PASS' if ok else 'FAIL'}  sim {s.t:5.1f}s  wall {wall:4.1f}s ({1e3 * wall / s.k:.2f} ms/tick)  "
          f"max tilt {math.degrees(s.max_tilt):4.1f} deg  max speed {speed.max():4.1f} m/s  "
          f"est err max {err.max():.2f} m  hardest touchdown {s.truth.max_impact:.2f} m/s  "
          f"end mode {s.core.mode.value}{'  CRASHED' if crashed else ''}")
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "n", "e", "d", "n_est", "e_est", "d_est", "mode"])
            for k in range(len(t)):
                w.writerow([f"{t[k]:.3f}", *(f"{x:.3f}" for x in tr.p[k]), *(f"{x:.3f}" for x in tr.p_est[k]),
                            tr.mode[k]])
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", help="one of: " + ", ".join(SCENARIOS) + ", all")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--wind", help="mean wind north,east,down in m/s, e.g. 4,0,0")
    ap.add_argument("--csv", help="write a truth/estimate trace to this file")
    ap.add_argument("--log", help="write the raw sensor log to this file (for flightcore_replay.py)")
    a = ap.parse_args()
    if not a.scenario:
        ap.print_help()
        return 0
    wind = tuple(float(x) for x in a.wind.split(",")) if a.wind else None
    names = list(SCENARIOS) if a.scenario == "all" else [a.scenario]
    for n in names:
        if n not in SCENARIOS:
            print(f"unknown scenario '{n}'. choices: {', '.join(SCENARIOS)}, all")
            return 2
    if len(names) > 1 and (a.csv or a.log):
        print("--csv/--log need a single scenario")
        return 2
    results = [run_scenario(n, a.seed, wind, a.csv, a.log) for n in names]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
