#!/usr/bin/env python3
"""
One screen summary of a main.py flight log (`--log run.jsonl`): what the
autopilot decided, when, and where. Instead of reading thousands of lines.

    python tools/flight_summary.py ~/sitl_run1.jsonl [more.jsonl ...]

Prints: length and loop rate (steps per second: the number to watch on the Pi),
highest point, farthest from home, whether the battery was ever reported, then
one line per mode change (time since start, mode, height, distance from home,
the autopilot's note). "home" is the pose origin (where the flight controller
set its origin), which is where the drone took off.
"""

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, TextIO, Tuple


def load(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """Rows of the log, and how many lines could not be read (a Ctrl+C can cut
    the last line in half)."""
    rows, bad = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(row, dict) and "t" in row and "mode" in row:
                rows.append(row)
            else:
                bad += 1
    return rows, bad


def _clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _home_m(row: Dict[str, Any]) -> Optional[float]:
    x, y = row.get("x"), row.get("y")
    return None if x is None or y is None else math.hypot(x, y)


def _num(value: Optional[float], fmt: str = "{:.1f}") -> str:
    return "-" if value is None else fmt.format(value)


def summarize(rows: List[Dict[str, Any]], name: str = "log", bad_lines: int = 0) -> List[str]:
    if not rows:
        return [f"{name}: no flight steps in it"]
    t0, t1 = rows[0]["t"], rows[-1]["t"]
    duration = t1 - t0
    lines = [f"{name}: {len(rows)} steps, {_clock(duration)} long"
             + (f", loop {(len(rows) - 1) / duration:.1f} steps/s" if duration > 0 else "")
             + (f"  ({bad_lines} unreadable line(s) skipped)" if bad_lines else "")]

    heights = [r["z"] for r in rows if r.get("z") is not None]
    dists = [d for d in (_home_m(r) for r in rows) if d is not None]
    if heights:
        lines.append(f"highest {max(heights):.1f} m, farthest {max(dists) if dists else 0:.1f} m from home")

    batteries = [r["battery"] for r in rows if r.get("battery") is not None]
    if batteries:
        lines.append(f"battery {batteries[0]:.0f} % -> {batteries[-1]:.0f} % (lowest {min(batteries):.0f} %)")
    else:
        lines.append("battery: NEVER reported -> low-battery RETURN/LAND could not have triggered")

    lines.append(f"{'time':>6}  {'mode':<8} {'alt':>5} {'home':>6}  note")
    previous = None
    for r in rows:
        if r["mode"] == previous:
            continue
        previous = r["mode"]
        lines.append(f"{_clock(r['t'] - t0):>6}  {r['mode']:<8} {_num(r.get('z')):>5} "
                     f"{_num(_home_m(r)):>6}  {r.get('note') or ''}".rstrip())
    last = rows[-1]
    lines.append(f"end: {last['mode']}, {_num(last.get('z'))} m up, {_num(_home_m(last))} m from home")
    return lines


def main(argv: Optional[List[str]] = None, out: TextIO = sys.stdout) -> int:
    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print("usage: python tools/flight_summary.py run.jsonl [more.jsonl ...]", file=out)
        return 2
    code = 0
    for i, path in enumerate(paths):
        if i:
            print(file=out)
        path = os.path.expanduser(path)
        try:
            rows, bad = load(path)
        except OSError as e:
            print(f"{path}: cannot read ({e.strerror})", file=out)
            code = 1
            continue
        for line in summarize(rows, os.path.basename(path), bad):
            print(line, file=out)
    return code


if __name__ == "__main__":
    sys.exit(main())
