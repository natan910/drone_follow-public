"""
See the fleet console working before any drone exists: starts the console
and a few simulated drones that fly, report, and answer operator tasking
through the REAL relay (fleet/reporter.py over HTTP) -- only the flying is
fake.

    python tools/fleet_demo.py                  # then open the URL it prints
    python tools/fleet_demo.py --drones 4 --drop-after 30
        # one drone stops reporting after 30 s, to see "signal lost"

Try: send a task, upload any photo with a face in it ("target locked"),
tap Calibrate (watch it measure, then report an HFOV), RETURN a drone.
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comms.phone_server import CommandBox, EnrollmentBox  # noqa: E402
from fleet.reporter import FleetReporter  # noqa: E402
from fleet.server import FleetServer, FleetStore  # noqa: E402


class FakeDrone:
    """Just enough flight to make the console meaningful: moves toward a
    goal set by its task, drains its battery, and services its boxes the way
    RealPlatform does (between steps, on one thread)."""

    def __init__(self, drone_id: str, kind: str, phase: float, battery: float):
        self.id, self.kind, self.phase = drone_id, kind, phase
        self.box, self.commands = EnrollmentBox(), CommandBox()
        self.x, self.y, self.z, self.yaw = 0.0, 0.0, 0.0, 0.0
        self.battery = battery
        self.task = "FOLLOW" if kind == "follow" else "PATROL"
        self.enrolled = kind == "follow"
        self.cal = {"state": "idle"}
        self.hover = 0.3
        self.subject = None

    def service(self):
        req = self.box.poll()
        if req:
            image, reply = req
            self.enrolled = image is not None
            reply.set_result({"ok": True, "target": True, "faces_in_photo": 1, "body_looks": 1}
                             if image is not None else {"ok": True, "target": False})
        req = self.commands.poll()
        if req:
            cmd, reply = req
            self.task = cmd.get("task", self.task)
            self.hover = cmd.get("hover_height_m", self.hover)
            if "calibrate_distance_m" in cmd:
                self.cal = {"state": "measuring", "samples": 0, "needed": 15,
                            "distance_m": cmd["calibrate_distance_m"]}
            reply.set_result({"ok": True, **cmd})

    def step(self, t: float, dt: float) -> dict:
        self.service()
        if self.cal["state"] == "measuring":
            self.cal["samples"] += 1
            if self.cal["samples"] >= self.cal["needed"]:
                self.cal = {"state": "done", "hfov_deg": 71.4, "aspect": 0.5625}

        mode, goal, speed = "PATROL", None, 1.2
        if self.task in ("FOLLOW", "HOVER") and self.enrolled:
            a = 0.07 * t + self.phase                     # the subject strolls a figure-eight
            self.subject = (-14 + 7 * math.sin(a), 12 + 4 * math.sin(2 * a))
            goal, mode = self.subject, ("HOVER" if self.task == "HOVER" else "TRACK")
        elif self.task in ("PATROL", "FOLLOW"):
            self.subject = None
            r = 13 if self.kind == "orbit" else 20
            a = 0.05 * t + self.phase
            goal = (r * math.sin(a) + (6 if self.kind == "orbit" else 0), r * math.cos(a) * 0.7 - 4)
        elif self.task == "RETURN":
            goal, mode = (0.0, 0.0), "RETURN"
        elif self.task in ("HOLD", "LAND"):
            mode = self.task
        if self.task == "LAND" or (self.task == "RETURN" and math.hypot(self.x, self.y) < 0.5):
            self.z = max(0.0, self.z - 0.4 * dt)
        else:
            self.z += (2.0 - self.z) * min(1.0, dt)

        if goal is not None:
            dx, dy = goal[0] - self.x, goal[1] - self.y
            dist = math.hypot(dx, dy)
            if dist > 0.3:
                step = min(dist, speed * dt)
                self.x += dx / dist * step
                self.y += dy / dist * step
                target_yaw = math.atan2(dx, dy)
                diff = (target_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
                self.yaw += max(-1.5 * dt, min(1.5 * dt, diff))
        self.battery = max(0.0, self.battery - 0.02 * dt)
        return {"mode": mode, "task": self.task, "tracker": "TRACKING" if self.subject else "SEARCHING",
                "hover_height_m": self.hover, "parked": False, "map_known": min(0.95, 0.1 + t / 600),
                "hfov_deg": self.cal.get("hfov_deg", 66.0), "calibrated": self.cal["state"] == "done",
                "target_xy": [round(v, 1) for v in self.subject] if self.subject else None,
                "target_enrolled": self.enrolled, "battery": round(self.battery, 1),
                "x": round(self.x, 1), "y": round(self.y, 1), "z": round(self.z, 1),
                "yaw_deg": round(math.degrees(self.yaw), 1), "geofence_m": 30.0,
                "calibration": dict(self.cal),
                "reid": {"source": "face", "best_sim": 0.91} if self.subject else {}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--token", default="demo")
    p.add_argument("--drones", type=int, default=3, choices=range(1, 7))
    p.add_argument("--drop-after", type=float, default=0.0,
                   help="make the last drone go silent after this many seconds (0 = never)")
    p.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until Ctrl-C)")
    args = p.parse_args()

    server = FleetServer(FleetStore(), args.token, "0.0.0.0", args.port, verbose=False)
    server.start()
    url = f"http://127.0.0.1:{server.port}"
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    kinds = ["follow", "orbit", "patrol", "orbit", "patrol", "follow"]
    drones = [FakeDrone(f"{names[i]}-{i + 1}", kinds[i], i * 2.1, 92 - 11 * i) for i in range(args.drones)]
    reporters = [FleetReporter(url, d.id, args.token, interval_s=0.5,
                               enrollment=d.box, commands=d.commands) for d in drones]
    for r in reporters:
        r.start()
    print(f"\nFleet console (demo):  {url}/?token={args.token}\n\nCtrl-C to stop.")

    t0 = last = time.monotonic()
    try:
        while not args.seconds or time.monotonic() - t0 < args.seconds:
            now = time.monotonic()
            t, dt = now - t0, now - last
            last = now
            for i, (d, r) in enumerate(zip(drones, reporters)):
                status = d.step(t, dt)
                silent = args.drop_after and i == len(drones) - 1 and t > args.drop_after
                if not silent:
                    r.update(status)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for r in reporters:
            r.stop()
        server.stop()


if __name__ == "__main__":
    main()
