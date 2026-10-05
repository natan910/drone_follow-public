"""
Fleet operations console: one place to watch and task every drone.

Each drone POSTs its status here once a second (fleet/reporter.py, turned on
with main.py --fleet-url/--drone-id). The reply to that POST carries anything
the operator queued for it -- a task switch, a hover height, a target photo,
a calibration request, a launch -- and the drone acknowledges each one in its
next report. The drone always opens the connection, so this works even when the
drone is on LTE or behind NAT and the console can't reach it directly.

    GET    /                              the console (tactical map, assets, tasking)
    POST   /report                        drone -> console: status + acks; reply = its outbox
    GET    /fleet                         JSON: every drone's last report
    GET    /fleet/<id>                    JSON: one drone
    DELETE /fleet/<id>                    forget a drone
    POST   /fleet/<id>/command            {"task": "HOVER"} / {"hover_height_m": 0.5} /
                                          {"calibrate_distance_m": 2.0} / {"launch": true}
                                          (same rules as the phone page)
    POST   /fleet/<id>/target             body = raw image: new target photo
    DELETE /fleet/<id>/target             forget the target
    GET    /events?since=<seq>            operator log: online / signal lost / sent / ack

Everything but the page itself needs the token. Same threat model as
comms/phone_server.py: plain HTTP on a private network you control.

    python -m fleet.server --token <secret>
"""

import base64
import collections
import hmac
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Deque, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2

from comms.phone_server import CommandError, PhoneServer, parse_command

Status = Dict[str, object]
DRONE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
PAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "console.html")


@dataclass(frozen=True)
class DroneStatus:
    drone_id: str
    status: Status
    age_s: float
    stale: bool
    pending: int = 0   # tasking queued or sent but not yet acknowledged


class UnknownDrone(KeyError):
    pass


class FleetStore:
    """Thread-safe: last status per drone, per-drone outbox of operator
    tasking, and a rolling operator event log. Reports overwrite; nothing is
    averaged."""

    def __init__(self, wall: Callable[[], float] = time.time, max_events: int = 300):
        self._lock = threading.Lock()
        self._wall = wall
        self._drones: Dict[str, Tuple[Status, float]] = {}   # drone_id -> (status, received_at)
        self._outbox: Dict[str, List[dict]] = {}              # queued, not yet collected
        self._inflight: Dict[str, Dict[int, str]] = {}        # collected, not yet acked: id -> summary
        self._lost: Dict[str, bool] = {}                      # stale flag we last logged
        self._events: Deque[dict] = collections.deque(maxlen=max_events)
        self._seq = 0

    # ---- drone side -----------------------------------------------------
    def report(self, drone_id: str, status: Status, now: float) -> None:
        with self._lock:
            if drone_id not in self._drones:
                self._log(drone_id, "online", "first contact")
            elif self._lost.get(drone_id):
                self._log(drone_id, "online", "signal reacquired")
            self._lost[drone_id] = False
            self._drones[drone_id] = (dict(status), now)
            self._outbox.setdefault(drone_id, [])
            self._inflight.setdefault(drone_id, {})

    def take_outbox(self, drone_id: str) -> List[dict]:
        """Everything queued for this drone; moves it to in-flight."""
        with self._lock:
            items, self._outbox[drone_id] = self._outbox.get(drone_id, []), []
            for item in items:
                self._inflight.setdefault(drone_id, {})[item["id"]] = item.pop("_summary")
            return items

    def ack(self, drone_id: str, acks: List[dict]) -> None:
        with self._lock:
            inflight = self._inflight.setdefault(drone_id, {})
            for a in acks:
                if not isinstance(a, dict) or "id" not in a:
                    continue
                summary = inflight.pop(a["id"], f"#{a['id']}")
                if a.get("ok"):
                    self._log(drone_id, "ack", f"{summary}: {_describe(a.get('result'))}")
                else:
                    self._log(drone_id, "error", f"{summary}: {a.get('error', 'failed')}")

    # ---- operator side ----------------------------------------------------
    def enqueue(self, drone_id: str, item: dict, summary: str) -> int:
        with self._lock:
            if drone_id not in self._drones:
                raise UnknownDrone(drone_id)
            self._seq += 1
            item = {"id": self._seq, **item, "_summary": summary}
            self._outbox[drone_id].append(item)
            self._log(drone_id, "sent", summary)
            return item["id"]

    def get(self, drone_id: str, now: float, stale_after_s: float) -> Optional[DroneStatus]:
        with self._lock:
            if drone_id not in self._drones:
                return None
            return self._row(drone_id, now, stale_after_s)

    def snapshot(self, now: float, stale_after_s: float) -> List[DroneStatus]:
        with self._lock:
            return [self._row(d, now, stale_after_s) for d in sorted(self._drones)]

    def forget(self, drone_id: str) -> bool:
        with self._lock:
            known = self._drones.pop(drone_id, None) is not None
            for table in (self._outbox, self._inflight, self._lost):
                table.pop(drone_id, None)
            if known:
                self._log(drone_id, "removed", "removed from the console")
            return known

    def events(self, since: int = 0) -> List[dict]:
        with self._lock:
            return [e for e in self._events if e["seq"] > since]

    # ---- internals (lock held) -------------------------------------------
    def _row(self, drone_id: str, now: float, stale_after_s: float) -> DroneStatus:
        status, received_at = self._drones[drone_id]
        age = now - received_at
        stale = age > stale_after_s
        if stale and not self._lost.get(drone_id):
            self._log(drone_id, "lost", f"no report for {age:.0f} s")
        self._lost[drone_id] = stale
        pending = len(self._outbox.get(drone_id, [])) + len(self._inflight.get(drone_id, {}))
        return DroneStatus(drone_id, status, age, stale, pending)

    def _log(self, drone_id: str, kind: str, text: str) -> None:
        self._seq += 1
        self._events.append({"seq": self._seq, "t": self._wall(), "drone_id": drone_id,
                             "kind": kind, "text": text})


def _describe(result) -> str:
    if not isinstance(result, dict):
        return "done"
    if "faces_in_photo" in result:
        looks = result.get("body_looks")
        return "target locked" + (f", body looks {looks}" if looks is not None else "")
    if result.get("target") is False:
        return "target cleared"
    if result.get("launch"):
        return "launching"
    return "accepted"   # command replies just echo the command back


def _summarise(command: dict) -> str:
    parts = []
    if command.get("launch"):
        parts.append("launch")
    if "task" in command:
        parts.append(f"task {command['task']}")
    if "hover_height_m" in command:
        parts.append(f"hover height {command['hover_height_m']:g} m")
    if "calibrate_distance_m" in command:
        parts.append(f"calibrate optics at {command['calibrate_distance_m']:g} m")
    return ", ".join(parts)


class FleetServer:
    def __init__(self, store: FleetStore, token: str, host: str = "0.0.0.0", port: int = 8090,
                 max_bytes: int = 200_000, max_image_bytes: int = 8_000_000,
                 stale_after_s: float = 15.0, verbose: bool = True,
                 now: Callable[[], float] = time.monotonic):
        self.store, self.token = store, token
        self.host, self.port = host, port
        self.max_bytes, self.max_image_bytes = max_bytes, max_image_bytes
        self.stale_after_s, self.verbose, self.now = stale_after_s, verbose, now
        self._httpd: Optional[ThreadingHTTPServer] = None
        with open(PAGE_PATH, encoding="utf-8") as f:
            self.page = f.read()

    def start(self) -> None:
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._make_handler())
        self.port = self._httpd.server_address[1]  # resolves port 0 to the real one
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # never log query strings (they hold the token)
                if server.verbose and urlparse(self.path).path not in ("/report", "/fleet", "/events"):
                    print(f"[fleet] {self.command} {urlparse(self.path).path} -> "
                          f"{args[1] if len(args) > 1 else ''}")

            def _reply(self, code: int, body, ctype: str = "text/plain; charset=utf-8") -> None:
                data = body if isinstance(body, bytes) else str(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _json(self, code: int, obj) -> None:
                self._reply(code, json.dumps(obj), "application/json")

            def _authorised(self) -> bool:
                supplied = (parse_qs(urlparse(self.path).query).get("token", [""])[0]
                            or self.headers.get("X-Token", ""))
                if hmac.compare_digest(supplied.encode(), server.token.encode()):
                    return True
                self._reply(401, "missing or wrong token")
                return False

            def _body(self, limit: int) -> Optional[bytes]:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    self._reply(400, "empty body")
                    return None
                if length > limit:
                    self._reply(413, f"body too large (limit {limit} bytes)")
                    return None
                return self.rfile.read(length)

            def _route(self) -> Tuple[str, Optional[str], Optional[str]]:
                """'/fleet/d1/target' -> ('fleet', 'd1', 'target')"""
                parts = [p for p in urlparse(self.path).path.split("/") if p]
                return ((parts[0] if parts else ""), (parts[1] if len(parts) > 1 else None),
                        (parts[2] if len(parts) > 2 else None)) if len(parts) <= 3 else ("", None, None)

            # ---- GET ------------------------------------------------------
            def do_GET(self):
                root, drone_id, sub = self._route()
                if root == "" and drone_id is None:
                    return self._reply(200, server.page, "text/html; charset=utf-8")
                if not self._authorised():
                    return
                if root == "fleet" and drone_id is None:
                    rows = server.store.snapshot(server.now(), server.stale_after_s)
                    return self._json(200, [_as_dict(r) for r in rows])
                if root == "fleet" and sub is None:
                    row = server.store.get(drone_id, server.now(), server.stale_after_s)
                    if row is None:
                        return self._reply(404, f"no reports from drone {drone_id!r} yet")
                    return self._json(200, _as_dict(row))
                if root == "events" and drone_id is None:
                    try:
                        since = int(parse_qs(urlparse(self.path).query).get("since", ["0"])[0])
                    except ValueError:
                        since = 0
                    return self._json(200, {"events": server.store.events(since)})
                self._reply(404, "not found")

            # ---- POST -----------------------------------------------------
            def do_POST(self):
                root, drone_id, sub = self._route()
                if not self._authorised():
                    return
                if root == "report" and drone_id is None:
                    return self._report()
                if root == "fleet" and drone_id and sub == "command":
                    return self._command(drone_id)
                if root == "fleet" and drone_id and sub == "target":
                    return self._target(drone_id)
                self._reply(404, "not found")

            def _report(self):
                raw = self._body(server.max_bytes)
                if raw is None:
                    return
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    return self._reply(400, "that is not valid JSON")
                if not isinstance(body, dict):
                    return self._reply(400, "expected a JSON object")
                drone_id = body.get("drone_id")
                if not isinstance(drone_id, str) or not DRONE_ID.match(drone_id):
                    return self._reply(422, '"drone_id" must be 1-40 letters, digits, "-", "_" or "."')
                acks = body.get("acks") or []
                status = {k: v for k, v in body.items() if k not in ("drone_id", "acks")}
                server.store.report(drone_id, status, server.now())
                if isinstance(acks, list):
                    server.store.ack(drone_id, acks)
                self._json(200, {"ok": True, "drone_id": drone_id,
                                 "outbox": server.store.take_outbox(drone_id)})

            def _command(self, drone_id: str):
                raw = self._body(10_000)
                if raw is None:
                    return
                try:
                    command = parse_command(json.loads(raw))
                except json.JSONDecodeError:
                    return self._reply(400, "that is not valid JSON")
                except CommandError as e:
                    return self._reply(e.code, e.message)
                self._queue(drone_id, {"kind": "command", "command": command}, _summarise(command))

            def _target(self, drone_id: str):
                raw = self._body(server.max_image_bytes)
                if raw is None:
                    return
                image = PhoneServer._decode(raw)   # decodes + shrinks to a sensible size
                if image is None:
                    return self._reply(415, "that is not an image I can read (try a JPEG or PNG)")
                ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    return self._reply(500, "could not re-encode the photo")
                item = {"kind": "target", "image_jpeg_b64": base64.b64encode(jpeg.tobytes()).decode()}
                self._queue(drone_id, item, "new target photo")

            def _queue(self, drone_id: str, item: dict, summary: str):
                try:
                    item_id = server.store.enqueue(drone_id, item, summary)
                except UnknownDrone:
                    return self._reply(404, f"no reports from drone {drone_id!r} yet")
                self._json(202, {"ok": True, "id": item_id, "queued": summary})

            # ---- DELETE ---------------------------------------------------
            def do_DELETE(self):
                root, drone_id, sub = self._route()
                if root != "fleet" or not drone_id or sub not in (None, "target"):
                    return self._reply(404, "not found")
                if not self._authorised():
                    return
                if sub == "target":
                    return self._queue(drone_id, {"kind": "clear_target"}, "forget target")
                removed = server.store.forget(drone_id)
                self._json(200 if removed else 404, {"ok": removed, "drone_id": drone_id})

        return Handler


def _as_dict(row: DroneStatus) -> dict:
    return {"drone_id": row.drone_id, "status": row.status, "age_s": row.age_s,
            "stale": row.stale, "pending": row.pending}


def main() -> None:
    import argparse
    import secrets
    import socket

    p = argparse.ArgumentParser(description="Fleet operations console: run this once, "
                                             "on a machine every drone can reach.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--token", help="shared secret drones and operators must send (default: random)")
    p.add_argument("--stale-after", type=float, default=15.0,
                   help="seconds without a report before a drone is flagged as lost")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    token = args.token or secrets.token_urlsafe(8)
    server = FleetServer(FleetStore(), token, args.host, args.port,
                         stale_after_s=args.stale_after, verbose=not args.quiet)
    server.start()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    print(f"\nConsole:  http://{ip}:{server.port}/?token={token}\n")
    print("Each drone joins with:  python main.py ... "
          f"--fleet-url http://{ip}:{server.port} --drone-id <name> --fleet-token {token}")
    print("Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
