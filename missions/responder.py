"""
R4 responder: fixed sensors call the drone, a human says go.

A drone flies 10-15 minutes per battery: it cannot patrol all night. Cheap fixed
sensors (a PIR motion sensor, a door contact, a camera's motion event) can
watch all night and call the drone to the right place:

    sensor --HTTP--> TriggerServer (this file, on the drone)
        on the ground: push "Motion: back door" with a [Launch + check] button
                       (tap it: the phone asks the drone's phone page to launch)
        in the air:    push without a button
    after launch (or right away if already flying): the drone investigates the
    zone's centre (R1), then goes back to patrolling.

The drone never launches by itself (CLAUDE.md: no auto-launch on a real drone).
The tap is the human decision; outdoors, that human must also see the drone (VLOS).

Sensor calls (token = the phone page token, --token):
    GET  http://<drone>:8091/trigger?token=T&zone=back%20gate&source=pir-1
    POST http://<drone>:8091/trigger?token=T   {"zone": "back gate", "source": "pir-1"}
    GET  http://<drone>:8091/status?token=T
Zone names come from zones.json. The same source re-triggers at most every
source_cooldown_s; a trigger waits pending_s for the drone to be airborne.
Works with ESPHome (http_request), Shelly (actions -> URL), Home Assistant
(rest_command), or curl.

Threading: the HTTP thread only records the pending trigger (under a lock) and
queues a push. The main loop picks the trigger up (take_pending) and steers.
"""

import hmac
import json
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Optional, Sequence
from urllib.parse import parse_qs, urlparse

from missions.entities import Alert, EventLog
from missions.mission_config import ResponderConfig
from missions.perimeter import centroid


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))     # no packet is sent; just picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def launch_actions(phone_url: str, token: str) -> list:
    base = phone_url.rstrip("/")
    return [{"action": "http", "label": "Launch + check", "url": f"{base}/command", "method": "POST",
             "headers": {"X-Token": token, "Content-Type": "application/json"},
             "body": '{"launch": true}', "clear": True},
            {"action": "view", "label": "Open drone page", "url": f"{base}/?token={token}"}]


class Responder:
    def __init__(self, zones: Sequence[dict], notifier=None, phone_url: Optional[str] = None,
                 phone_token: Optional[str] = None, config: Optional[ResponderConfig] = None,
                 is_airborne: Callable[[], bool] = lambda: False, events: Optional[EventLog] = None,
                 clock: Callable[[], float] = time.time):
        self.centres: Dict[str, tuple] = {z["name"]: centroid(z["polygon"]) for z in zones}
        self.notifier, self.phone_url, self.phone_token = notifier, phone_url, phone_token
        self.cfg = config or ResponderConfig()
        self.is_airborne, self.events, self._clock = is_airborne, events or EventLog(), clock
        self._lock = threading.Lock()
        self._pending: Optional[dict] = None
        self._last_by_source: Dict[str, float] = {}
        self.triggers = self.ignored = 0

    def trigger(self, zone: str, source: str = "sensor") -> dict:
        """Any thread. Raises KeyError for an unknown zone."""
        if zone not in self.centres:
            raise KeyError(zone)
        source = str(source)[:40] or "sensor"
        now = self._clock()
        with self._lock:
            if now - self._last_by_source.get(source, -math.inf) < self.cfg.source_cooldown_s:
                self.ignored += 1
                return {"ok": True, "ignored": "cooldown", "zone": zone}
            self._last_by_source[source] = now
            x, y = self.centres[zone]
            self._pending = {"zone": zone, "source": source, "wall": now, "x": x, "y": y}
            self.triggers += 1
        airborne = bool(self.is_airborne())
        self.events.write("trigger", None, sensor=source, zone=zone, airborne=airborne)
        if self.notifier is not None:
            actions = (launch_actions(self.phone_url, self.phone_token)
                       if not airborne and self.phone_url and self.phone_token else [])
            self.notifier.submit(Alert(zone, 0.0, x, y, 0, wall_time=now, kind="sensor", detail=source,
                                       actions=actions))
        return {"ok": True, "zone": zone, "airborne": airborne}

    def take_pending(self) -> Optional[dict]:
        """Main loop. The latest trigger, once, if it is still fresh."""
        with self._lock:
            p, self._pending = self._pending, None
        if p is not None and self._clock() - p["wall"] > self.cfg.pending_s:
            return None
        return p

    def status(self) -> dict:
        with self._lock:
            p = self._pending
        return {"triggers": self.triggers, "ignored": self.ignored, "zones": sorted(self.centres),
                "pending": None if p is None else {"zone": p["zone"], "source": p["source"],
                                                   "age_s": round(self._clock() - p["wall"], 1)}}


class TriggerServer:
    def __init__(self, responder: Responder, token: str, host: str = "0.0.0.0", port: int = 8091,
                 verbose: bool = True):
        if not token:
            raise ValueError("the trigger server needs a token (--token)")
        self.responder, self.token, self.host, self.port, self.verbose = responder, token, host, port, verbose
        self._httpd: Optional[ThreadingHTTPServer] = None

    def start(self) -> None:
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._handler())
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, name="trigger-server", daemon=True).start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):      # never log query strings (they hold the token)
                if server.verbose:
                    print(f"[trigger] {self.command} {urlparse(self.path).path} -> {args[1] if len(args) > 1 else ''}")

            def _reply(self, code: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _drain(self) -> bytes:
                n = int(self.headers.get("Content-Length") or 0)
                if n > server.responder.cfg.max_body_bytes:
                    while n > 0:                     # read it anyway, or the client may never see our 413
                        n -= len(self.rfile.read(min(n, 65536)) or b"x" * n)
                    return b"__too_big__"
                return self.rfile.read(n) if n > 0 else b""

            def _authorised(self, query: dict) -> bool:
                supplied = query.get("token", [""])[0] or self.headers.get("X-Token", "")
                return hmac.compare_digest(supplied.encode(), server.token.encode())

            def _handle(self, body: bytes) -> None:
                url = urlparse(self.path)
                q = parse_qs(url.query)
                if body == b"__too_big__":
                    return self._reply(413, {"error": "body too large"})
                if not self._authorised(q):
                    return self._reply(401, {"error": "missing or wrong token"})
                if url.path == "/status":
                    return self._reply(200, server.responder.status())
                if url.path != "/trigger":
                    return self._reply(404, {"error": "not found"})
                data = {}
                if body:
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError:
                        return self._reply(400, {"error": "not JSON"})
                    if not isinstance(data, dict):
                        return self._reply(400, {"error": "expected {\"zone\": ..., \"source\": ...}"})
                zone = str(data.get("zone") or q.get("zone", [""])[0])
                source = str(data.get("source") or q.get("source", ["sensor"])[0])
                try:
                    self._reply(200, server.responder.trigger(zone, source))
                except KeyError:
                    self._reply(422, {"error": f"unknown zone {zone!r}", "zones": sorted(server.responder.centres)})

            def do_GET(self):
                self._handle(b"")

            def do_POST(self):
                self._handle(self._drain())

        return Handler
