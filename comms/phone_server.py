"""
Send the drone a photo of who to look for, and steer it live, from any phone browser.

    http://<drone-ip>:8080/?token=<secret>     control page (photo + task buttons + hover height)
    POST   /target   body = the raw image      enrol that face as the target
    DELETE /target                              forget the target (drone goes back to patrolling)
    POST   /command  body = {"task": "HOVER"}  and/or {"hover_height_m": 0.5}
                                                  change what the drone is doing, live
                     body = {"calibrate_distance_m": 2.0}
                                                  one-tap camera calibration (perception/calibration.py)
    GET    /status                              JSON: mode, task, tracker state, map coverage, ...

Every route except the static page needs the token, so join the drone to a
private Wi-Fi network you control and never expose this port to the internet.
Nothing here is encrypted (plain HTTP): treat the token as a light lock, not a vault.

Threading: this runs in its own thread and never touches the matcher or the
autopilot directly. It hands requests to the main loop through EnrollmentBox /
CommandBox and waits for the answer, so both are only ever touched from one thread.

Privacy: the photo is decoded in memory, handed to the matcher (which keeps only
a numeric embedding), and dropped. Nothing is written to disk.

Refusing a body that is too big: the server first reads and throws away the
rest of it (up to DRAIN_FACTOR times the limit), then answers. Answering and
closing while the phone is still sending makes the OS reset the connection,
and the phone shows a network error instead of "photo too large".
"""

import hmac
import json
import os
import queue
import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from datatypes import Task

Request = Tuple[Optional[np.ndarray], "Future[dict]"]  # image (None = clear the target), reply
Command = Tuple[dict, "Future[dict]"]                  # {"task": "HOVER", "hover_height_m": 0.5}, reply

MAX_COMMAND_BYTES = 10_000
DRAIN_FACTOR = 4          # read and discard an oversized body up to this many times the limit
DRAIN_CHUNK = 64 * 1024


class CommandError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def parse_command(body) -> dict:
    """Validate an operator command (from the phone page or the fleet console)
    into the dict RealPlatform understands. Raises CommandError(http_code, why)."""
    if not isinstance(body, dict) or not body:
        raise CommandError(400, 'expected an object, e.g. {"task": "HOVER"} '
                                'and/or {"hover_height_m": 0.5}')
    command: dict = {}
    if "task" in body:
        name = str(body["task"]).upper()
        if name not in Task.__members__:
            raise CommandError(422, f"unknown task {name!r}; choose from {', '.join(Task.__members__)}")
        command["task"] = name
    for key, lo, hi in (("hover_height_m", 0.0, None), ("calibrate_distance_m", 0.5, 10.0)):
        if key not in body:
            continue
        try:
            value = float(body[key])
        except (TypeError, ValueError):
            raise CommandError(422, f"{key} must be a number")
        if value < lo:
            raise CommandError(422, f"{key} cannot be negative" if lo == 0 else f"{key} must be at least {lo:g}")
        if hi is not None and value > hi:
            raise CommandError(422, f"{key} must be at most {hi:g}")
        command[key] = value
    if "launch" in body:          # the launch gate (autonomy/launch.py); only ever `true`
        if body["launch"] is not True:
            raise CommandError(422, 'launch must be true, e.g. {"launch": true}')
        command["launch"] = True
    if not command:
        raise CommandError(422, 'nothing to do: expected "task", "hover_height_m" '
                                'and/or "calibrate_distance_m"')
    return command


class EnrollmentBox:
    """Thread-safe hand-off of enrolment requests from the HTTP thread to the main loop."""

    def __init__(self) -> None:
        self._q: "queue.Queue[Request]" = queue.Queue()

    def submit(self, image: Optional[np.ndarray]) -> "Future[dict]":
        fut: "Future[dict]" = Future()
        self._q.put((image, fut))
        return fut

    def poll(self) -> Optional[Request]:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None


class CommandBox:
    """Thread-safe hand-off of live "what should the drone be doing" changes
    (task switches, hover-height edits) from the HTTP thread to the main loop.
    Mirrors EnrollmentBox so RealPlatform services both the same way."""

    def __init__(self) -> None:
        self._q: "queue.Queue[Command]" = queue.Queue()

    def submit(self, command: dict) -> "Future[dict]":
        fut: "Future[dict]" = Future()
        self._q.put((command, fut))
        return fut

    def poll(self) -> Optional[Command]:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None


PAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phone_page.html")
with open(PAGE_PATH, encoding="utf-8") as _f:
    PAGE = _f.read()   # the page lives in its own file so it can be edited as HTML

MAX_SIDE = 1280  # phone photos are huge; shrink before running a face model on them


class PhoneServer:
    def __init__(self, box: EnrollmentBox, status: Callable[[], dict], token: str,
                 host: str = "0.0.0.0", port: int = 8080, max_bytes: int = 8_000_000,
                 reply_timeout_s: float = 20.0, verbose: bool = True,
                 commands: Optional[CommandBox] = None):
        self.box, self.status, self.token = box, status, token
        self.commands = commands
        self.verbose = verbose
        self.host, self.port = host, port
        self.max_bytes, self.reply_timeout_s = max_bytes, reply_timeout_s
        self._httpd: Optional[ThreadingHTTPServer] = None

    def start(self) -> None:
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._make_handler())
        self.port = self._httpd.server_address[1]  # resolves port 0 to the real one
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()

    # ---- request handling -------------------------------------------------
    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            timeout = 30  # seconds; a stalled sender can't hold a server thread forever

            def log_message(self, fmt, *args):  # never log query strings (they hold the token)
                if server.verbose:
                    print(f"[phone] {self.command} {urlparse(self.path).path} -> "
                          f"{args[1] if len(args) > 1 else ''}")

            def _reply(self, code: int, body, ctype: str = "text/plain; charset=utf-8") -> None:
                data = body if isinstance(body, bytes) else str(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _length(self) -> Optional[int]:
                """Content-Length as a number; None (after a 400 reply) if it is garbage."""
                try:
                    return int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    self._reply(400, "bad Content-Length")
                    self.close_connection = True
                    return None

            def _refuse_body(self, code: int, message: str, length: int, limit: int) -> None:
                """Answer `code` to a body we will not use. Read it first (up to
                DRAIN_FACTOR x limit) so the sender finishes and sees the answer."""
                if length <= DRAIN_FACTOR * limit:
                    remaining = length
                    try:
                        while remaining > 0:
                            chunk = self.rfile.read(min(DRAIN_CHUNK, remaining))
                            if not chunk:
                                break
                            remaining -= len(chunk)
                    except OSError:
                        pass  # sender gave up; answer anyway
                self.close_connection = True
                self._reply(code, message)

            def _authorised(self) -> bool:
                supplied = (parse_qs(urlparse(self.path).query).get("token", [""])[0]
                            or self.headers.get("X-Token", ""))
                if hmac.compare_digest(supplied.encode(), server.token.encode()):
                    return True
                self._reply(401, "missing or wrong token")
                return False

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/":
                    self._reply(200, PAGE, "text/html; charset=utf-8")
                elif path == "/status":
                    if self._authorised():
                        self._reply(200, json.dumps(server.status()), "application/json")
                else:
                    self._reply(404, "not found")

            def do_POST(self):
                path = urlparse(self.path).path
                if path == "/target":
                    return self._post_target()
                if path == "/command":
                    return self._post_command()
                self._reply(404, "not found")

            def _post_target(self):
                if not self._authorised():
                    return
                length = self._length()
                if length is None:
                    return
                if length <= 0:
                    return self._reply(400, "empty body: send the image bytes")
                if length > server.max_bytes:
                    return self._refuse_body(413, f"photo too large (limit {server.max_bytes // 1_000_000} MB)",
                                             length, server.max_bytes)
                image = server._decode(self.rfile.read(length))
                if image is None:
                    return self._reply(415, "that is not an image I can read (try a JPEG or PNG)")
                self._answer(server.box.submit(image))

            def _post_command(self):
                if not self._authorised():
                    return
                length = self._length()
                if length is None:
                    return
                if server.commands is None:
                    return self._refuse_body(501, "this drone was not started with command support",
                                             length, MAX_COMMAND_BYTES)
                if length <= 0 or length > MAX_COMMAND_BYTES:
                    return self._refuse_body(400, "send a small JSON body, e.g. {\"task\": \"HOVER\"}",
                                             length, MAX_COMMAND_BYTES)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    return self._reply(400, "that is not valid JSON")
                try:
                    command = parse_command(body)
                except CommandError as e:
                    return self._reply(e.code, e.message)
                self._answer(server.commands.submit(command))

            def do_DELETE(self):
                if urlparse(self.path).path != "/target":
                    return self._reply(404, "not found")
                if self._authorised():
                    self._answer(server.box.submit(None))

            def _answer(self, fut: "Future[dict]") -> None:
                try:
                    self._reply(200, json.dumps(fut.result(timeout=server.reply_timeout_s)),
                                "application/json")
                except ValueError as e:      # e.g. no face in the photo
                    self._reply(422, str(e))
                except FutureTimeout:
                    self._reply(504, "the drone is busy, try again")
                except Exception as e:
                    self._reply(500, f"enrolment failed: {e}")

        return Handler

    @staticmethod
    def _decode(data: bytes) -> Optional[np.ndarray]:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        h, w = image.shape[:2]
        scale = MAX_SIDE / max(h, w)
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return image
