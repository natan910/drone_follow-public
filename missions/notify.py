"""
Push alerts to a phone, off the flight loop.

    main.py ... --mission perimeter --zones zones.json --alert-url https://ntfy.sh/<your-topic> [--alert-token tk_...]

Speaks the ntfy protocol (plain HTTP; ntfy is open source and self-hostable,
its phone app shows the push with the snapshot). Setup: PATROL.md "Push alerts".

  - With a snapshot: PUT <url>, body = the JPEG, headers Filename / Title / Message.
  - Without one:     POST <url>, body = the text, header Title.
  - Buttons (alert.actions, e.g. the R4 "Launch" button): header Actions, ntfy's
    simple format. An "http" button is sent BY THE PHONE when tapped, so the
    drone's address in it must be reachable from the phone (same Wi-Fi).

Privacy: the snapshot leaves the drone. Use your own server (or at least an
unguessable topic name + an access token) and https. The public ntfy.sh server
would see every photo, and the Launch button's token.

Never blocks the caller: submit() only queues. A full queue drops the OLDEST
alert (the newest is the one you want). Each alert gets `notify_retries`
attempts, `notify_retry_s` apart. Stats go in the mission status ("notify").
"""

import os
import queue
import threading
import time
import urllib.request
from typing import Callable, Optional

from missions.mission_config import PerimeterConfig

# send(method, url, body, headers, timeout_s) -> HTTP status code (raises on network errors)
Sender = Callable[[str, str, bytes, dict, float], int]


def _urllib_send(method: str, url: str, body: bytes, headers: dict, timeout_s: float) -> int:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return r.status


def _header(text: str) -> str:
    """HTTP headers are latin-1: keep them plain ASCII."""
    return text.encode("ascii", "replace").decode("ascii").replace("\n", " ")


def _quote(v: str) -> str:
    """ntfy simple format: quote a value that holds a comma or semicolon."""
    v = str(v)
    if any(ch in v for ch in ",;"):
        q = "'" if "'" not in v else '"'
        return f"{q}{v}{q}"
    return v


def format_actions(actions) -> str:
    """[{"action": "http", "label", "url", "method", "headers": {...}, "body", "clear"},
        {"action": "view", "label", "url"}] -> ntfy's Actions header (simple format)."""
    out = []
    for a in actions[:3]:                         # ntfy shows at most 3 buttons
        parts = [a["action"], _quote(a["label"]), _quote(a["url"])]
        if a.get("method"):
            parts.append(f"method={a['method']}")
        for k, v in (a.get("headers") or {}).items():
            parts.append(f"headers.{k}={_quote(v)}")
        if a.get("body") is not None:
            parts.append(f"body={_quote(a['body'])}")
        if a.get("clear"):
            parts.append("clear=true")
        out.append(", ".join(parts))
    return _header("; ".join(out))


def describe(alert, drone_id: str):
    """(title, text, priority) for a phone notification."""
    stamp = time.strftime("%H:%M:%S", time.localtime(alert.wall_time))
    where = f"{alert.x:.0f} m E / {alert.y:.0f} m N of launch"
    if alert.kind == "intrusion":
        who = f" ({alert.track})" if alert.track else ""
        return (f"{drone_id}: person in {alert.zone}",
                f"{alert.people} person(s){who} in {alert.zone!r} at {stamp}, {where}", "high")
    if alert.kind == "sensor":
        return (f"{drone_id}: {alert.detail} triggered at {alert.zone}",
                f"Sensor {alert.detail!r} at {alert.zone!r}, {stamp}. The drone will check it once airborne.",
                "high")
    if alert.kind == "object_missing":
        return (f"{drone_id}: {alert.detail} missing ({alert.zone})",
                f"Usually seen near {alert.zone!r}, not there at {stamp}, {where}", "default")
    if alert.kind == "object_appeared":
        return (f"{drone_id}: new {alert.detail} ({alert.zone})",
                f"Not usually there: {alert.detail} near {alert.zone!r} at {stamp}, {where}", "default")
    return f"{drone_id}: {alert.kind} {alert.zone}", f"{alert.detail} {stamp}, {where}", "default"


class AlertNotifier:
    def __init__(self, url: str, token: Optional[str] = None, config: Optional[PerimeterConfig] = None,
                 send: Optional[Sender] = None, sleep: Callable[[float], None] = time.sleep,
                 read_file: Optional[Callable[[str], bytes]] = None, threaded: bool = True,
                 drone_id: str = "drone"):
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"--alert-url must start with http:// or https:// (got {url!r})")
        self.url, self.token, self.drone_id = url, token, drone_id
        self.cfg = config or PerimeterConfig()
        self._send = send or _urllib_send
        self._sleep = sleep
        self._read = read_file or _read_bytes
        self._q: "queue.Queue" = queue.Queue(maxsize=max(1, self.cfg.notify_queue))
        self.sent = self.failed = self.dropped = 0
        self.last_error: Optional[str] = None
        self._stop = threading.Event()
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._run, name="alert-notifier", daemon=True)
            self._thread.start()

    # ---- caller side (any thread) -----------------------------------------------
    def submit(self, alert) -> None:
        while True:
            try:
                self._q.put_nowait(alert)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def stats(self) -> dict:
        return {"sent": self.sent, "failed": self.failed, "dropped": self.dropped,
                "queued": self._q.qsize(), "last_error": self.last_error}

    def close(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)

    # ---- worker side -------------------------------------------------------------
    def drain(self) -> None:
        """Send everything queued, on the calling thread (tests, threaded=False)."""
        while True:
            try:
                alert = self._q.get_nowait()
            except queue.Empty:
                return
            self._deliver(alert)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                alert = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            self._deliver(alert)

    def _deliver(self, alert) -> None:
        method, body, headers = self._request(alert)
        for attempt in range(max(1, self.cfg.notify_retries)):
            if attempt:
                self._sleep(self.cfg.notify_retry_s)
            try:
                code = self._send(method, self.url, body, headers, self.cfg.notify_timeout_s)
                if 200 <= code < 300:
                    self.sent += 1
                    self.last_error = None
                    return
                self.last_error = f"HTTP {code}"
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
        self.failed += 1
        print(f"Alert notification failed after {self.cfg.notify_retries} tries: {self.last_error}")

    def _request(self, alert):
        title, text, priority = describe(alert, self.drone_id)
        headers = {"Title": _header(title), "Priority": priority,
                   "Tags": "rotating_light" if priority == "high" else "mag"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if getattr(alert, "actions", None):
            headers["Actions"] = format_actions(alert.actions)
        photo = None
        if alert.snapshot:
            try:
                photo = self._read(alert.snapshot)
            except OSError as e:
                self.last_error = f"snapshot unreadable: {e}"
        if photo:
            headers.update({"Filename": _header(os.path.basename(alert.snapshot)),
                            "Message": _header(text), "Content-Type": "image/jpeg"})
            return "PUT", photo, headers
        headers["Content-Type"] = "text/plain; charset=utf-8"
        return "POST", text.encode("utf-8"), headers


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
