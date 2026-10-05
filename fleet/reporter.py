"""
The drone's link to the fleet console (fleet/server.py), in a background
thread, so the flight loop never waits on the network.

Out: the drone's latest status, once per interval. Only the MOST RECENT
status is ever sent -- if the console is slow or unreachable, older ones are
dropped rather than queued, so a network hiccup can never make the reporter
fall behind and flood stale data once the link recovers.

In: the reply to each report carries the operator's queued tasking (task
switches, hover height, a target photo, a calibration request). Those go
into the SAME EnrollmentBox / CommandBox the phone page uses, so the main
loop handles them identically, between frames, on its own thread. Their
results come back to the console as acks in the next report.

Nothing the drone does depends on this link: a dead console just means the
dashboard's "age" column ticks up.

    reporter = FleetReporter("http://<fleet-host>:8090", "drone-1", token,
                             enrollment=box, commands=commands)
    reporter.start()
    reporter.update(board)   # every loop tick; cheap, never blocks
    reporter.stop()
"""

import base64
import json
import threading
import time
import urllib.request
from concurrent.futures import Future
from typing import Callable, List, Optional, Tuple

Poster = Callable[[str, str, dict, float], dict]


def default_poster(url: str, token: str, body: dict, timeout_s: float) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/report?token={token}", data=data,
                                 method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def decode_jpeg_b64(text: str):
    import cv2
    import numpy as np
    data = base64.b64decode(text, validate=True)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("the console sent a photo that could not be decoded")
    return image


class FleetReporter:
    def __init__(self, url: str, drone_id: str, token: str, interval_s: float = 1.0,
                 timeout_s: float = 3.0, poster: Optional[Poster] = None,
                 on_error: Optional[Callable[[Exception], None]] = None,
                 enrollment=None, commands=None, ack_timeout_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic):
        self.url, self.drone_id, self.token = url, drone_id, token
        self.interval_s, self.timeout_s = interval_s, timeout_s
        self._poster = poster or default_poster
        self._on_error = on_error
        self.enrollment, self.commands = enrollment, commands
        self.ack_timeout_s, self._clock = ack_timeout_s, clock
        self._lock = threading.Lock()
        self._latest: Optional[dict] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # only ever touched from the reporter thread (or a test calling post_once directly)
        self._pending: List[Tuple[int, Future, float]] = []   # (item id, reply, give-up time)
        self._acks: List[dict] = []                           # finished, not yet delivered

    def update(self, status: dict) -> None:
        """Call every loop tick. Cheap (just replaces an in-memory dict) --
        never touches the network itself."""
        with self._lock:
            self._latest = dict(status)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + self.timeout_s + 1.0)
            self._thread = None

    def post_once(self) -> bool:
        """Send the latest status (plus any finished acks), then hand whatever
        the console replied with to the main loop. Returns True if there was a
        status to send. Never raises: a failed send drops the status but keeps
        the acks for next time."""
        with self._lock:
            status, self._latest = self._latest, None
        if status is None:
            return False
        self._collect_acks()
        acks = list(self._acks)
        try:
            reply = self._poster(self.url, self.token,
                                 {"drone_id": self.drone_id, **status, "acks": acks}, self.timeout_s)
        except Exception as e:
            if self._on_error:
                self._on_error(e)
            return True
        del self._acks[:len(acks)]
        for item in (reply or {}).get("outbox") or []:
            self._dispatch(item)
        return True

    # ---- console -> main loop ---------------------------------------------
    def _dispatch(self, item: dict) -> None:
        item_id, kind = item.get("id"), item.get("kind")
        try:
            if kind == "command":
                if self.commands is None:
                    raise RuntimeError("this drone was started without command support")
                fut = self.commands.submit(item["command"])
            elif kind in ("target", "clear_target"):
                if self.enrollment is None:
                    raise RuntimeError("this drone was started without target upload support")
                image = decode_jpeg_b64(item["image_jpeg_b64"]) if kind == "target" else None
                fut = self.enrollment.submit(image)
            else:
                raise ValueError(f"unknown tasking kind {kind!r}")
        except Exception as e:
            self._acks.append({"id": item_id, "ok": False, "error": str(e)})
            return
        self._pending.append((item_id, fut, self._clock() + self.ack_timeout_s))

    def _collect_acks(self) -> None:
        now, still = self._clock(), []
        for item_id, fut, give_up in self._pending:
            if fut.done():
                try:
                    self._acks.append({"id": item_id, "ok": True, "result": fut.result()})
                except Exception as e:
                    self._acks.append({"id": item_id, "ok": False, "error": str(e)})
            elif now > give_up:
                self._acks.append({"id": item_id, "ok": False,
                                   "error": "the drone's main loop did not handle it in time"})
            else:
                still.append((item_id, fut, give_up))
        self._pending = still

    def _run(self) -> None:
        while not self._stop.is_set():
            self.post_once()
            self._stop.wait(self.interval_s)
