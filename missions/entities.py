"""
The things the watch reasons about (the ontology), as plain records, plus the
event log that ties them together.

    Sortie   one flight, launch to landing                        id "S20260927-143210"
    Zone     a named area ("back gate", "study")                  zones.json
    Place    a viewpoint: where the drone was + which way it looked   missions/baseline.py
    Track    one anonymous person followed across detections      id "T3", missions/tracks.py
    Object   a COCO thing seen from a Place ("laptop")            missions/baseline.py
    Sensor   a fixed trigger (PIR, door contact)                  missions/responder.py
    Alert    something a human should look at                     below

Every event names the entities it is about by id (track, zone, place, sensor,
sortie), so events.jsonl replays as "what happened, where, involving what".
It is also labelled data for later training.

Anonymous by design: a Track is "someone", never a name. The only identity the
system knows is the enrolled owner, and the owner is skipped, not tracked.
"""

import json
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

Box = Tuple[int, int, int, int]      # left, top, right, bottom (pixels)


@dataclass
class PersonSighting:
    """One person in one detection frame, feet projected onto the ground."""
    x: float
    y: float
    score: float
    box: Optional[Box] = None


@dataclass
class ObjectSighting:
    """One COCO object in one detection frame. No ground position: objects sit on
    tables and shelves, the flat-ground projection would put them metres off."""
    label: str
    score: float
    box: Optional[Box] = None


@dataclass
class Scene:
    """One detection result: what the camera showed, and where the drone was
    when the frame was taken (not when the detector finished)."""
    t: float
    pose: object                      # datatypes.Pose at capture
    pitch_deg: float
    people: List[PersonSighting] = field(default_factory=list)
    objects: List[ObjectSighting] = field(default_factory=list)
    frame: Optional[np.ndarray] = None
    ms: Optional[float] = None        # detector run time
    far: int = 0                      # people found but too far / above the horizon to place


@dataclass
class Alert:
    zone: str
    t: float                  # obs.now of the detection that confirmed it (0 for sensor triggers)
    x: float
    y: float
    people: int               # confirmed people inside that zone right then
    wall_time: float = field(default_factory=time.time)
    snapshot: Optional[str] = None
    kind: str = "intrusion"   # intrusion | object_appeared | object_missing | sensor
    detail: str = ""          # object label, sensor name
    track: Optional[str] = None
    actions: list = field(default_factory=list)   # push buttons (missions/notify.py). Hold tokens:
                                                  # never in to_dict / status / files

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("actions", None)
        d["x"], d["y"] = round(self.x, 1), round(self.y, 1)
        return d


def new_sortie_id(clock: Callable[[], float] = time.time) -> str:
    return time.strftime("S%Y%m%d-%H%M%S", time.localtime(clock()))


def _plain(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (set, tuple)):
        return list(v)
    return str(v)


class EventLog:
    """Append-only JSON lines (one event per line) + the last few in memory for the
    status page. Thread-safe: the sensor server writes from its own thread."""

    def __init__(self, path: Optional[str] = None, keep: int = 50, clock: Callable[[], float] = time.time):
        self.path, self._clock = path, clock
        self.recent: deque = deque(maxlen=keep)
        self.count = 0
        self._lock = threading.Lock()
        self._f = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._f = open(path, "a", encoding="utf-8")

    def write(self, kind: str, t: Optional[float], /, **fields) -> dict:
        """t = flight time (obs.now), None when the event has none (a sensor call arrives on
        its own thread). fields may hold their own "t" / "type" (an alert's dict does): the
        arguments win."""
        e = {**fields, "type": kind, "t": None if t is None else round(float(t), 2),
             "wall": round(self._clock(), 3)}
        with self._lock:
            self.recent.append(e)
            self.count += 1
            if self._f is not None:
                self._f.write(json.dumps(e, default=_plain) + "\n")
                self._f.flush()
        return e

    def close(self) -> None:
        with self._lock:
            if self._f is not None:
                self._f.close()
                self._f = None
