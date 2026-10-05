"""
Where recordings live on disk, and the small file formats around them.

    <root>/                          default: datasets/ (git-ignored)
      splits.json                    optional override: {"eval": [session ids], "train": [...]}
      <session_id>/
        session.json                 who / what / which camera (one per recording)
        frames/000000.jpg ...        raw camera frames, never drawn on
        frames.jsonl                 one line per frame: file, time, pose, camera pitch, mode
        labels.jsonl                 one line per frame: person boxes (teacher or human), tracks

A session is one continuous recording. Splits are by SESSION, never by frame:
consecutive frames are near-duplicates, so a frame-level split would leak the
test set into training and the benchmark would lie.
"""

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional

DEFAULT_ROOT = "datasets"
EVAL_PERCENT = 15          # share of sessions held out for the benchmark (by hash of the id)


@dataclass
class SessionInfo:
    session_id: str
    subject: Optional[str] = None     # the one consented person in view, if the clip has only them
    source: str = "drone"             # "drone", "webcam", "pole" (camera on a pole), ...
    camera: str = ""                  # e.g. "pi", "webcam:0"
    width: int = 0
    height: int = 0
    started: str = ""                 # local time, ISO format
    notes: str = ""
    consent: bool = True              # every filmed person agreed (owner's rule)
    extra: Dict[str, object] = field(default_factory=dict)


def new_session_id(subject: Optional[str], now: Optional[float] = None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now if now is not None else time.time()))
    return f"{stamp}-{safe_name(subject) if subject else 'na'}"


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s.strip())[:40] or "na"


def session_dir(root: str, session_id: str) -> str:
    return os.path.join(root, session_id)


def write_session(root: str, info: SessionInfo) -> str:
    d = session_dir(root, info.session_id)
    os.makedirs(os.path.join(d, "frames"), exist_ok=True)
    with open(os.path.join(d, "session.json"), "w") as f:
        json.dump(asdict(info), f, indent=2)
    return d


def read_session(root: str, session_id: str) -> SessionInfo:
    with open(os.path.join(session_dir(root, session_id), "session.json")) as f:
        raw = json.load(f)
    known = {k: raw[k] for k in SessionInfo.__dataclass_fields__ if k in raw}
    return SessionInfo(**known)


def list_sessions(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.isfile(os.path.join(root, d, "session.json")))


# ---- JSON lines -----------------------------------------------------------------

def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass     # a line cut short by a crash or power loss: skip it
    return out


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    """Atomic: write a temp file, then rename, so a crash never leaves half a file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def frames_path(root: str, sid: str) -> str:
    return os.path.join(session_dir(root, sid), "frames.jsonl")


def labels_path(root: str, sid: str) -> str:
    return os.path.join(session_dir(root, sid), "labels.jsonl")


def image_path(root: str, sid: str, file: str) -> str:
    return os.path.join(session_dir(root, sid), "frames", file)


def read_frames(root: str, sid: str) -> List[dict]:
    return read_jsonl(frames_path(root, sid))


def read_labels(root: str, sid: str) -> Dict[str, dict]:
    """file name -> label row."""
    return {r["file"]: r for r in read_jsonl(labels_path(root, sid)) if "file" in r}


def write_labels(root: str, sid: str, labels: Dict[str, dict]) -> None:
    write_jsonl(labels_path(root, sid), (labels[k] for k in sorted(labels)))


# ---- splits ------------------------------------------------------------------------

def load_split_override(root: str) -> Dict[str, List[str]]:
    p = os.path.join(root, "splits.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        raw = json.load(f)
    return {k: list(v) for k, v in raw.items() if k in ("train", "eval")}


def split_of(session_id: str, override: Optional[Dict[str, List[str]]] = None) -> str:
    """'eval' or 'train'. The hash never changes, so a session never moves
    between splits: the benchmark stays comparable across months."""
    override = override or {}
    if session_id in override.get("eval", []):
        return "eval"
    if session_id in override.get("train", []):
        return "train"
    h = int(hashlib.sha1(session_id.encode()).hexdigest(), 16) % 100
    return "eval" if h < EVAL_PERCENT else "train"


def sessions_in(root: str, split: Optional[str]) -> Iterator[str]:
    override = load_split_override(root)
    for sid in list_sessions(root):
        if split is None or split_of(sid, override) == split:
            yield sid
