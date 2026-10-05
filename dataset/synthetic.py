"""
A fake dataset of painted people (shirt + trousers + head rectangles), laid
out exactly like a real recording. Used by the tests and by
`python -m training.train smoke`, to prove the whole pipeline runs end to end
on a machine before spending hours on real data. Not training data.
"""

import json
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from dataset.labels import link_tracks
from dataset.layout import (SessionInfo, frames_path, image_path, write_jsonl, write_labels,
                            write_session)

CLOTHES = [((30, 30, 200), (120, 70, 40)), ((200, 60, 30), (15, 15, 15)),
           ((40, 170, 40), (120, 170, 190)), ((240, 240, 240), (60, 60, 60)),
           ((20, 200, 220), (90, 30, 30)), ((150, 40, 150), (200, 200, 200))]


def paint_person(img: np.ndarray, box: Tuple[int, int, int, int], shirt, trousers) -> None:
    l, t, r, b = box
    ph, pw = b - t, r - l
    cv2.rectangle(img, (l + pw // 3, t), (r - pw // 3, t + ph // 7), (150, 180, 220), -1)
    cv2.rectangle(img, (l, t + ph // 7), (r, t + ph // 2), shirt, -1)
    cv2.rectangle(img, (l + pw // 8, t + ph // 2), (r - pw // 8, b), trousers, -1)


def scene(rng: np.random.Generator, people: List[Tuple[int, int, int, int, int]],
          w: int = 320, h: int = 240) -> np.ndarray:
    """people: (l, t, r, b, clothes index)."""
    bg = rng.integers(60, 200, 3)
    img = np.clip(np.full((h, w, 3), bg, np.int16) + rng.integers(-20, 20, (h, w, 3)), 0, 255).astype(np.uint8)
    for l, t, r, b, c in people:
        paint_person(img, (l, t, r, b), *CLOTHES[c % len(CLOTHES)])
    return img


def random_box(rng: np.random.Generator, w: int, h: int, cx: Optional[float] = None,
               cy: Optional[float] = None, ph: Optional[int] = None) -> Tuple[int, int, int, int]:
    ph = ph or int(rng.integers(h // 4, int(h * 0.8)))
    pw = max(8, ph // 3)
    cx = cx if cx is not None else rng.uniform(pw / 2, w - pw / 2)
    cy = cy if cy is not None else rng.uniform(ph / 2, h - ph / 2)
    l, t = int(np.clip(cx - pw / 2, 0, w - pw)), int(np.clip(cy - ph / 2, 0, h - ph))
    return l, t, l + pw, t + ph


def make_session(root: str, sid: str, rng: np.random.Generator, n_frames: int = 30,
                 subject: Optional[int] = None, extra_people: int = 0,
                 w: int = 320, h: int = 240, verified: bool = True) -> None:
    """One recording. subject = clothes index of the single named person (their
    identity); otherwise `extra_people` anonymous people walk around."""
    info = SessionInfo(sid, subject=f"p{subject}" if subject is not None else None, source="synthetic",
                       width=w, height=h, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_session(root, info)
    walkers = []
    ids = [subject] if subject is not None else list(rng.integers(0, len(CLOTHES), extra_people))
    for c in ids:
        ph = int(rng.integers(h // 3, int(h * 0.7)))
        walkers.append([rng.uniform(20, w - 20), rng.uniform(ph / 2, h - ph / 2), rng.uniform(-6, 6), ph, int(c)])
    frames, rows = [], []
    for k in range(n_frames):
        people = []
        for wk in walkers:
            wk[0] = float(np.clip(wk[0] + wk[2], 15, w - 15))
            if wk[0] in (15, w - 15):
                wk[2] = -wk[2]
            people.append((*random_box(rng, w, h, wk[0], wk[1], wk[3]), wk[4]))
        img = scene(rng, people, w, h)
        name = f"{k:06d}.jpg"
        cv2.imwrite(image_path(root, sid, name), img)
        frames.append({"file": name, "t": k * 0.25, "alt_m": 2.0, "pitch_deg": 30.0})
        rows.append({"file": name, "verified": verified, "by": "synthetic",
                     "people": [{"box": [l, t, r, b], "score": None, "track": -1} for l, t, r, b, _ in people]})
    write_jsonl(frames_path(root, sid), frames)
    rows = link_tracks(rows, [f["t"] for f in frames], 0.5)
    write_labels(root, sid, {r["file"]: r for r in rows})


def make_dataset(root: str, seed: int = 0, sessions_per_subject: int = 2, n_frames: int = 30) -> List[str]:
    """Every clothes set as a subject in a few sessions, plus crowd sessions.
    Writes splits.json so the eval split surely holds subjects seen twice."""
    rng = np.random.default_rng(seed)
    made, held_out = [], []
    for c in range(len(CLOTHES)):
        for j in range(sessions_per_subject):
            sid = f"20260101-00{c}{j}00-p{c}"
            make_session(root, sid, rng, n_frames, subject=c)
            made.append(sid)
            if c >= len(CLOTHES) - 2:          # the last two people exist only in eval
                held_out.append(sid)
    for j in range(2):
        sid = f"20260102-00{j}000-na"
        make_session(root, sid, rng, n_frames, extra_people=3)
        made.append(sid)
    held_out.append(made[-1])
    with open(os.path.join(root, "splits.json"), "w") as f:
        json.dump({"eval": held_out, "train": [s for s in made if s not in held_out]}, f, indent=1)
    return made
