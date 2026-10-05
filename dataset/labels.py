"""
Label rows and the few operations on them (pure functions, no files, no GUI).

One row per frame, in <session>/labels.jsonl:

    {"file": "000123.jpg",
     "people": [{"box": [l, t, r, b], "score": 0.87, "track": 4}, ...],
     "verified": false,          # true once a human has checked this frame
     "by": "yolox_s"}            # who made the boxes: a teacher model name, or "human"

"score" is None for a box a human drew. "track" links the same person across
consecutive frames of one session (-1 = not linked); it is derived data,
recomputed by link_tracks() on every autolabel run.
"""

from typing import Dict, List, Optional, Sequence, Tuple

Box = Tuple[int, int, int, int]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def teacher_row(file: str, scored: Sequence[Tuple[Box, float]], by: str) -> dict:
    return {"file": file, "verified": False, "by": by,
            "people": [{"box": [int(v) for v in b], "score": round(float(s), 3), "track": -1}
                       for b, s in scored]}


def kept_people(row: dict, min_score: float) -> List[dict]:
    """Boxes trusted enough to train on: every human box, teacher boxes above min_score."""
    return [p for p in row.get("people", [])
            if p.get("score") is None or p["score"] >= min_score]


# ---- human edits (used by tools/label_review.py) ----------------------------------

def add_box(row: dict, box: Box) -> dict:
    l, t, r, b = box
    box = (min(l, r), min(t, b), max(l, r), max(t, b))
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return row                                  # a click, not a drag
    people = list(row.get("people", [])) + [{"box": list(box), "score": None, "track": -1}]
    return {**row, "people": people}


def box_at(row: dict, x: float, y: float) -> Optional[int]:
    """Index of the smallest box containing (x, y): with overlapping people the
    one you clicked on is almost always the smaller, nearer-looking box."""
    best, area = None, None
    for i, p in enumerate(row.get("people", [])):
        l, t, r, b = p["box"]
        if l <= x <= r and t <= y <= b:
            a = (r - l) * (b - t)
            if area is None or a < area:
                best, area = i, a
    return best


def delete_box(row: dict, index: int) -> dict:
    people = [p for i, p in enumerate(row.get("people", [])) if i != index]
    return {**row, "people": people}


def accept(row: dict, min_score: float) -> dict:
    """Human says: this frame is right. Teacher boxes under min_score that the
    human left in place are kept (they chose not to delete them), so they are
    promoted to trusted boxes."""
    people = [{**p, "score": p["score"] if p.get("score") is None or p["score"] >= min_score else None}
              for p in row.get("people", [])]
    by = row.get("by", "")
    return {**row, "people": people, "verified": True,
            "by": by if by.endswith("+human") or by == "human" else (by + "+human" if by else "human")}


# ---- linking the same person across frames -------------------------------------------

def link_tracks(rows: List[dict], times: Sequence[float], min_score: float,
                iou_min: float = 0.3, max_gap_s: float = 1.0) -> List[dict]:
    """Greedy IoU tracker over one session's rows (in time order). Each trusted
    box gets a track id; a box continues the track of the best-overlapping box
    from the previous frame if that frame is at most max_gap_s older. Good enough
    for pseudo-identities; not a real multi-object tracker."""
    out, next_id = [], 0
    prev: List[Tuple[List[int], int]] = []           # (box, track) of the last frame
    prev_t: Optional[float] = None
    for row, t in zip(rows, times):
        people = [dict(p, track=-1) for p in row.get("people", [])]
        if prev_t is not None and t - prev_t > max_gap_s:
            prev = []
        pairs = []
        for i, p in enumerate(people):
            if p.get("score") is not None and p["score"] < min_score:
                continue
            for j, (pbox, _) in enumerate(prev):
                v = iou(p["box"], pbox)
                if v >= iou_min:
                    pairs.append((v, i, j))
        used_i, used_j = set(), set()
        for v, i, j in sorted(pairs, reverse=True):
            if i not in used_i and j not in used_j:
                people[i]["track"] = prev[j][1]
                used_i.add(i)
                used_j.add(j)
        for i, p in enumerate(people):
            if p["track"] == -1 and (p.get("score") is None or p["score"] >= min_score):
                p["track"] = next_id
                next_id += 1
        out.append({**row, "people": people})
        prev = [(p["box"], p["track"]) for p in people if p["track"] >= 0]
        prev_t = t
    return out


def track_lengths(rows: List[dict]) -> Dict[int, int]:
    n: Dict[int, int] = {}
    for row in rows:
        for p in row.get("people", []):
            if p.get("track", -1) >= 0:
                n[p["track"]] = n.get(p["track"], 0) + 1
    return n
