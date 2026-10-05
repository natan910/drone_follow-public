"""
Turn recordings + labels into training/eval lists.

    export(root, out_dir) writes:
      detector_train.json / detector_eval.json   frames with person boxes (our format)
      coco_train.json / coco_eval.json           the same in COCO format, for the
                                                 alternative route (Megvii's YOLOX trainer)
      reid_train.json / reid_eval.json           person crops with an identity
      manifest.json                              what went in: sessions, counts, settings

Image paths are relative to the dataset root, so the export folder and the
dataset can be copied to another machine (e.g. AWS) together.

Identities for re-ID (no face model involved, so no licence baggage):
  * the session has a `subject` and the frame shows exactly one person
        -> id "subj:<subject>"   (same id across sessions: the valuable kind)
  * otherwise, a track (same person across consecutive frames) long enough
        -> id "<session>:t<track>"   (a pseudo-identity, only within one session)
`group` = session + track: the benchmark never matches a crop against crops of
its own track (near-duplicates would make re-ID look perfect).
"""

import json
import os
import random
import time
from typing import Dict, List, Optional

from dataset.labels import kept_people, track_lengths
from dataset.layout import (list_sessions, load_split_override, read_frames, read_labels,
                            read_session, split_of)


def _rel(sid: str, file: str) -> str:
    return f"{sid}/frames/{file}"


def export(root: str, out_dir: str, min_score: float = 0.5, negatives_share: float = 0.2,
           eval_verified_only: bool = True, min_reid_px: int = 48, min_track_len: int = 4,
           reid_every: int = 1, seed: int = 0) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    override = load_split_override(root)
    det: Dict[str, List[dict]] = {"train": [], "eval": []}
    neg: Dict[str, List[dict]] = {"train": [], "eval": []}
    reid: Dict[str, List[dict]] = {"train": [], "eval": []}
    used, verified = {"train": [], "eval": []}, {"train": 0, "eval": 0}

    for sid in list_sessions(root):
        split = split_of(sid, override)
        info = read_session(root, sid)
        labels = read_labels(root, sid)
        frames = read_frames(root, sid)
        if not labels:
            continue
        used[split].append(sid)
        rows = [labels[f["file"]] for f in frames if f["file"] in labels]
        lengths = track_lengths(rows)
        meta = {f["file"]: f for f in frames}
        for k, row in enumerate(rows):
            if split == "eval" and eval_verified_only and not row.get("verified"):
                continue
            verified[split] += bool(row.get("verified"))
            people = kept_people(row, min_score)
            fm = meta.get(row["file"], {})
            item = {"image": _rel(sid, row["file"]), "session": sid,
                    "width": info.width, "height": info.height,
                    "boxes": [p["box"] for p in people], "verified": bool(row.get("verified")),
                    "alt_m": fm.get("alt_m"), "pitch_deg": fm.get("pitch_deg")}
            (det if people else neg)[split].append(item)

            if k % reid_every:
                continue
            for p in people:
                l, t, r, b = p["box"]
                if b - t < min_reid_px:
                    continue
                track = p.get("track", -1)
                if info.subject and len(people) == 1:
                    ident = f"subj:{info.subject}"
                elif track >= 0 and lengths.get(track, 0) >= min_track_len:
                    ident = f"{sid}:t{track}"
                else:
                    continue
                reid[split].append({"image": _rel(sid, row["file"]), "box": [l, t, r, b],
                                    "id": ident, "group": f"{sid}:t{track}", "session": sid})

    counts = {}
    for split in ("train", "eval"):
        n_neg = int(len(det[split]) * negatives_share / max(1e-6, 1 - negatives_share))
        rng.shuffle(neg[split])
        det[split] += neg[split][:n_neg]
        _dump(os.path.join(out_dir, f"detector_{split}.json"), det[split])
        _dump(os.path.join(out_dir, f"coco_{split}.json"), to_coco(det[split]))
        _dump(os.path.join(out_dir, f"reid_{split}.json"), reid[split])
        counts[split] = {"sessions": len(used[split]), "frames": len(det[split]),
                         "boxes": sum(len(d["boxes"]) for d in det[split]),
                         "verified_frames": verified[split],
                         "reid_crops": len(reid[split]),
                         "reid_ids": len({r["id"] for r in reid[split]})}
    manifest = {"root": os.path.abspath(root), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "min_score": min_score, "eval_verified_only": eval_verified_only,
                "sessions": used, "counts": counts}
    _dump(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest


def to_coco(items: List[dict]) -> dict:
    images, anns = [], []
    for i, it in enumerate(items):
        images.append({"id": i, "file_name": it["image"], "width": it["width"], "height": it["height"]})
        for l, t, r, b in it["boxes"]:
            anns.append({"id": len(anns), "image_id": i, "category_id": 1, "iscrowd": 0,
                         "bbox": [l, t, r - l, b - t], "area": (r - l) * (b - t)})
    return {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "person"}]}


def load(out_dir: str, name: str) -> list:
    with open(os.path.join(out_dir, name)) as f:
        return json.load(f)


def _dump(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def root_of(out_dir: str, override: Optional[str] = None) -> str:
    """Dataset root for an export: explicit, else the one recorded in its manifest."""
    if override:
        return override
    return load(out_dir, "manifest.json")["root"]
