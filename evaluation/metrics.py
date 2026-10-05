"""
Scores, numpy only.

Detector
    average_precision   area under the precision/recall curve at one IoU (0.5 by
                        default): 1.0 = every person found, nothing invented.
    recall_by_size      share of people found, split by how tall they are in the
                        frame: small = far away = what a drone mostly sees.
Re-ID
    reid_scores         rank-1 (is the closest other crop the same person?) and
                        mAP (are ALL their crops ranked above strangers?). Crops of
                        the query's own track never count (near-duplicates).
    suggest_thresholds  where to set acquire/keep in config.py for this model.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dataset.labels import iou

Box = Sequence[float]


def match_frame(pred: Sequence[Tuple[Box, float]], gt: Sequence[Box], iou_min: float
                ) -> Tuple[List[Tuple[float, bool]], List[bool]]:
    """Greedy (highest score first) one-to-one matching in one frame.
    Returns ([(score, is_true_positive)], [gt was found])."""
    found = [False] * len(gt)
    out = []
    for box, score in sorted(pred, key=lambda p: -p[1]):
        best, best_iou = -1, iou_min
        for j, g in enumerate(gt):
            if not found[j]:
                v = iou(box, g)
                if v >= best_iou:
                    best, best_iou = j, v
        if best >= 0:
            found[best] = True
        out.append((float(score), best >= 0))
    return out, found


def average_precision(frames: Sequence[Tuple[Sequence[Tuple[Box, float]], Sequence[Box]]],
                      iou_min: float = 0.5) -> Dict[str, float]:
    """frames: [(predictions [(box, score)], ground truth [box])]. All-point
    interpolated AP (PASCAL VOC 2010+ style), plus precision/recall at the
    detector's own threshold (every prediction it returned)."""
    scored: List[Tuple[float, bool]] = []
    n_gt = 0
    for pred, gt in frames:
        s, _ = match_frame(pred, gt, iou_min)
        scored += s
        n_gt += len(gt)
    if n_gt == 0:
        return {"ap": float("nan"), "precision": float("nan"), "recall": float("nan"), "gt": 0, "pred": len(scored)}
    scored.sort(key=lambda x: -x[0])
    tp = np.cumsum([t for _, t in scored]) if scored else np.zeros(0)
    fp = np.cumsum([not t for _, t in scored]) if scored else np.zeros(0)
    recall = tp / n_gt if len(tp) else np.zeros(0)
    precision = tp / np.maximum(tp + fp, 1e-9) if len(tp) else np.zeros(0)
    r = np.concatenate([[0.0], recall, [recall[-1] if len(recall) else 0.0]])
    p = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(p) - 2, -1, -1):          # precision envelope
        p[i] = max(p[i], p[i + 1])
    idx = np.nonzero(r[1:] != r[:-1])[0]
    ap = float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))
    return {"ap": round(ap, 4),
            "precision": round(float(precision[-1]), 4) if len(precision) else 0.0,
            "recall": round(float(recall[-1]), 4) if len(recall) else 0.0,
            "gt": n_gt, "pred": len(scored)}


SIZE_BINS = ((0, 48, "small"), (48, 128, "medium"), (128, 1e9, "large"))


def recall_by_size(frames: Sequence[Tuple[Sequence[Tuple[Box, float]], Sequence[Box]]],
                   iou_min: float = 0.5, bins=SIZE_BINS) -> Dict[str, Dict[str, float]]:
    """Recall split by ground-truth box height in pixels (original image)."""
    hit = {name: 0 for _, _, name in bins}
    tot = {name: 0 for _, _, name in bins}
    for pred, gt in frames:
        _, found = match_frame(pred, gt, iou_min)
        for g, f in zip(gt, found):
            h = g[3] - g[1]
            for lo, hi, name in bins:
                if lo <= h < hi:
                    tot[name] += 1
                    hit[name] += f
    return {name: {"recall": round(hit[name] / tot[name], 4) if tot[name] else None, "n": tot[name]}
            for _, _, name in bins}


def reid_scores(emb: np.ndarray, ids: Sequence[str], groups: Sequence[str],
                sessions: Optional[Sequence[str]] = None, cross_session: bool = False) -> Dict[str, float]:
    """Every crop is a query against all the others. Gallery excludes the
    query's own group (track); with cross_session, its whole session too
    (harder: other day, other clothes, other light). Queries with no valid
    positive in the gallery are skipped."""
    emb = np.asarray(emb, np.float32)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    sim = emb @ emb.T
    ids, groups = np.asarray(ids), np.asarray(groups)
    sess = np.asarray(sessions) if sessions is not None else groups
    rank1, aps = [], []
    for q in range(len(ids)):
        valid = groups != groups[q]
        if cross_session:
            valid &= sess != sess[q]
        pos = valid & (ids == ids[q])
        if not pos.any():
            continue
        s = sim[q][valid]
        p = pos[valid]
        order = np.argsort(-s)
        p = p[order]
        rank1.append(bool(p[0]))
        hits = np.cumsum(p)
        prec = hits[p] / (np.nonzero(p)[0] + 1)
        aps.append(float(prec.mean()))
    if not rank1:
        return {"rank1": float("nan"), "map": float("nan"), "queries": 0}
    return {"rank1": round(float(np.mean(rank1)), 4), "map": round(float(np.mean(aps)), 4),
            "queries": len(rank1)}


def suggest_thresholds(emb: np.ndarray, ids: Sequence[str], groups: Sequence[str]
                       ) -> Dict[str, float]:
    """Similarities of same-person pairs (different tracks) vs different-person
    pairs. acquire = the level only 1 % of strangers reach (few false pick-ups);
    keep = the level 90 % of true matches reach (holds on while tracking)."""
    emb = np.asarray(emb, np.float32)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    sim = emb @ emb.T
    ids, groups = np.asarray(ids), np.asarray(groups)
    iu = np.triu_indices(len(ids), 1)
    same = ids[iu[0]] == ids[iu[1]]
    other_group = groups[iu[0]] != groups[iu[1]]
    s = sim[iu]
    pos, neg = s[same & other_group], s[~same]
    if len(pos) == 0 or len(neg) == 0:
        return {}
    acquire = float(np.quantile(neg, 0.99))
    keep = min(float(np.quantile(pos, 0.10)), acquire)     # keep is the LOWER bar (hysteresis)
    return {"acquire": round(acquire, 3), "keep": round(keep, 3),
            "pos_median": round(float(np.median(pos)), 3),
            "neg_median": round(float(np.median(neg)), 3)}
