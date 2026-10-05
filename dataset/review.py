"""
Our own label review tool: fix the teacher's boxes by hand. OpenCV window,
no extra installs, no server.

    python tools/data.py review --split eval        # the benchmark frames first

Keys (window must have focus):
    SPACE / ENTER   frame is right -> mark verified, save, next frame
    d / a           next / previous frame (no change)
    u               jump to the next frame nobody verified yet
    x               remove every box (there is no person in this frame)
    z               undo the last change on this frame
    q / ESC         save and quit
Mouse:
    left-drag       draw a missing box
    right-click     delete the box under the pointer

Colours: green = trusted box (human, or teacher above --min-score);
grey = teacher box below --min-score (ignored for training unless you accept
the frame with it left in); a thick border means the frame is verified.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from dataset.labels import accept, add_box, box_at, delete_box
from dataset.layout import image_path, read_frames, read_labels, sessions_in, write_labels

WINDOW = "label review  (SPACE ok, d/a next/prev, u unverified, x empty, z undo, q quit)"


class Reviewer:
    def __init__(self, root: str, split: Optional[str], min_score: float = 0.5,
                 only_unverified: bool = False, max_width: int = 1280):
        self.root, self.min_score, self.max_width = root, min_score, max_width
        self.labels: Dict[str, Dict[str, dict]] = {}
        self.items: List[Tuple[str, str]] = []
        for sid in sessions_in(root, split):
            lab = read_labels(root, sid)
            if not lab:
                continue
            self.labels[sid] = lab
            for fr in read_frames(root, sid):
                row = lab.get(fr["file"])
                if row is not None and not (only_unverified and row.get("verified")):
                    self.items.append((sid, fr["file"]))
        self.i = 0
        self.undo: List[dict] = []
        self.drag: Optional[Tuple[int, int]] = None
        self.pointer = (0, 0)
        self.scale = 1.0
        self.dirty: set = set()

    # ---- state helpers -----------------------------------------------------
    def row(self) -> dict:
        sid, f = self.items[self.i]
        return self.labels[sid][f]

    def set_row(self, new: dict) -> None:
        sid, f = self.items[self.i]
        self.undo.append(self.labels[sid][f])
        self.labels[sid][f] = new
        self.dirty.add(sid)

    def save(self) -> None:
        for sid in sorted(self.dirty):
            write_labels(self.root, sid, self.labels[sid])
        self.dirty.clear()

    def go(self, k: int) -> None:
        self.i = int(np.clip(self.i + k, 0, len(self.items) - 1))
        self.undo = []

    def next_unverified(self) -> None:
        for j in list(range(self.i + 1, len(self.items))) + list(range(0, self.i)):
            sid, f = self.items[j]
            if not self.labels[sid][f].get("verified"):
                self.i, self.undo = j, []
                return

    # ---- drawing -------------------------------------------------------------
    def render(self) -> np.ndarray:
        sid, f = self.items[self.i]
        img = cv2.imread(image_path(self.root, sid, f))
        if img is None:
            img = np.zeros((480, 640, 3), np.uint8)
        row = self.row()
        for p in row.get("people", []):
            l, t, r, b = p["box"]
            low = p.get("score") is not None and p["score"] < self.min_score
            colour = (150, 150, 150) if low else (0, 220, 0)
            cv2.rectangle(img, (l, t), (r, b), colour, 1 if low else 2)
            if p.get("score") is not None:
                cv2.putText(img, f"{p['score']:.2f}", (l, max(12, t - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        self.scale = min(1.0, self.max_width / img.shape[1])
        if self.scale < 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        if self.drag is not None:
            cv2.rectangle(img, self.drag, self.pointer, (255, 200, 0), 1)
        if row.get("verified"):
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 220, 0), 6)
        done = sum(1 for s, ff in self.items if self.labels[s][ff].get("verified"))
        text = f"{self.i + 1}/{len(self.items)}  verified {done}  {sid}/{f}  by {row.get('by', '?')}"
        cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    # ---- input -----------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, param) -> None:
        self.pointer = (x, y)
        k = 1.0 / self.scale
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag is not None:
            (x0, y0), self.drag = self.drag, None
            new = add_box(self.row(), (int(x0 * k), int(y0 * k), int(x * k), int(y * k)))
            if new is not self.row():
                self.set_row(new)
        elif event == cv2.EVENT_RBUTTONDOWN:
            j = box_at(self.row(), x * k, y * k)
            if j is not None:
                self.set_row(delete_box(self.row(), j))

    def on_key(self, key: int) -> bool:
        """False = quit."""
        if key in (ord("q"), 27):
            return False
        if key in (32, 13, 10):
            self.set_row(accept(self.row(), self.min_score))
            self.save()
            self.go(1)
        elif key == ord("d"):
            self.go(1)
        elif key == ord("a"):
            self.go(-1)
        elif key == ord("u"):
            self.next_unverified()
        elif key == ord("x"):
            self.set_row({**self.row(), "people": []})
        elif key == ord("z") and self.undo:
            sid, f = self.items[self.i]
            self.labels[sid][f] = self.undo.pop()
            self.dirty.add(sid)
        return True

    def run(self) -> int:
        if not self.items:
            print("Nothing to review: record, then run `python tools/data.py autolabel` first.")
            return 1
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        try:
            while True:
                cv2.imshow(WINDOW, self.render())
                key = cv2.waitKey(30) & 0xFF
                if key != 255 and not self.on_key(key):
                    break
        finally:
            self.save()
            cv2.destroyAllWindows()
        return 0
