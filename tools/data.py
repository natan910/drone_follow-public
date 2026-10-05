"""
Our own training data, start to finish. One script, five steps:

    # 1. record (no autopilot, nothing flies): drone camera, webcam, phone-on-a-pole...
    python tools/data.py record --subject subject-01 --source webcam
    #    (while flying, use main.py --record datasets --record-subject subject-01 instead)

    # 2. pseudo-label every new frame with the teacher detector (YOLOX-S, Apache-2.0)
    python tools/data.py autolabel

    # 3. fix the teacher's mistakes by hand; do the eval split first, it is the benchmark
    python tools/data.py review --split eval
    python tools/data.py review --split train --only-unverified

    # 4. build the training / eval lists
    python tools/data.py export --out exports/v1

    # 5. what have we got?
    python tools/data.py stats

Everything goes under --root (default datasets/, git-ignored). See TRAINING.md.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.layout import (DEFAULT_ROOT, SessionInfo, list_sessions, load_split_override,  # noqa: E402
                            new_session_id, read_frames, read_labels, read_session, split_of)


def cmd_record(a) -> int:
    import cv2
    from dataset.recorder import Recorder
    from perception.stream_input import PiCameraSource, WebcamSource
    size = tuple(int(v) for v in a.size.split("x")) if a.size else None
    if a.camera == "pi":
        cam = PiCameraSource(size or (1280, 960))
    else:
        cam = WebcamSource(a.camera_index, size)
    info = SessionInfo(new_session_id(a.subject), subject=a.subject, source=a.source,
                       camera=a.camera if a.camera == "pi" else f"webcam:{a.camera_index}",
                       started=time.strftime("%Y-%m-%dT%H:%M:%S"), notes=a.notes or "")
    rec = Recorder(a.root, info, fps=a.fps, max_gb=a.max_gb)
    print(f"Recording to {rec.dir} at {a.fps:g} fps. q in the window or Ctrl-C to stop.")
    t0 = time.monotonic()
    try:
        while a.seconds is None or time.monotonic() - t0 < a.seconds:
            frame = cam.read()
            if frame is None:
                time.sleep(0.02)
                continue
            rec.maybe_record(frame)
            if not a.no_window:
                view = frame.copy()                   # never draw on the recorded frame
                st = rec.status()
                cv2.putText(view, f"REC {st['frames']} frames {st['mb']} MB", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("record", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if rec.stopped:
                break
    except KeyboardInterrupt:
        pass
    finally:
        rec.close()
        cam.release()
        cv2.destroyAllWindows()
    print(f"Done: {rec.status()}  resolution {info.width}x{info.height}")
    return 0


def cmd_autolabel(a) -> int:
    from dataset.autolabel import autolabel_session, make_teacher
    if not os.path.exists(a.teacher):
        print(f"Teacher model {a.teacher} not found: run ./setup.sh --models")
        return 1
    teacher = make_teacher(a.teacher, a.teacher_size)
    by = os.path.splitext(os.path.basename(a.teacher))[0]
    sessions = a.session or list_sessions(a.root)
    for sid in sessions:
        r = autolabel_session(a.root, sid, teacher, by, a.min_score, a.redo,
                              progress=lambda i, n: print(f"  {sid}: {i}/{n}", end="\r"))
        print(f"{sid}: labelled {r['labelled']}, kept {r['kept']}, missing {r['missing_images']}")
    return 0


def cmd_review(a) -> int:
    from dataset.review import Reviewer
    return Reviewer(a.root, a.split, a.min_score, a.only_unverified).run()


def cmd_export(a) -> int:
    from dataset.export import export
    m = export(a.root, a.out, min_score=a.min_score,
               eval_verified_only=not a.eval_unverified, reid_every=a.reid_every)
    for split, c in m["counts"].items():
        print(f"{split:5s}: {c}")
    if m["counts"]["eval"]["frames"] == 0:
        print("WARNING: no eval frames. Verify some eval-split frames (review --split eval), or put "
              "sessions in datasets/splits.json under \"eval\". Without them the benchmark measures nothing.")
    print(f"Wrote {a.out}")
    return 0


def cmd_stats(a) -> int:
    override = load_split_override(a.root)
    tot = {"train": [0, 0, 0], "eval": [0, 0, 0]}
    for sid in list_sessions(a.root):
        info = read_session(a.root, sid)
        frames, labels = read_frames(a.root, sid), read_labels(a.root, sid)
        ver = sum(1 for r in labels.values() if r.get("verified"))
        split = split_of(sid, override)
        tot[split][0] += len(frames)
        tot[split][1] += len(labels)
        tot[split][2] += ver
        print(f"{sid:32s} {split:5s} {info.source:7s} {info.width}x{info.height}  subject={info.subject or '-':10s} "
              f"frames {len(frames):6d}  labelled {len(labels):6d}  verified {ver:5d}")
    for split, (f, l, v) in tot.items():
        print(f"TOTAL {split:5s}: frames {f}  labelled {l}  verified {v}")
    return 0


def parse(argv=None):
    p = argparse.ArgumentParser(description="record / label / export our own training data")
    p.add_argument("--root", default=DEFAULT_ROOT, help="dataset folder (default: datasets/)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="record frames without the autopilot")
    r.add_argument("--subject", help="the ONE consented person in view (omit if several)")
    r.add_argument("--source", default="webcam", help="drone / webcam / pole ...")
    r.add_argument("--camera", choices=["webcam", "pi"], default="webcam")
    r.add_argument("--camera-index", type=int, default=0)
    r.add_argument("--size", help="request a resolution, e.g. 1280x720 (default: camera's own)")
    r.add_argument("--fps", type=float, default=4.0)
    r.add_argument("--max-gb", type=float, default=20.0)
    r.add_argument("--seconds", type=float)
    r.add_argument("--notes")
    r.add_argument("--no-window", action="store_true")

    al = sub.add_parser("autolabel", help="teacher boxes for every unlabelled frame")
    al.add_argument("--teacher", default="models/person_yolo.onnx")
    al.add_argument("--teacher-size", type=int, default=640)
    al.add_argument("--min-score", type=float, default=0.5, help="teacher boxes trusted from this score")
    al.add_argument("--redo", action="store_true", help="relabel teacher frames too (never human ones)")
    al.add_argument("--session", action="append", help="only this session (repeatable)")

    rv = sub.add_parser("review", help="fix boxes by hand")
    rv.add_argument("--split", choices=["train", "eval"], help="default: both")
    rv.add_argument("--min-score", type=float, default=0.5)
    rv.add_argument("--only-unverified", action="store_true")

    ex = sub.add_parser("export", help="build training / eval lists")
    ex.add_argument("--out", default="exports/latest")
    ex.add_argument("--min-score", type=float, default=0.5)
    ex.add_argument("--eval-unverified", action="store_true",
                    help="also use teacher-only frames for eval (then it measures agreement with the teacher)")
    ex.add_argument("--reid-every", type=int, default=2, help="re-ID crops from every Nth frame (near-duplicates)")

    sub.add_parser("stats", help="per-session counts")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    return {"record": cmd_record, "autolabel": cmd_autolabel, "review": cmd_review,
            "export": cmd_export, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
