"""
Score models on our frozen eval split, and gate new ones.

    # detectors: today's YOLOX vs our own
    python tools/benchmark.py detector --data exports/v1 \\
        --model yolo:models/person_yolo.onnx --model own:runs/det_v1/person_own.onnx

    # re-ID: colour histogram vs OpenCV Zoo's net vs ours
    python tools/benchmark.py reid --data exports/v1 --model color \\
        --model onnx:models/person_reid_youtu_2021nov.onnx --model onnx:runs/reid_v1/reid_own.onnx

    # gate: the LAST --model is the candidate, the FIRST is the baseline. Exit code 1
    # if the candidate is not better; with --promote, copy it into models/ if it is.
    python tools/benchmark.py detector --data exports/v1 --gate \\
        --model own:models/person_own.onnx --model own:runs/det_v2/person_own.onnx \\
        --promote models/person_own.onnx

Model specs:  detector  yolo:PATH[:SIZE]  own:PATH[:SIZE]  hog
              reid      color  onnx:PATH  fused:PATH (70 % net + 30 % colour, like --reid fused)
Results are printed and saved as JSON under eval_results/ (one file per run).
"""

import argparse
import json
import math
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.export import load, root_of  # noqa: E402
from evaluation.benchmark import LOW_CONF, bench_detector, bench_reid, headline  # noqa: E402


def make_detector(spec: str):
    from perception.person_detector import CenterPersonDetector, HOGPersonDetector, YoloPersonDetector
    kind, _, rest = spec.partition(":")
    path, _, size = rest.partition(":")
    if kind == "hog":
        return HOGPersonDetector(hit_threshold=-1.0)
    if kind == "yolo":        # OpenCV Zoo YOLOX layout, like the drone's default
        return YoloPersonDetector(path, int(size or 640), "yolox", LOW_CONF, rgb=False, scale_01=False)
    if kind == "own":
        return CenterPersonDetector(path, int(size or 320), LOW_CONF)
    raise SystemExit(f"unknown detector spec {spec!r}")


def make_embedder(spec: str):
    from perception.body_reid import ColorHistogramEmbedder, FusedEmbedder, OnnxReidEmbedder
    kind, _, path = spec.partition(":")
    if kind == "color":
        return ColorHistogramEmbedder()
    if kind == "onnx":
        return OnnxReidEmbedder(path)
    if kind == "fused":
        return FusedEmbedder([(OnnxReidEmbedder(path), 0.7), (ColorHistogramEmbedder(), 0.3)])
    raise SystemExit(f"unknown re-ID spec {spec!r}")


def parse(argv=None):
    p = argparse.ArgumentParser(description="benchmark detectors / re-ID on our eval split")
    p.add_argument("kind", choices=["detector", "reid"])
    p.add_argument("--data", required=True, help="an export folder (tools/data.py export)")
    p.add_argument("--root", help="dataset root (default: the one in the export's manifest)")
    p.add_argument("--model", action="append", required=True, help="model spec, repeatable (see top of file)")
    p.add_argument("--split", default="eval", choices=["eval", "train"], help="eval (default); train = sanity check only")
    p.add_argument("--flight-conf", type=float, default=0.4, help="detector threshold used in flight (config.py)")
    p.add_argument("--gate", action="store_true", help="exit 1 unless the last model beats the first")
    p.add_argument("--min-gain", type=float, default=0.0, help="with --gate: required improvement")
    p.add_argument("--promote", help="with --gate: copy the candidate's file here if it passes")
    p.add_argument("--out-dir", default="eval_results")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    root = root_of(a.data, a.root)
    items = load(a.data, f"{'detector' if a.kind == 'detector' else 'reid'}_{a.split}.json")
    if not items:
        print(f"No {a.split} items in {a.data}: nothing to measure (see tools/data.py export warnings).")
        return 1
    results = {}
    for spec in a.model:
        print(f"== {spec}")
        if a.kind == "detector":
            r = bench_detector(make_detector(spec), items, root, a.flight_conf)
        else:
            r = bench_reid(make_embedder(spec), items, root)
        results[spec] = r
        print(json.dumps(r, indent=1))
    os.makedirs(a.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(a.out_dir, f"{a.kind}-{stamp}.json")
    with open(out, "w") as f:
        json.dump({"kind": a.kind, "data": a.data, "split": a.split, "results": results}, f, indent=1)
    print("\nsummary (" + ("AP50" if a.kind == "detector" else "mAP") + "):")
    for spec, r in results.items():
        print(f"  {headline(a.kind, r):.4f}  {spec}")
    print(f"saved {out}")

    if not a.gate:
        return 0
    if len(a.model) < 2:
        print("--gate needs at least two --model (baseline first, candidate last)")
        return 1
    base, cand = headline(a.kind, results[a.model[0]]), headline(a.kind, results[a.model[-1]])
    passed = math.isfinite(cand) and (not math.isfinite(base) or cand >= base + a.min_gain)
    print(f"GATE {'PASS' if passed else 'FAIL'}: candidate {cand:.4f} vs baseline {base:.4f} (min gain {a.min_gain})")
    if passed and a.promote:
        src = a.model[-1].split(":")[1]
        if os.path.abspath(src) != os.path.abspath(a.promote):
            os.makedirs(os.path.dirname(a.promote) or ".", exist_ok=True)
            shutil.copy2(src, a.promote)
            card = os.path.join(os.path.dirname(src), "model_card.json")
            if os.path.exists(card):
                shutil.copy2(card, os.path.splitext(a.promote)[0] + ".card.json")
            print(f"promoted {src} -> {a.promote}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
