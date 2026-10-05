# Our own models: data, training, benchmark

Goal: the person detector and the body re-ID net are **ours**: our architecture, our
weights, trained on our footage, measured on our benchmark. Nothing pretrained is
downloaded. The only outside model involved is the *teacher* that pre-draws boxes
(YOLOX-S, Apache-2.0), and humans correct its work.

```
record ──> autolabel ──> review ──> export ──> train ──> benchmark ──> models/
(drone)    (teacher)     (human)    (lists)    (Mac)     (gate)        (flight)
```

## 0. Setup (training machine only; the drone never needs it)
```
./setup.sh --train                  # adds torch + onnx to .venv
python -m training.train smoke      # whole pipeline on synthetic data; must end "SMOKE PASS"
```
`smoke` also checks that cv2.dnn reads our ONNX files with the same numbers as PyTorch
(`cv2.dnn vs torch max diff ... OK`). On Apple Silicon the Python must be arm64:
`python -c "import platform; print(platform.machine())"` must print `arm64` (setup warns otherwise).

## 1. Record
Frames are saved raw (never drawn on) with pose, altitude, camera pitch and mode.
```
# while flying / dry-running: add to your usual main.py command
python main.py ... --record datasets --record-subject subject-01
# no autopilot at all (webcam, drone camera on the bench, phone on a pole):
python tools/data.py record --subject subject-01 --source webcam
python tools/data.py record --camera pi --size 1280x960 --source drone
```
- `--subject NAME` only when that one consented person is the only one in view: it gives
  re-ID a real identity that links sessions. Crowd clips: leave it out (tracks become
  pseudo-identities, one per walk-past).
- 4 fps default (`--record-fps`); roughly 2 GB per hour at 720p, stops at 20 GB per session or
  2 GB free disk. Never slows the control loop (frames are dropped instead).
- Resolution is written to `session.json` on the first frame (`tools/data.py stats` shows it).

What to record (this is the moat; more variety beats more hours):
- altitudes 2 to 15 m, camera pitch 0 to 90 deg, every time of day, sun and overcast;
- people walking, running, sitting, partly hidden, in groups, close and far;
- the same person on different days in different clothes (re-ID needs it);
- as many different consented people as you can (re-ID quality grows with identities);
- empty scenes too (teach the detector what is *not* a person).

## 2. Auto-label
```
python tools/data.py autolabel      # YOLOX-S (models/person_yolo.onnx) boxes every new frame
```
Never overwrites a frame a human verified. `--redo` relabels teacher frames only.

## 3. Review (your own tool, OpenCV window)
```
python tools/data.py review --split eval                    # FIRST: the benchmark frames
python tools/data.py review --split train --only-unverified # then as much train as you like
```
SPACE = correct (verify, save, next) · drag = add box · right-click = delete box ·
`x` = nobody here · `z` = undo · `d`/`a` = next/prev · `u` = next unverified · `q` = quit.

Splits are **by session** (hash of the id, ~15 % eval) and never change, so scores stay
comparable for months. Force a session into a split with `datasets/splits.json`:
`{"eval": ["session-001-subject-01"], "train": []}`. The benchmark only uses **verified**
eval frames: verify every eval frame, or the benchmark measures agreement with the teacher.

## 4. Export
```
python tools/data.py export --out exports/v1
python tools/data.py stats
```
Writes our lists plus COCO json (for route B below) and `manifest.json` (provenance).

## 5. Train
```
python -m training.train detector --data exports/v1 --name det_v1
python -m training.train reid --data exports/v1 --name reid_v1 --init runs/det_v1/best.pt
```
Output in `runs/<name>/`: `person_own.onnx` / `reid_own.onnx`, checkpoints, `log.jsonl`,
`model_card.json` (data, settings, git commit, date). Useful knobs: `--size 416` (detector
input; bigger = finds smaller people, slower on the Pi), `--width 0.5` (faster, weaker),
`--epochs`, `--batch`, `--device cpu|mps|cuda`.

## 6. Benchmark and promote
```
python tools/benchmark.py detector --data exports/v1 \
    --model yolo:models/person_yolo.onnx --model own:runs/det_v1/person_own.onnx
python tools/benchmark.py reid --data exports/v1 --model color \
    --model onnx:models/person_reid_youtu_2021nov.onnx --model onnx:runs/reid_v1/reid_own.onnx
# gate: last model must beat the first; if so it is copied into models/
python tools/benchmark.py detector --data exports/v1 --gate \
    --model own:models/person_own.onnx --model own:runs/det_v2/person_own.onnx --promote models/person_own.onnx
```
Reports AP50/AP75, precision/recall at the flight threshold, recall by person size, AP by
altitude, ms per frame; re-ID rank-1/mAP within and across sessions, and suggested
`acquire_threshold` / `keep_threshold` for `config.py` (our net needs its own; the
defaults are YouTu's). Also run `tools/bench_perception.py` on the Pi: speed matters as much.

Fly with them:
```
python main.py ... --person-detector own --own-model models/person_own.onnx \
    --reid fused --reid-model models/reid_own.onnx
```

## Where to train: alternatives
| | Machine | Use for | Notes |
|---|---|---|---|
| **A (default)** | a local machine with a supported accelerator (`mps`, `cuda`, or CPU) | everything: detector, re-ID, later the face student | data stays local; use the same training code as B. Small models can run on a workstation overnight. |
| B | AWS g4dn (NVIDIA T4, `cuda`) | long runs, several runs in parallel, big face datasets | same commands. Copy `datasets/` + `exports/` up (`rsync`), install the CUDA torch wheel from pytorch.org, `--device cuda`, copy `runs/<name>/` back. Stop the instance after. |
| C | Megvii's YOLOX trainer (Apache-2.0) on B | a YOLOX-nano/tiny/s fine-tuned from COCO weights, if our own net can't match its accuracy | uses `exports/v1/coco_*.json`. Starts from COCO-pretrained weights (Apache-2.0; COCO images have mixed licences, low risk). Heavier setup, CUDA-oriented. Its ONNX runs as `--person-detector yolo --yolo-format yolox --yolo-size 416`. |
| not advised | Colab / Kaggle | | uploads identifiable footage of your people to a third party: outside what they consented to. |

Recommendation: **A for everything**; B only when a run takes too long or blocks the local machine.
One training framework (`training/train.py`) covers every model, on A or B.

## Honest limits
- The detector starts from scratch (no ImageNet weights: cleaner licence, needs more data).
  Expect it to trail YOLOX-S until you have tens of thousands of varied frames; teacher
  pseudo-labels make those frames nearly free. It has ~0.7 M parameters (YOLOX-S: ~9 M), so it
  should be much faster on the Pi than YOLOX-S at 640: check with the benchmark's `ms_median`.
- Re-ID from scratch needs many identities. Few people = it memorises them. Start its
  backbone from the detector (`--init`), and keep `--reid fused` (colour histogram mixed in).
- Pseudo-identities (tracks) can split one person into several ids: mild label noise.
- Identities never come from a face model, so no InsightFace licence reaches these weights.
- GDPR: footage of people is personal data. Keep `datasets/` off GitHub (add it to `.gitignore`),
  keep the consent forms, delete a person's sessions if they withdraw.
