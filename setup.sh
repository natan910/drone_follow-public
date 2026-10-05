#!/usr/bin/env bash
# One-command bootstrap: the known-good recipe for this project (Python 3.11-3.13,
# wheel-only installs, opencv-python pinned below 5.0 -- see requirements-core.txt
# for why), instead of re-discovering it turn by turn.
#
# On a Raspberry Pi with the camera stack installed (`sudo apt install -y python3-picamera2`)
# it builds the venv from the system Python with --system-site-packages, because picamera2
# exists only as an apt package compiled for the system Python (3.13 on Pi OS Trixie):
# a venv made from any other Python cannot import it, and `--camera pi` needs it.
#
#   ./setup.sh              venv + core deps (numpy, opencv) + run the tests
#   ./setup.sh --full       also installs insightface/onnxruntime/pymavlink/pyserial
#   ./setup.sh --models     also downloads the body re-ID ONNX models
#   ./setup.sh --train      also installs PyTorch for training our own models (TRAINING.md)
#   ./setup.sh --full --models --no-test   all of the above, skip the test run
#   ./setup.sh --pi         force the Pi recipe (system Python + --system-site-packages);
#                           normally detected automatically
#
# Safe to re-run any time; every step is skip-if-already-done.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

FULL=0
MODELS=0
TRAIN=0
RUN_TEST=1
PI=0
for arg in "$@"; do
  case "$arg" in
    --full)    FULL=1 ;;
    --models)  MODELS=1 ;;
    --train)   TRAIN=1 ;;
    --no-test) RUN_TEST=0 ;;
    --pi)      PI=1 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

# --- Python ------------------------------------------------------------
# Known-good: 3.11 (Intel Mac + ArduPilot SITL), 3.12 (CI), 3.13 (Raspberry Pi 5,
# Pi OS Trixie, 2026-10-02: opencv, insightface, onnxruntime, pymavlink, pyserial
# all installed from wheels). CI runs 3.11-3.13.
KNOWN_GOOD="3.11 3.12 3.13"
VENV_OPTS=""
SYS_PY=/usr/bin/python3
if [ "$PI" = "0" ] && [ "$(uname -s)" = "Linux" ] && [ -x "$SYS_PY" ] \
   && "$SYS_PY" -c "import picamera2" >/dev/null 2>&1; then
  PI=1
  echo "==> picamera2 found in the system Python: Raspberry Pi recipe"
fi
if [ "$PI" = "1" ]; then
  if ! "$SYS_PY" -c "import picamera2" >/dev/null 2>&1; then
    echo "--pi: the system Python ($SYS_PY) cannot import picamera2." >&2
    echo "Install it first: sudo apt install -y python3-picamera2" >&2
    exit 1
  fi
  PYTHON="$SYS_PY"
  VENV_OPTS="--system-site-packages"
else
  PYTHON=""
  for cand in python3.13 python3.12 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
  done
  if [ -z "$PYTHON" ]; then
    echo "No python3 found on PATH." >&2
    exit 1
  fi
fi
PY_VER="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case " $KNOWN_GOOD " in
  *" $PY_VER "*) ;;
  *) echo "Warning: using Python $PY_VER ($PYTHON). Known-good: $KNOWN_GOOD;" >&2
     echo "onnxruntime/insightface in particular may have no wheel for $PY_VER yet." >&2 ;;
esac

# --- venv ------------------------------------------------------------------
if [ ! -d .venv ]; then
  echo "==> Creating .venv with $PYTHON ($PY_VER)${VENV_OPTS:+ $VENV_OPTS}"
  "$PYTHON" -m venv $VENV_OPTS .venv
else
  echo "==> .venv already exists, reusing it"
  if [ "$PI" = "1" ] && ! grep -qi "include-system-site-packages *= *true" .venv/pyvenv.cfg; then
    echo "    WARNING: this .venv cannot see the system picamera2 (made without --system-site-packages)." >&2
    echo "    --camera pi will fail. Fix: rm -rf .venv && ./setup.sh" >&2
  fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip -q

# --- core deps ---------------------------------------------------------
echo "==> Installing core dependencies (numpy, opencv-python<5)"
pip install --only-binary=":all:" -q -r requirements-core.txt
python -c "import cv2; cv2.HOGDescriptor" || {
  echo "cv2.HOGDescriptor missing -- got an unexpected opencv-python version." >&2
  python -c "import cv2; print('installed:', cv2.__version__)" >&2
  exit 1
}
echo "    opencv-python $(python -c 'import cv2; print(cv2.__version__)') OK (HOGDescriptor present)"
pip install -q pytest      # the suite is unittest-style; pytest gives the short failure reports

if [ "$FULL" = "1" ]; then
  echo "==> Installing full extras (insightface, onnxruntime, pymavlink, pyserial)"
  echo "opencv-python<5" > /tmp/df_constraints.txt   # stop these pulling opencv back to 5.x
  pip install --only-binary=":all:" -c /tmp/df_constraints.txt -q \
    insightface onnxruntime pymavlink pyserial
  rm -f /tmp/df_constraints.txt
fi

if [ "$TRAIN" = "1" ]; then
  echo "==> Installing training extras (torch, onnx) from requirements-train.txt"
  if [ "$(uname -s)" = "Darwin" ] && [ "$(python -c 'import platform; print(platform.machine())')" != "arm64" ]; then
    echo "    WARNING: this Python is $(python -c 'import platform; print(platform.machine())'), not arm64." >&2
    echo "    On an Apple Silicon Mac that means Rosetta: no Apple GPU, and no new torch wheels." >&2
    echo "    Fix: rm -rf .venv, install an arm64 Python 3.13 (python.org universal2), rerun ./setup.sh --train" >&2
  fi
  echo "opencv-python<5" > /tmp/df_constraints.txt
  pip install -c /tmp/df_constraints.txt -q -r requirements-train.txt
  rm -f /tmp/df_constraints.txt
  python - <<'PY'
import torch
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"    torch {torch.__version__}, training device: {dev}")
PY
  echo "    next: python -m training.train smoke"
fi

# --- models --------------------------------------------------------------
if [ "$MODELS" = "1" ]; then
  echo "==> Downloading body re-ID models into models/"
  mkdir -p models
  fetch() {  # fetch <url> <dest>
    if [ -f "$2" ]; then
      echo "    $2 already present, skipping"
    else
      curl -fL -o "$2" "$1"
      echo "    $2: $(shasum -a 256 "$2" 2>/dev/null || sha256sum "$2")"
    fi
  }
  fetch "https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx" \
        "models/person_yolo.onnx"
  fetch "https://github.com/opencv/opencv_zoo/raw/main/models/person_reid_youtureid/person_reid_youtu_2021nov.onnx" \
        "models/person_reid_youtu_2021nov.onnx"
  fetch "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
        "models/face_detection_yunet_2023mar.onnx"
  fetch "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
        "models/face_recognition_sface_2021dec.onnx"
  echo "    (no pinned checksums here -- these are third-party files that can be"
  echo "     revised upstream; the sha256 above is printed so you can note it down"
  echo "     and compare on a future re-download if you want to detect changes)"
fi

if [ "$RUN_TEST" = "1" ]; then
  echo "==> Running the test suite"
  python -m unittest discover -s tests -t . -q
  echo "==> All good. Try: python main.py --platform sim --show --seconds 60"
fi
