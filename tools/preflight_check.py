"""
Run before every flight (or wire into a systemd ExecStartPre -- see
deploy/drone-follow.service) to catch a broken setup on the ground instead
of mid-air: missing model files, a camera that won't open, range sensors
that aren't talking, an autopilot link that isn't up.

Every check runs independently and reports pass/fail/warn rather than
stopping at the first problem, so one run tells you everything that's
wrong at once.

    python tools/preflight_check.py
        # webcam dry run setup: opencv import, model files for the
        # configured backend/re-ID, camera opens

    python tools/preflight_check.py --driver mavlink --mavlink /dev/serial0 \\
        --ranger "0:/dev/ttyUSB0,-45:/dev/ttyUSB1,45:/dev/ttyUSB2" --camera pi
        # also checks the MAVLink link and every range sensor port

    python tools/preflight_check.py --driver mavlink --autopilot px4 --mavlink udpin:127.0.0.1:14540
        # the link must be up AND the flight controller must run PX4 (default
        # --autopilot auto: either firmware passes, the report says which)

    python tools/preflight_check.py --skip-camera --skip-hardware
        # file/import checks only -- useful in CI or over SSH with no camera

Exit code 0: everything checked either passed or only warned. Exit code 1:
at least one hard failure. Nothing here is destructive -- it never arms,
takes off, or sends a movement command.
"""

import argparse
import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AppConfig  # noqa: E402


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    warning: bool = False  # ok=False + warning=True: shown, but doesn't fail the run

    @property
    def status(self) -> str:
        if self.ok:
            return "PASS"
        return "WARN" if self.warning else "FAIL"


def _ok(name: str, detail: str) -> CheckResult:
    return CheckResult(name, True, detail)


def _fail(name: str, detail: str, warning: bool = False) -> CheckResult:
    return CheckResult(name, False, detail, warning=warning)


# --- individual checks ------------------------------------------------------
# Each takes plain values (paths, indices) rather than argparse.Namespace, and
# anything that touches hardware takes an injectable factory, so tests can
# supply a fake without opening a real camera/port/link.

# 3.11: Intel Mac + ArduPilot SITL; 3.12: CI; 3.13: Raspberry Pi 5 (Pi OS Trixie system Python,
# needed for apt's picamera2). Keep in step with setup.sh KNOWN_GOOD and the CI matrix.
KNOWN_GOOD_PYTHONS = ("3.11", "3.12", "3.13")


def check_python_version(version_info=None) -> CheckResult:
    vi = version_info or sys.version_info
    v = f"{vi[0]}.{vi[1]}"
    if v in KNOWN_GOOD_PYTHONS:
        return _ok("python version", f"{v} (known-good)")
    return _fail("python version", f"{v} -- {'/'.join(KNOWN_GOOD_PYTHONS)} are known-good; "
                 "onnxruntime/insightface may have no wheel for this version yet",
                 warning=True)


def check_opencv() -> CheckResult:
    try:
        import cv2
    except ImportError as e:
        return _fail("opencv import", f"cannot import cv2: {e}")
    return _ok("opencv import", f"cv2 {cv2.__version__}")


def check_model_file(label: str, path: str, exists_fn: Callable[[str], bool] = os.path.isfile) -> CheckResult:
    if exists_fn(path):
        return _ok(label, path)
    return _fail(label, f"{path} not found -- see README / setup.sh --models")


def check_camera(index: int = 0, kind: str = "webcam", opener: Optional[Callable] = None, import_fn: Optional[Callable] = None) -> CheckResult:
    """Opens the camera, reads one frame, releases it. `opener(index) ->
    object with isOpened()/read()/release()` so tests can inject a fake.
    `import_fn(name)` likewise for the pi path's picamera2 import check."""
    if kind == "pi":
        imp = import_fn or importlib.import_module
        try:
            imp("picamera2")
        except ImportError as e:
            return _fail("camera (pi)", f"picamera2 not importable: {e}")
        return _ok("camera (pi)", "picamera2 importable (not opened -- can only run on the Pi itself)")

    import cv2
    open_fn = opener or cv2.VideoCapture
    cap = open_fn(index)
    try:
        if not cap.isOpened():
            return _fail("camera (webcam)", f"could not open camera index {index}")
        ok, frame = cap.read()
        if not ok or frame is None:
            return _fail("camera (webcam)", f"camera {index} opened but returned no frame")
        h, w = frame.shape[:2]
        return _ok("camera (webcam)", f"index {index}, {w}x{h}")
    finally:
        cap.release()


def check_ranger(ports: Dict[float, str], opener: Optional[Callable] = None) -> CheckResult:
    """Opens every configured TF-Luna serial port. `opener(port, baud) ->
    object with close()` (a pyserial Serial or a fake)."""
    if not ports:
        return _ok("range sensors", "none configured")
    if opener is None:
        try:
            import serial
        except ImportError as e:
            return _fail("range sensors", f"pyserial not importable: {e}")
        opener = lambda port, baud: serial.Serial(port, baud, timeout=0)

    opened, problems = [], []
    for bearing, port in ports.items():
        try:
            conn = opener(port, 115200)
            conn.close()
            opened.append(f"{bearing:g}deg:{port}")
        except Exception as e:
            problems.append(f"{bearing:g}deg:{port} ({e})")
    if problems:
        return _fail("range sensors", f"could not open: {', '.join(problems)}"
                     + (f" -- ok: {', '.join(opened)}" if opened else ""))
    return _ok("range sensors", f"opened {len(opened)}: {', '.join(opened)}")


def parse_ranger(spec: str) -> Dict[float, str]:
    """Same syntax as main.py's --ranger: 'bearing_deg:port,...' """
    ports = {}
    for item in spec.split(","):
        bearing, port = item.split(":", 1)
        ports[float(bearing)] = port
    return ports


FIRMWARE = {3: "ardupilot", 12: "px4"}   # MAV_AUTOPILOT_* numbers


def check_mavlink(connection: str, baud: int = 921600, timeout_s: float = 5.0,
                   connector: Optional[Callable] = None, autopilot: str = "auto") -> CheckResult:
    """Connects and waits for one heartbeat from a flight controller (ground
    stations and other components are skipped). `connector(device, baud,
    timeout_s)` returns the MAV_AUTOPILOT number (3 ArduPilot, 12 PX4), or None
    for no heartbeat; a bool (True = heartbeat seen) is also accepted. Tests
    inject a fake without pymavlink.

    autopilot: "auto" accepts either firmware; "ardupilot" / "px4" fail on the other."""
    if connector is None:
        try:
            from pymavlink import mavutil
        except ImportError as e:
            return _fail("mavlink link", f"pymavlink not importable: {e}")
        from control.flight_controller import detect_autopilot

        def connector(device, baud, timeout_s):
            conn = mavutil.mavlink_connection(device, baud=baud)
            try:
                return detect_autopilot(conn, mavutil.mavlink, timeout_s)
            except RuntimeError:
                return None                          # no heartbeat in time
            finally:
                conn.close()

    try:
        seen = connector(connection, baud, timeout_s)
    except Exception as e:
        return _fail("mavlink link", f"{connection}: {e}")
    if seen is None or seen is False:
        return _fail("mavlink link", f"{connection}: no heartbeat within {timeout_s:.0f}s")
    if seen is True:                                 # connector that only knows "heartbeat or not"
        return _ok("mavlink link", f"{connection}: heartbeat received")
    found = FIRMWARE.get(seen)
    if found is None:
        return _fail("mavlink link", f"{connection}: heartbeat from unsupported firmware "
                     f"(MAV_AUTOPILOT {seen}); only ArduPilot and PX4 are supported")
    if autopilot not in ("auto", found):
        return _fail("mavlink link", f"{connection}: expected {autopilot}, but the flight controller runs {found}")
    return _ok("mavlink link", f"{connection}: heartbeat received ({found})")


def check_disk_space(path: str = ".", min_mb: float = 200.0,
                     usage_fn: Callable[[str], object] = shutil.disk_usage) -> CheckResult:
    try:
        free_mb = usage_fn(path).free / 1_000_000
    except OSError as e:
        return _fail("disk space", str(e), warning=True)
    if free_mb < min_mb:
        return _fail("disk space", f"{free_mb:.0f} MB free, below the {min_mb:.0f} MB guideline "
                     "(flight logs / maps need room to grow)", warning=True)
    return _ok("disk space", f"{free_mb:.0f} MB free")


# --- putting it together ----------------------------------------------------

def model_file_checks(cfg: AppConfig, backend: str, reid_enabled: bool,
                      reid_detector: str, reid_embedder: str,
                      exists_fn: Callable[[str], bool] = os.path.isfile) -> List[CheckResult]:
    checks = []
    if backend == "opencv":
        checks.append(check_model_file("face detector model", cfg.matcher.yunet_model, exists_fn))
        checks.append(check_model_file("face recognition model", cfg.matcher.sface_model, exists_fn))
    if reid_enabled:
        if reid_detector == "yolo":
            checks.append(check_model_file("person detector model", cfg.reid.yolo_model, exists_fn))
        elif reid_detector == "own":
            checks.append(check_model_file("own person detector model", cfg.reid.own_model, exists_fn))
        if reid_embedder in ("onnx", "fused"):
            checks.append(check_model_file("body re-ID model", cfg.reid.reid_model, exists_fn))
    return checks


def run_all(args, cfg: Optional[AppConfig] = None) -> List[CheckResult]:
    cfg = cfg or AppConfig()
    results = [check_python_version(), check_opencv(), check_disk_space()]
    results += model_file_checks(cfg, args.backend, args.reid != "off", args.person_detector, args.reid)

    if not args.skip_camera and not args.skip_hardware:
        results.append(check_camera(args.camera_index, args.camera))

    if args.ranger and not args.skip_ranger and not args.skip_hardware:
        results.append(check_ranger(parse_ranger(args.ranger)))

    if args.driver == "mavlink" and not args.skip_mavlink and not args.skip_hardware:
        results.append(check_mavlink(args.mavlink, args.baud,
                                     autopilot=getattr(args, "autopilot", "auto")))

    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["insightface", "opencv"], default="insightface")
    p.add_argument("--reid", choices=["off", "color", "onnx", "fused"], default="color")
    p.add_argument("--person-detector", choices=["hog", "yolo", "own"], default="yolo")
    p.add_argument("--camera", choices=["webcam", "pi"], default="webcam")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--ranger", help='TF-Luna ports as "bearing_deg:port,...", same syntax as main.py')
    p.add_argument("--driver", choices=["print", "mavlink"], default="print")
    p.add_argument("--mavlink", default="/dev/serial0")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--autopilot", choices=["auto", "ardupilot", "px4"], default="auto",
                   help="firmware the flight controller must run (default auto: either, reported)")
    p.add_argument("--skip-camera", action="store_true", help="skip opening the camera")
    p.add_argument("--skip-ranger", action="store_true", help="skip opening range sensor ports")
    p.add_argument("--skip-mavlink", action="store_true", help="skip the MAVLink heartbeat check")
    p.add_argument("--skip-hardware", action="store_true",
                   help="skip camera + ranger + mavlink (file/import checks only -- CI, SSH, no hardware attached)")
    return p.parse_args(argv)


def main(argv=None, cfg: Optional[AppConfig] = None) -> int:
    args = parse_args(argv)
    results = run_all(args, cfg)

    width = max((len(r.name) for r in results), default=0)
    hard_failures = 0
    for r in results:
        print(f"[{r.status:4}] {r.name.ljust(width)}  {r.detail}")
        if not r.ok and not r.warning:
            hard_failures += 1

    print()
    if hard_failures:
        print(f"{hard_failures} check(s) failed. Fix these before flying.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
