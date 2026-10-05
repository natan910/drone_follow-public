"""
Entry point. Wires a platform (where observations come from and decisions go)
to the autopilot (the brain), and runs the loop.

    # 1. The toy world: no hardware at all. Watch it explore, find a person, follow.
    python main.py --platform sim --show --seconds 300

    # 2. Laptop webcam dry run: commands are printed, nothing can fly.
    python main.py --platform real --phone --auto-launch
    python main.py --platform real --reference target.jpg --backend opencv --auto-launch
    python main.py --platform real --phone --reid fused --person-detector yolo \\
        --yolo-model models/yolo11n_320.onnx --reid-model models/person_reid_youtu_2021nov.onnx

    # 3. On the drone (Raspberry Pi camera + MAVLink + range sensors). It waits on the
    #    ground, disarmed, until you tap Launch on the phone page or fleet console:
    python main.py --platform real --camera pi --driver mavlink --backend opencv \\
        --ranger "0:/dev/ttyUSB0,-45:/dev/ttyUSB1,45:/dev/ttyUSB2" --phone \\
        --map area.npz --log flight.jsonl

    # 4. Same, also reporting into a fleet/server.py dashboard:
    python main.py --platform real --driver mavlink --phone \\
        --fleet-url http://<fleet-host>:8090 --drone-id drone-1 --fleet-token <token>

    # 5. Our own flight controller (flightcore/) flying its own simulator: no ArduPilot, no SITL.
    #    Webcam for the camera, simulated physics for the body. Nothing real can fly.
    python main.py --platform real --driver native-sim --backend opencv --fixed-camera --fixed-pitch 0 --phone --auto-launch

    # 6. Our own person detector (TRAINING.md), and recording raw frames for training:
    python main.py --platform real --phone --person-detector own --own-model models/person_own.onnx
    python main.py --platform real --phone --record datasets --record-subject subject-01

    # 7. PX4 instead of ArduPilot (PX4.md). --autopilot auto (default) reads it from the heartbeat.
    python -u main.py --platform real --driver mavlink --autopilot px4 --mavlink udpin:127.0.0.1:14540 \\
        --backend opencv --camera webcam --fixed-camera --fixed-pitch 0 --phone

Launch gate (autonomy/launch.py): with --platform real nothing arms until a Launch
command (phone page / fleet console button) or --auto-launch, AND preflight passes
(heartbeat; battery known with --driver mavlink and above the return level).

Press q in a preview window, or Ctrl-C, to stop.
"""

import argparse
import math
import os
import secrets
import socket
import time
from typing import Dict, List, Optional

import cv2

from autonomy.autopilot import Autopilot
from autonomy.launch import LaunchGate, link_ok
from autonomy.runner import run
from comms.phone_server import CommandBox, EnrollmentBox, PhoneServer
from config import AppConfig
from control.flight_controller import open_flight_controller
from control.native_driver import NativeDriver
from control.print_driver import PrintDriver
from datatypes import Decision, Observation
from missions.wiring import add_mission_args, make_mission
from fleet.reporter import FleetReporter
from mapping.occupancy_grid import OccupancyGrid
from perception.calibration import FovCalibrator, load_calibration
from perception.target_finder import TargetFinder, make_finder
from perception.range_sensors import TFLunaRing
from perception.stream_input import PiCameraSource, WebcamSource
from platforms.real import RealPlatform
from platforms.sim import SimPlatform
from sim.scenarios import demo_world
from sim.viewer import render
from telemetry.flight_log import FlightLog


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Autonomous person-following drone")
    p.add_argument("--platform", choices=["sim", "real"], default="real")
    p.add_argument("--seconds", type=float, help="stop after this long in the air (default: run until stopped)")
    p.add_argument("--show", action="store_true", help="sim: show the top-down viewer")
    p.add_argument("--no-window", action="store_true", help="real: no camera preview")
    p.add_argument("--backend", choices=["insightface", "opencv"], default="insightface")
    p.add_argument("--reference", help="photo of the target (or send one from the phone); "
                   "a full-length photo also teaches body re-ID what they look like")
    p.add_argument("--reid", choices=["off", "color", "onnx", "fused"], default="color",
                   help="body re-ID when the face isn't visible (default: color, no download)")
    p.add_argument("--person-detector", choices=["hog", "yolo", "own"], default="yolo",
                   help="'yolo' (default) needs a downloaded ONNX model -- see README; "
                   "'own' is our own detector trained with training/train.py (TRAINING.md); "
                   "'hog' needs nothing but is much weaker on close/partial/turned-around views")
    p.add_argument("--yolo-model", help="person detector ONNX (with --person-detector yolo)")
    p.add_argument("--yolo-format", choices=["auto", "v8", "v5", "yolox"], default="yolox")
    p.add_argument("--yolo-size", type=int,
                   help="input size the YOLO export was made at (default: 640 for yolox, else config.py)")
    p.add_argument("--own-model", help="our own detector's ONNX (with --person-detector own; "
                   "default: config.py own_model)")
    p.add_argument("--own-size", type=int, help="input size our detector was exported at (default: config.py)")
    p.add_argument("--reid-model", help="re-ID network ONNX (with --reid onnx/fused)")
    p.add_argument("--camera", choices=["webcam", "pi"], default="webcam")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--fixed-camera", action="store_true",
                   help="the camera cannot physically tilt (e.g. a laptop webcam): stop the "
                   "autopilot from believing a gimbal is aiming it, which otherwise causes wild "
                   "yaw oscillation on a dry run. Use --fixed-pitch to set its (unchanging) tilt.")
    p.add_argument("--fixed-pitch", type=float, default=0.0,
                   help="tilt (deg, 0 = level, 90 = straight down) of a --fixed-camera; a laptop "
                   "lid webcam pointed at your face is normally 0")
    p.add_argument("--hfov", type=float,
                   help="force the camera's horizontal field of view, in degrees (overrides the "
                   "saved calibration). Normally you don't need this: tap 'Calibrate' once on the "
                   "phone page or fleet console instead")
    p.add_argument("--calibration-file", default="camera_calibration.json",
                   help="where the one-tap camera calibration is saved and loaded from")
    p.add_argument("--no-geofence", action="store_true",
                   help="the --driver print dry run dead-reckons position from commands it never "
                   "actually executes, so a long stationary test can 'drift' past the geofence and "
                   "force RETURN for no real reason; this turns that check off for the dry run")
    p.add_argument("--driver", choices=["print", "mavlink", "native-sim"], default="print",
                   help="print: dry run. mavlink: a flight controller over MAVLink, ArduPilot or PX4 "
                   "(SITL or real; see --autopilot). native-sim: our own flight controller (flightcore/) "
                   "flying its own simulator, no ArduPilot/PX4 needed")
    p.add_argument("--mavlink", default="/dev/serial0",
                   help="serial device or udpin:host:port (ArduPilot SITL: udpin:127.0.0.1:14551, "
                   "PX4 SITL: udpin:127.0.0.1:14540)")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--autopilot", choices=["auto", "ardupilot", "px4"], default="auto",
                   help="firmware behind --driver mavlink (default: read it from the heartbeat)")
    p.add_argument("--auto-launch", action="store_true",
                   help="arm and take off as soon as preflight passes, without the phone/console "
                   "Launch button. For SITL and dry runs only: never on a real drone")
    p.add_argument("--max-altitude", type=float, metavar="M",
                   help="altitude ceiling, metres above the launch point (default: config.py, 15). "
                   "Never climbs above it; RETURN if 3 m above it anyway")
    p.add_argument("--ranger", help='TF-Luna ports as "bearing_deg:port,...", e.g. "0:/dev/ttyUSB0,-45:/dev/ttyUSB1"')
    p.add_argument("--phone", action="store_true", help="start the photo-upload server")
    p.add_argument("--phone-port", type=int, default=8080)
    p.add_argument("--token", help="phone server secret (default: random)")
    p.add_argument("--map", help="load this map at start if it exists; save it on exit")
    p.add_argument("--log", help="write a JSON-lines flight log here")
    p.add_argument("--record", metavar="DIR",
                   help="save raw camera frames + pose (4 fps, before the overlay) under DIR for "
                   "training our own models, e.g. --record datasets (TRAINING.md). Never commit DIR")
    p.add_argument("--record-subject", metavar="NAME",
                   help="who is in the footage (goes into the session id and session.json)")
    p.add_argument("--record-source", metavar="WHERE",
                   help="what filmed it, for session.json (default: 'drone' with --camera pi, else 'webcam')")
    p.add_argument("--record-fps", type=float, default=4.0, help="frames saved per second (default 4)")
    p.add_argument("--record-max-gb", type=float, default=20.0,
                   help="stop recording when the session reaches this size (default 20)")
    p.add_argument("--fleet-url", help="report this drone's status to a fleet/server.py dashboard "
                   "at this URL, e.g. http://<fleet-host>:8090 (optional; never affects flight "
                   "if the fleet server is slow or unreachable -- see fleet/reporter.py)")
    p.add_argument("--drone-id", help="this drone's name on the fleet dashboard (required with --fleet-url)")
    p.add_argument("--fleet-token", help="fleet server token (default: reuse --token)")
    p.add_argument("--fleet-interval", type=float, default=1.0, help="seconds between fleet status reports")
    add_mission_args(p)
    return p.parse_args(argv)


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def parse_ranger(spec: str) -> Dict[float, str]:
    ports = {}
    for item in spec.split(","):
        bearing, port = item.split(":", 1)
        ports[float(bearing)] = port
    return ports


def draw_lines(frame, lines: List[str], colour=(0, 255, 0)) -> None:
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (15, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)


def draw_overlay(frame, decision: Decision, obs: Observation, fps: float, ms: float) -> None:
    if obs.detection is not None:
        top, right, bottom, left = obs.detection.bbox
        colour = (0, 255, 0) if obs.detection.source == "face" else (0, 200, 255)  # green face, amber body
        cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
        cv2.putText(frame, obs.detection.source, (left, max(15, top - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    draw_lines(frame, [f"{decision.mode.name}  {decision.note}",
                       f"cmd: yaw {decision.cmd.yaw_rate_dps:+.1f} deg/s  fwd {decision.cmd.forward_mps:+.2f} m/s",
                       f"{fps:.1f} FPS   matcher {ms:.0f} ms"])


def draw_preflight(frame, launch: dict) -> None:
    lines = ["PREFLIGHT  " + ("launching" if launch["state"] == "launching" else
                              "waiting for launch" + (" (auto)" if launch["auto"] else ": tap Launch"))]
    lines += launch["problems"] or ["preflight OK"]
    if launch.get("last_error"):
        lines.append(launch["last_error"])
    draw_lines(frame, lines, (0, 200, 255) if launch["problems"] else (0, 255, 0))


def configure_reid(args, cfg: AppConfig) -> None:
    """Flags -> cfg.reid. Optional flags are read with getattr, so a caller (a test,
    a tool) can pass a Namespace holding only the flags it cares about."""
    r = cfg.reid
    r.enabled = args.reid != "off"
    if r.enabled:
        r.embedder, r.detector, r.yolo_format = args.reid, args.person_detector, args.yolo_format
    if args.yolo_format == "yolox":      # OpenCV Zoo's YOLOX: BGR, 0..255, 640 x 640
        r.yolo_rgb, r.yolo_scale_01, r.yolo_input_size = False, False, 640
    if getattr(args, "yolo_size", None):
        r.yolo_input_size = args.yolo_size
    if getattr(args, "yolo_model", None):
        r.yolo_model = args.yolo_model
    if getattr(args, "own_model", None):
        r.own_model = args.own_model
    if getattr(args, "own_size", None):
        r.own_input_size = args.own_size
    if getattr(args, "reid_model", None):
        r.reid_model = args.reid_model


def make_autopilot(args, cfg: AppConfig) -> Autopilot:
    grid = OccupancyGrid(cfg.map)
    if args.map and os.path.exists(args.map):
        grid.load(args.map)
        print(f"Loaded map from {args.map}")
    return Autopilot(cfg, grid)


def make_recorder(args):
    """--record: a dataset/recorder.py Recorder (background writer, drops frames
    rather than slowing the loop), or None. Optional flags via getattr (see configure_reid)."""
    if not args.record:
        return None
    from dataset.layout import SessionInfo, new_session_id
    from dataset.curator import build_recorder
    subject = getattr(args, "record_subject", None) or "unknown"
    camera = getattr(args, "camera", None) or ""
    source = getattr(args, "record_source", None) or ("drone" if camera == "pi" else "webcam")
    info = SessionInfo(new_session_id(subject), subject=subject, source=source, camera=camera)
    try:
        recorder = build_recorder(args, info)
    except OSError as e:
        raise SystemExit(f"ERROR: --record {args.record}: {e}")
    print(f"Recording raw frames to {recorder.dir} (subject {subject!r}). Never commit this folder.")
    return recorder


def safe_record(recorder, frame, obs: Observation, decision: Optional[Decision]) -> bool:
    """Recording must never take the flight down: any error stops recording
    (returns False), the loop goes on."""
    try:
        recorder.maybe_record(frame, obs, decision)
        return True
    except Exception as e:
        print(f"Recorder: stopped after an error ({type(e).__name__}: {e}). Flight is unaffected.")
        return False


def make_driver(args, cfg: AppConfig):
    """The vehicle. For --driver mavlink this waits (up to 10 s) for the flight controller's
    heartbeat and picks the ArduPilot or PX4 driver; it never arms or moves anything."""
    if args.driver == "mavlink":
        try:
            return open_flight_controller(args.mavlink, args.baud, args.autopilot, gimbal=cfg.camera.gimbal)
        except ImportError:
            raise SystemExit("ERROR: pymavlink is missing: pip install pymavlink (or ./setup.sh --full)")
        except (RuntimeError, OSError) as e:   # no heartbeat, wrong firmware, port in use
            raise SystemExit(f"ERROR: flight controller: {e}")
    if args.driver == "native-sim":
        return NativeDriver.simulated()
    return PrintDriver()


def run_sim(args, cfg: AppConfig) -> None:
    world = demo_world()
    platform = SimPlatform(world)
    autopilot = make_autopilot(args, cfg)
    frames = [0]

    def hook(obs, decision):
        frames[0] += 1
        if args.show and frames[0] % 3 == 0:
            cv2.imshow("toy world", render(world, autopilot))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

    run(platform, autopilot, seconds=args.seconds, on_step=hook)
    print(f"Done at t={world.t:.0f}s, collisions={world.collisions}, mode={autopilot.mode.name}")
    if args.map:
        autopilot.grid.save(args.map)
    cv2.destroyAllWindows()


def run_real(args, cfg: AppConfig) -> None:
    cfg.matcher.backend, cfg.matcher.reference_image = args.backend, args.reference
    cfg.show_window = not args.no_window
    if args.fixed_camera:
        cfg.camera.gimbal = False
        cfg.camera.fixed_pitch_deg = args.fixed_pitch
    calibrated = False
    if args.hfov is not None:
        cfg.camera.hfov_deg = args.hfov
    else:
        try:
            saved = load_calibration(args.calibration_file)
        except ValueError as e:
            print(f"WARNING: ignoring camera calibration: {e}")
            saved = None
        if saved:
            cfg.camera.hfov_deg, cfg.camera.aspect = saved["hfov_deg"], saved["aspect"]
            calibrated = True
            print(f"Camera calibration: hfov {cfg.camera.hfov_deg:.1f} deg, aspect {cfg.camera.aspect:.3f} "
                  f"(measured {saved.get('measured_at', '?')})")
        else:
            print(f"Camera not calibrated yet: using the default hfov {cfg.camera.hfov_deg:.0f} deg. "
                  "Tap 'Calibrate' on the phone page or fleet console once, on the ground.")
    if args.no_geofence:
        cfg.safety.geofence_radius_m = float("inf")
    if args.max_altitude is not None:
        if args.max_altitude < 2.0:
            raise SystemExit("--max-altitude must be at least 2 m (takeoff climbs to 1.5 m)")
        cfg.safety.max_altitude_m = args.max_altitude
    # a flight controller (real or SITL) must report its battery; print / native-sim report None
    cfg.safety.require_battery = args.driver == "mavlink"
    fleet_token = None
    if args.fleet_url:
        if not args.drone_id:
            raise SystemExit("--fleet-url needs --drone-id (a name for this drone on the fleet console)")
        fleet_token = args.fleet_token or args.token
        if not fleet_token:
            raise SystemExit("--fleet-url needs --fleet-token (or --token) -- the fleet server's shared secret")
    if args.auto_launch and args.driver == "mavlink" and not args.mavlink.startswith(("udp", "tcp")):
        print("WARNING: --auto-launch on a serial flight controller: it will arm and take off by itself. "
              "Props off unless you mean it.")
    if not (args.phone or args.fleet_url or args.auto_launch):
        raise SystemExit("Nothing can launch this drone: pass --phone and/or --fleet-url (for the "
                         "Launch button), or --auto-launch (SITL / dry runs only)")
    configure_reid(args, cfg)
    driver = make_driver(args, cfg)   # first: a wrong port fails in seconds, before models load or the camera opens
    recorder = make_recorder(args)    # likewise: a bad --record folder fails before anything starts
    print(f"Loading matcher ({cfg.matcher.backend}"
          + (f", body re-ID: {cfg.reid.embedder} + {cfg.reid.detector})..." if cfg.reid.enabled else ")..."))
    mission = make_mission(args, cfg)
    matcher = mission.finder or make_finder(cfg.matcher, cfg.reid)
    mission.attach(matcher, recorder)

    camera = (PiCameraSource() if args.camera == "pi" else WebcamSource(args.camera_index))
    ranger = TFLunaRing(parse_ranger(args.ranger)) if args.ranger else None
    cfg.safety.require_scan = ranger is not None  # no sensors = dry run only
    if ranger is None and args.driver == "mavlink":
        print("WARNING: flying with NO range sensors. Open field, low speed, pilot ready.")

    remote = args.phone or bool(args.fleet_url)   # either way, someone can send photos and tasks
    box = EnrollmentBox() if remote else None
    commands = CommandBox() if remote else None
    platform = RealPlatform(camera, matcher, driver, ranger, box, commands,
                            calibrator=FovCalibrator(cfg.camera.face_height_m),
                            calibration_path=args.calibration_file)
    autopilot = make_autopilot(args, cfg)
    mission.bind(autopilot)
    gate = LaunchGate(cfg.safety, driver, auto=args.auto_launch)

    board: Dict[str, object] = {}
    server = None
    if args.phone:
        token = args.token or cfg.phone.token or secrets.token_urlsafe(8)
        extra = ({"fleet_console": f"{args.fleet_url.rstrip('/')}/?token={fleet_token}"}
                 if args.fleet_url else {})
        server = PhoneServer(box, lambda: {**board, **extra}, token, cfg.phone.host, args.phone_port,
                             int(cfg.phone.max_upload_mb * 1_000_000), commands=commands)
        server.start()
        print(f"\nPhone page:  http://{local_ip()}:{server.port}/?token={token}\n")

    fleet_reporter = None
    if args.fleet_url:
        fleet_reporter = FleetReporter(args.fleet_url, args.drone_id, fleet_token,
                                       interval_s=args.fleet_interval, enrollment=box, commands=commands)
        fleet_reporter.start()
        print(f"Reporting to the fleet console at {args.fleet_url} as {args.drone_id!r}")
    geofence = cfg.safety.geofence_radius_m

    log = FlightLog(args.log) if args.log else None
    recording = recorder is not None
    fps, last = 0.0, time.perf_counter()

    def refresh_board(obs: Observation, launch: dict, mode: Optional[str] = None) -> None:
        if isinstance(matcher, TargetFinder):
            board["reid"] = dict(matcher.stats)
        cal = platform.calibration_status()
        board.update(autopilot.status(), target_enrolled=matcher.has_target,
                     battery=obs.battery_pct, x=round(obs.pose.x, 1), y=round(obs.pose.y, 1),
                     z=round(obs.pose.z, 1), yaw_deg=round(math.degrees(obs.pose.yaw), 1),
                     geofence_m=geofence if math.isfinite(geofence) else None,  # JSON has no Infinity
                     calibration=cal, calibrated=calibrated or cal["state"] == "done",
                     launch=launch, link=link_ok(driver))
        if mode is not None:
            board["mode"] = mode
        if recording:
            board["recording"] = recorder.status()
        if fleet_reporter is not None:
            fleet_reporter.update(dict(board))

    def show(draw) -> Optional[bool]:
        if cfg.show_window and platform.last_frame is not None:
            draw(platform.last_frame)
            cv2.imshow("drone_follow", platform.last_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return None

    def ground_hook(obs: Observation, launch: dict):
        """Every loop before launch: status pages up to date, preview window, q to quit."""
        nonlocal recording
        if recording:   # ground footage is training data too
            recording = safe_record(recorder, platform.last_frame, obs, None)
        refresh_board(obs, launch, mode="PREFLIGHT")
        return show(lambda frame: draw_preflight(frame, launch))

    flying = {"state": "launched", "ready": True, "problems": [], "auto": args.auto_launch, "last_error": None}

    def hook(obs: Observation, decision: Decision):
        nonlocal fps, last, recording
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        if recording:   # the raw frame: before draw_overlay paints on it
            recording = safe_record(recorder, platform.last_frame, obs, decision)
        mission.step(platform.last_frame, obs, decision, board)
        refresh_board(obs, flying)
        if log:
            log.log(obs, decision)
        return show(lambda frame: draw_overlay(frame, decision, obs, fps, platform.last_match_ms))

    try:
        with driver:  # always stop -> land -> disconnect on the way out
            while gate.wait(platform, autopilot, ground_hook):
                try:
                    driver.arm()          # the flight controller's own pre-arm checks can still refuse
                except RuntimeError as e:
                    gate.failed(str(e))
                    continue
                platform.set_home(driver.pose())   # home, geofence and ceiling: from where we launch
                platform.launched = True
                print(f"Armed. Home set here; ceiling {cfg.safety.max_altitude_m:g} m.")
                driver.takeoff()
                run(platform, autopilot, seconds=args.seconds, on_step=hook)
                break
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.stop()
        if fleet_reporter is not None:
            fleet_reporter.stop()
        if log:
            log.close()
        if recorder is not None:
            recorder.close()
            st = recorder.status()
            print(f"Recorded {st['frames']} frames ({st['mb']} MB, {st['dropped']} dropped) to {recorder.dir}")
        platform.close()
        mission.close()
        cv2.destroyAllWindows()
        if args.map:
            autopilot.grid.save(args.map)
            print(f"Saved map to {args.map}")


def main() -> None:
    args = parse_args()
    cfg = AppConfig()
    if args.platform == "sim":
        run_sim(args, cfg)
    else:
        run_real(args, cfg)


if __name__ == "__main__":
    main()
