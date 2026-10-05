"""
BaseDriver for a PX4 flight controller over MAVLink (OFFBOARD mode).

Same contract as MavlinkDriver (ArduPilot), and it reuses that driver's gimbal
handling. Pick one with `main.py --autopilot auto|ardupilot|px4` (see
control/flight_controller.py). What PX4 does differently, and what this does:

  * ArduPilot GUIDED -> PX4 OFFBOARD. PX4 only ENTERS offboard while setpoints
    are already arriving, and LEAVES it (failsafe COM_OBL_RC_ACT) if they stop
    for COM_OF_LOSS_T (default 1 s). The brain may run slower than that, so a
    background thread resends the latest setpoint at stream_hz.
  * That thread never keeps a stale command alive: if send() has not been
    called for command_timeout_s, it streams zero velocity (hold) instead.
    If the whole program dies, the stream stops and PX4's own offboard-loss
    failsafe takes over (the equivalent of ArduPilot's GUID_TIMEOUT).
  * Flight modes are two numbers (main mode, sub mode), set with
    MAV_CMD_DO_SET_MODE. Heartbeat custom_mode = main << 16 | sub << 24.
  * Takeoff is PX4's own AUTO.TAKEOFF. Its altitude (param7) is above MEAN SEA
    LEVEL, not above home (ArduPilot: above home), so the current AMSL
    altitude (on the ground, from GLOBAL_POSITION_INT) is added.
  * Velocity frame: MAV_FRAME_BODY_NED. PX4 rotates it by heading only, which
    is exactly our forward/right/up. PX4 REJECTS MAV_FRAME_BODY_OFFSET_NED,
    the frame the ArduPilot driver uses.
  * pose() is relative to where it armed (= PX4's home). PX4's local origin
    is wherever its estimator started, which can be metres away, and the
    geofence / RETURN logic assumes home = (0, 0, 0). PX4's own HOME_POSITION
    is NOT used: PX4 shifts home's altitude to correct baro drift, which made
    the height on the ground read +1.0 m in SITL.
  * We send our own HEARTBEAT at 1 Hz as an onboard controller.

Threading: only the stream thread and the main loop touch the link. Every
send goes through self._lock; only the main loop reads (_pump). The brain
still calls the driver from the main loop only.

Checked against PX4 v1.17.0 source (mavlink_receiver.cpp, navigator takeoff,
commander params) and flown against PX4 v1.17.0 SITL (SIH quad) in the cloud
sandbox with tools/px4_hover_test.py.
"""

import logging
import math
import threading
from typing import Any, Dict, Optional, Tuple

from control.mavlink_driver import MavlinkDriver
from datatypes import DriveCommand, Pose

log = logging.getLogger(__name__)

# ---- PX4 flight modes ------------------------------------------------------
PX4_MAIN_MODES = {1: "MANUAL", 2: "ALTCTL", 3: "POSCTL", 4: "AUTO", 5: "ACRO",
                  6: "OFFBOARD", 7: "STABILIZED", 8: "RATTITUDE"}
PX4_AUTO_SUB_MODES = {1: "READY", 2: "TAKEOFF", 3: "LOITER", 4: "MISSION", 5: "RTL",
                      6: "LAND", 7: "RTGS", 8: "FOLLOW_TARGET", 9: "PRECLAND"}

MODE_OFFBOARD = (6, 0)
MODE_HOLD = (4, 3)       # AUTO.LOITER, shown as "Hold" in QGroundControl
MODE_TAKEOFF = (4, 2)
MODE_LAND = (4, 6)
MODE_RTL = (4, 5)


def px4_custom_mode(main: int, sub: int = 0) -> int:
    """The HEARTBEAT custom_mode number PX4 reports for (main, sub)."""
    return (main << 16) | (sub << 24)


def px4_mode_parts(custom_mode: int) -> Tuple[int, int]:
    return (custom_mode >> 16) & 0xFF, (custom_mode >> 24) & 0xFF


def px4_mode_name(custom_mode: Optional[int]) -> str:
    """e.g. 393216 -> 'OFFBOARD', 50593792 -> 'AUTO.LOITER'. For logs and tools."""
    if custom_mode is None:
        return "?"
    main, sub = px4_mode_parts(custom_mode)
    name = PX4_MAIN_MODES.get(main, f"main{main}")
    if main == 4:
        name += "." + PX4_AUTO_SUB_MODES.get(sub, f"sub{sub}")
    return name


# ---- SET_POSITION_TARGET_LOCAL_NED type_mask (a set bit = IGNORE that field) --
# Velocity (bits 3-5) + yaw rate (bit 11). Bit 9 (FORCE_SET) must stay CLEAR:
# it is "use force", not "ignore force", and PX4 checks it.
PX4_VELOCITY_AND_YAW_RATE = 0b111 | (0b111 << 6) | (1 << 10)   # 0x05C7

# MAVLink numbers that are the same in every dialect (so tests need no pymavlink)
_REFUSED = (1, 2, 3, 4)        # MAV_RESULT TEMPORARILY_REJECTED, DENIED, UNSUPPORTED, FAILED
_LANDED_ON_GROUND = 1          # MAV_LANDED_STATE_*
_LANDED_FLYING = (2, 3, 4)     # IN_AIR, TAKEOFF, LANDING


def _is_zero(cmd: DriveCommand) -> bool:
    return not (cmd.forward_mps or cmd.right_mps or cmd.up_mps or cmd.yaw_rate_dps)


class Px4Driver(MavlinkDriver):
    def __init__(self, conn: Any, mav: Any, *,
                 stream_hz: float = 20.0,
                 command_timeout_s: float = 1.0,
                 offboard_prime_s: float = 1.0,
                 frame: str = "body",
                 start_thread: bool = True,
                 **kwargs):
        """kwargs: everything MavlinkDriver takes (takeoff_alt_m, gimbal, clock, sleep, ...).

        stream_hz          setpoint resend rate (PX4 needs > 2 Hz; 20 is comfortable)
        command_timeout_s  resend the last command this long, then zero velocity (hold)
        offboard_prime_s   stream setpoints this long before asking for OFFBOARD
        frame              "body" (MAV_FRAME_BODY_NED; checked on PX4 v1.17) or "local"
                           (rotated here by our last known yaw; for older firmware
                           that rejects BODY_NED: "coordinate frame 8 unsupported")
        start_thread       False in tests: they call _stream_tick() by hand"""
        super().__init__(conn, mav, **kwargs)
        if frame not in ("body", "local"):
            raise ValueError("frame must be 'body' or 'local'")
        self.stream_hz = stream_hz
        self.command_timeout_s = command_timeout_s
        self.offboard_prime_s = offboard_prime_s
        self.frame = frame
        self.start_thread = start_thread
        self._lock = threading.RLock()
        self._cmd = DriveCommand()
        self._cmd_at = -1e9
        self._streaming = False          # True only between takeoff() and land()/RTL/disconnect
        self._stale_logged = False
        self._hb_sent_at = -1e9
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._autopilot: Optional[int] = None
        self._home_ned: Optional[Tuple[float, float, float]] = None   # local NED where it armed
        self._alt_amsl: Optional[float] = None
        self._landed_state: Optional[int] = None
        self._acks: Dict[int, int] = {}
        self._last_text = ""

    @classmethod
    def open(cls, url: str = "/dev/serial0", baud: int = 921600, **kwargs) -> "Px4Driver":
        """e.g. open("/dev/serial0") on the drone, open("udpin:127.0.0.1:14540") for PX4 SITL."""
        from pymavlink import mavutil
        return cls(mavutil.mavlink_connection(url, baud=baud), mavutil.mavlink, **kwargs)

    # ---- incoming messages ------------------------------------------------
    def _pump(self) -> None:
        m = self.mav
        while True:
            msg = self.conn.recv_match(blocking=False)
            if msg is None:
                return
            kind = msg.get_type()
            if kind == "HEARTBEAT":
                if msg.type == m.MAV_TYPE_GCS or msg.autopilot == m.MAV_AUTOPILOT_INVALID:
                    continue  # a ground station, camera, companion...: not the flight controller
                self._autopilot = msg.autopilot
                self._mode = msg.custom_mode
                self._armed = bool(msg.base_mode & m.MAV_MODE_FLAG_SAFETY_ARMED)
                self._last_heartbeat = self._clock()
            elif kind == "LOCAL_POSITION_NED":
                self._north, self._east, self._down = msg.x, msg.y, msg.z
            elif kind == "ATTITUDE":
                self._yaw = msg.yaw
            elif kind == "SYS_STATUS":
                self._battery = None if msg.battery_remaining < 0 else float(msg.battery_remaining)
            elif kind == "GLOBAL_POSITION_INT":
                self._alt_amsl = msg.alt / 1000.0           # mm above mean sea level
            elif kind == "EXTENDED_SYS_STATE":
                self._landed_state = msg.landed_state
            elif kind == "COMMAND_ACK":
                self._acks[msg.command] = msg.result
                if msg.command == m.MAV_CMD_DO_MOUNT_CONTROL:
                    self._on_mount_ack(msg.result)
            elif kind == "STATUSTEXT":
                text = msg.text.decode(errors="replace") if isinstance(msg.text, bytes) else str(msg.text)
                text = text.rstrip("\x00").strip()
                if msg.severity <= 4:                     # EMERGENCY..WARNING
                    self._last_text = text
                    log.warning("PX4: %s", text)

    # ---- outgoing ---------------------------------------------------------
    def _command(self, command: int, *params: float) -> None:
        """COMMAND_LONG with up to 7 params. Forgets any earlier ACK for it."""
        p = list(params) + [0.0] * (7 - len(params))
        with self._lock:
            self._acks.pop(command, None)
            c = self.conn
            c.mav.command_long_send(c.target_system, c.target_component, command, 0, *p)

    def _send_mode(self, mode: Tuple[int, int]) -> None:
        m = self.mav
        base = m.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | (m.MAV_MODE_FLAG_SAFETY_ARMED if self._armed else 0)
        self._command(m.MAV_CMD_DO_SET_MODE, base, mode[0], mode[1])

    def _in_mode(self, mode: Tuple[int, int]) -> bool:
        if self._mode is None:
            return False
        main, sub = px4_mode_parts(self._mode)
        return main == mode[0] and (mode[0] != 4 or sub == mode[1])

    def _wait_cmd(self, condition, command: int, timeout_s: float, what: str) -> None:
        """Like _wait, but fails at once if PX4 answers the command with a refusal."""
        deadline = self._clock() + timeout_s
        while True:
            self._pump()
            if condition():
                return
            result = self._acks.get(command)
            if result in _REFUSED:
                raise RuntimeError(f"PX4 refused {what} (MAV_RESULT {result}){self._why()}")
            if self._clock() > deadline:
                raise RuntimeError(f"timed out waiting for {what}{self._why()}")
            self._sleep(0.05)

    def _why(self) -> str:
        return f". PX4 said: {self._last_text!r}" if self._last_text else ""

    def _set_mode(self, mode: Tuple[int, int], timeout_s: float = 5.0) -> None:
        self._send_mode(mode)
        name = px4_mode_name(px4_custom_mode(*mode))
        self._wait_cmd(lambda: self._in_mode(mode), self.mav.MAV_CMD_DO_SET_MODE, timeout_s, f"{name} mode")

    def _setpoint(self, cmd: DriveCommand) -> None:
        """One SET_POSITION_TARGET_LOCAL_NED: velocity + yaw rate. Caller holds the lock."""
        c, m = self.conn, self.mav
        down = -cmd.up_mps                                   # NED: down = -up
        if self.frame == "body":
            frame, vx, vy = m.MAV_FRAME_BODY_NED, cmd.forward_mps, cmd.right_mps
        else:
            cy, sy = math.cos(self._yaw), math.sin(self._yaw)
            frame = m.MAV_FRAME_LOCAL_NED
            vx = cy * cmd.forward_mps - sy * cmd.right_mps   # north
            vy = sy * cmd.forward_mps + cy * cmd.right_mps   # east
        c.mav.set_position_target_local_ned_send(
            0, c.target_system, c.target_component, frame, PX4_VELOCITY_AND_YAW_RATE,
            0, 0, 0, vx, vy, down, 0, 0, 0,
            0.0, math.radians(cmd.yaw_rate_dps))              # yaw rate rad/s, + = clockwise

    def _stream_tick(self) -> None:
        """Called stream_hz times a second by the thread: our heartbeat (1 Hz)
        and, while in the air under our control, the latest setpoint."""
        now = self._clock()
        with self._lock:
            if now - self._hb_sent_at >= 1.0:
                m, c = self.mav, self.conn
                c.mav.heartbeat_send(m.MAV_TYPE_ONBOARD_CONTROLLER, m.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                self._hb_sent_at = now
            if not self._streaming:
                return
            fresh = now - self._cmd_at <= self.command_timeout_s
            if not fresh and not _is_zero(self._cmd):
                if not self._stale_logged:
                    log.warning("no new command for %.1f s: holding (zero velocity)", self.command_timeout_s)
                    self._stale_logged = True
                self._cmd = DriveCommand()
            self._setpoint(self._cmd)

    def _run_stream(self) -> None:
        period = 1.0 / self.stream_hz
        while not self._stop_evt.wait(period):
            try:
                self._stream_tick()
            except Exception as e:  # never let the thread die silently mid-flight
                log.warning("setpoint stream: %s", e)

    def _start_thread(self) -> None:
        if self.start_thread and self._thread is None:
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._run_stream, name="px4-setpoints", daemon=True)
            self._thread.start()

    def _stop_streaming(self) -> None:
        with self._lock:
            self._streaming = False
            self._cmd = DriveCommand()

    # ---- BaseDriver: lifecycle --------------------------------------------
    def connect(self) -> None:
        super().connect()          # heartbeat; LOCAL_POSITION_NED, ATTITUDE, SYS_STATUS at 10 Hz
        m = self.mav
        if self._autopilot is not None and self._autopilot != m.MAV_AUTOPILOT_PX4:
            raise RuntimeError(f"this flight controller is not PX4 (MAV_AUTOPILOT {self._autopilot}): "
                               "use --autopilot ardupilot (or auto)")
        for msg_id, interval_us in ((m.MAVLINK_MSG_ID_ATTITUDE, 50_000),            # 20 Hz: fresher yaw
                                    (m.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 200_000),
                                    (m.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 200_000)):
            self._command(m.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, interval_us)
        self._start_thread()

    def arm(self) -> None:
        # Hold (AUTO.LOITER) needs a position estimate, like GUIDED on ArduPilot.
        self._set_mode(MODE_HOLD)
        m = self.mav
        self._last_text = ""
        self._command(m.MAV_CMD_COMPONENT_ARM_DISARM, 1)     # plain arm: PX4's preflight checks run
        try:
            self._wait_cmd(lambda: self._armed, m.MAV_CMD_COMPONENT_ARM_DISARM, 10, "arming")
        except RuntimeError as e:
            raise RuntimeError(f"the flight controller refused to arm ({e}). Read its preflight "
                               "messages in QGroundControl (GPS lock? compass? calibration?)") from None
        self._home_ned = (self._north, self._east, self._down)   # home = where it armed

    def takeoff(self) -> None:
        m = self.mav
        self._pump()                                         # latest GLOBAL_POSITION_INT
        if self._alt_amsl is None:
            log.warning("no GLOBAL_POSITION_INT from PX4: it will climb to its own MIS_TAKEOFF_ALT")
            alt_amsl = float("nan")
        else:
            alt_amsl = self._alt_amsl + self.takeoff_alt_m   # still on the ground: ground AMSL + height
        nan = float("nan")
        # pitch, -, -, yaw (NaN = keep), lat, lon (NaN = here), altitude AMSL
        self._command(m.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, nan, nan, nan, alt_amsl)
        self._wait_cmd(lambda: self._height() >= 0.9 * self.takeoff_alt_m,
                       m.MAV_CMD_NAV_TAKEOFF, 30, "takeoff altitude")
        self._enter_offboard()

    def _enter_offboard(self) -> None:
        with self._lock:
            self._cmd, self._cmd_at = DriveCommand(), self._clock()
            self._stale_logged = False
            self._streaming = True
        # PX4 accepts OFFBOARD only while setpoints are already arriving.
        period = 1.0 / self.stream_hz
        end = self._clock() + self.offboard_prime_s
        while self._clock() < end:
            with self._lock:
                self._cmd_at = self._clock()
                self._setpoint(self._cmd)
            self._pump()
            self._sleep(period)
        self._set_mode(MODE_OFFBOARD)

    def send(self, cmd: DriveCommand) -> None:
        permitted = self.autonomy_permitted()
        with self._lock:
            self._cmd_at = self._clock()
            if not permitted:
                self._cmd = DriveCommand()   # a human has the controls: never replay this later
                return
            self._cmd = cmd
            self._stale_logged = False
            self._setpoint(cmd)

    def stop(self) -> None:
        with self._lock:
            self._cmd, self._cmd_at = DriveCommand(), self._clock()
        if self.autonomy_permitted():
            self.send(DriveCommand())

    def set_camera_pitch(self, pitch_down_deg: float) -> None:
        with self._lock:
            super().set_camera_pitch(pitch_down_deg)

    def land(self) -> None:
        self._stop_streaming()
        self._switch_best_effort(MODE_LAND)

    def return_to_launch(self) -> None:
        self._stop_streaming()
        self._switch_best_effort(MODE_RTL)

    def _switch_best_effort(self, mode: Tuple[int, int], tries: int = 3, wait_s: float = 1.0) -> None:
        """For the shutdown path: never raises. Resends the mode until PX4 reports
        it (or is disarmed), because one lost packet here must not leave the
        drone hovering in OFFBOARD with no setpoints."""
        for _ in range(tries):
            self._send_mode(mode)
            deadline = self._clock() + wait_s
            while True:
                self._pump()
                if self._in_mode(mode) or not self._armed:
                    return
                if self._clock() > deadline:
                    break
                self._sleep(0.05)
        log.warning("PX4 did not confirm %s mode after %d tries%s",
                    px4_mode_name(px4_custom_mode(*mode)), tries, self._why())

    def disconnect(self) -> None:
        self._stop_streaming()
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.conn.close()

    # ---- BaseDriver: state ------------------------------------------------
    def _height(self) -> float:
        home_down = self._home_ned[2] if self._home_ned else 0.0
        return -(self._down - home_down)

    def pose(self) -> Pose:
        self._pump()
        hn, he, _ = self._home_ned or (0.0, 0.0, 0.0)
        return Pose(self._east - he, self._north - hn, self._yaw, self._height())

    def airborne(self) -> bool:
        self._pump()
        if not self._armed:
            return False
        if self._landed_state in _LANDED_FLYING:
            return True
        if self._landed_state == _LANDED_ON_GROUND:
            return False
        return self._height() > 0.3

    def autonomy_permitted(self) -> bool:
        self._pump()
        fresh = self._clock() - self._last_heartbeat < self.heartbeat_timeout_s
        # armed too: PX4 drops back to OFFBOARD by itself after an auto-disarm on the ground
        return fresh and self._armed and self._in_mode(MODE_OFFBOARD)

    # ---- extras for tools -------------------------------------------------
    def armed(self) -> bool:
        self._pump()
        return self._armed

    def mode_name(self) -> str:
        self._pump()
        return px4_mode_name(self._mode)
