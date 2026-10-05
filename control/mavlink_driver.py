"""
BaseDriver for a real flight controller over MAVLink (ArduPilot Copter).

Design rules:
  * We only ever command velocity + yaw rate, in the drone's own body frame.
  * We only send while the flight controller is in GUIDED mode. If the pilot
    flips the mode switch to anything else, autonomy_permitted() goes False,
    the autopilot goes silent, and this driver drops any stray command.
  * If our program dies, ArduPilot stops obeying old velocity commands after
    GUID_TIMEOUT and holds position; the pilot switches modes and flies home.
  * Arming is a plain arm request: the flight controller's own pre-arm checks
    run and can refuse it. We never send the "force arm" magic number.
  * Gimbal tilt is best-effort: a tilt command is only sent when the angle
    changes (or now and then, in case a packet was lost), and after a few
    refusals in a row we assume there is no gimbal and stop sending.

The connection is injected, so tests use a fake and never need pymavlink or
hardware. Use MavlinkDriver.open(...) for the real thing.
"""

import logging
import math
import time
from typing import Any, Callable, Optional

from control.base_driver import BaseDriver
from datatypes import DriveCommand, Pose

log = logging.getLogger(__name__)

# MAVLink "type_mask" bits for SET_POSITION_TARGET_LOCAL_NED: a set bit means IGNORE that field.
_IGNORE_POSITION = 0b111          # bits 0-2: x, y, z
_IGNORE_ACCELERATION = 0b111 << 6  # bits 6-8
_IGNORE_FORCE = 1 << 9
_IGNORE_YAW = 1 << 10
# Use velocity (bits 3-5) and yaw rate (bit 11); ignore everything else.
VELOCITY_AND_YAW_RATE = _IGNORE_POSITION | _IGNORE_ACCELERATION | _IGNORE_FORCE | _IGNORE_YAW

COPTER_GUIDED = 4  # ArduCopter custom mode number


class MavlinkDriver(BaseDriver):
    def __init__(self, conn: Any, mav: Any, takeoff_alt_m: float = 1.5,
                 heartbeat_timeout_s: float = 2.0,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 gimbal: bool = True,
                 gimbal_min_step_deg: float = 0.5,
                 gimbal_resend_s: float = 2.0,
                 gimbal_max_refusals: int = 3):
        self.conn, self.mav = conn, mav          # a pymavlink connection, and pymavlink's `mavlink` module
        self.takeoff_alt_m = takeoff_alt_m
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self._clock, self._sleep = clock, sleep
        self._mode: Optional[int] = None
        self._armed = False
        self._last_heartbeat = -1e9
        self._north = self._east = self._down = self._yaw = 0.0
        self._battery: Optional[float] = None
        # gimbal=False (a camera bolted to the frame) makes set_camera_pitch a no-op.
        self.gimbal = gimbal
        self.gimbal_min_step_deg = gimbal_min_step_deg   # ignore requests closer than this to the last one sent
        self.gimbal_resend_s = gimbal_resend_s           # ...but repeat the last angle this often anyway
        self.gimbal_max_refusals = max(1, gimbal_max_refusals)
        self._mount_pitch_sent: Optional[float] = None
        self._mount_sent_at = -1e9
        self._mount_refusals = 0                         # consecutive refusals of DO_MOUNT_CONTROL
        self._mount_gave_up = False

    @classmethod
    def open(cls, url: str = "/dev/serial0", baud: int = 921600, **kwargs) -> "MavlinkDriver":
        """e.g. open("/dev/serial0") on the drone, open("udpin:127.0.0.1:14551") for SITL."""
        from pymavlink import mavutil
        return cls(mavutil.mavlink_connection(url, baud=baud), mavutil.mavlink, **kwargs)

    # ---- incoming messages ------------------------------------------------
    def _pump(self) -> None:
        """Read everything waiting on the link and remember the latest values."""
        m = self.mav
        while True:
            msg = self.conn.recv_match(blocking=False)
            if msg is None:
                return
            kind = msg.get_type()
            if kind == "HEARTBEAT":
                if getattr(msg, "type", None) == m.MAV_TYPE_GCS:
                    continue  # another ground station, not the autopilot
                self._mode = msg.custom_mode
                self._armed = bool(msg.base_mode & m.MAV_MODE_FLAG_SAFETY_ARMED)
                self._last_heartbeat = self._clock()
            elif kind == "LOCAL_POSITION_NED":
                self._north, self._east, self._down = msg.x, msg.y, msg.z
            elif kind == "ATTITUDE":
                self._yaw = msg.yaw
            elif kind == "SYS_STATUS":
                self._battery = None if msg.battery_remaining < 0 else float(msg.battery_remaining)
            elif kind == "COMMAND_ACK" and msg.command == m.MAV_CMD_DO_MOUNT_CONTROL:
                self._on_mount_ack(msg.result)

    def _on_mount_ack(self, result: int) -> None:
        m = self.mav
        if result in (m.MAV_RESULT_ACCEPTED, m.MAV_RESULT_IN_PROGRESS):
            self._mount_refusals = 0
            return
        if result == m.MAV_RESULT_TEMPORARILY_REJECTED:
            return  # "try again later": the resend timer does that, and it is not evidence of no gimbal
        self._mount_refusals += 1  # DENIED, UNSUPPORTED, FAILED, ...
        if self._mount_refusals >= self.gimbal_max_refusals and not self._mount_gave_up:
            self._mount_gave_up = True
            log.warning("flight controller refused the camera tilt command %d times in a row "
                        "(MAV_RESULT %s): assuming there is no gimbal and no longer sending tilt "
                        "commands. With a fixed camera, pass gimbal=False to MavlinkDriver.",
                        self._mount_refusals, result)

    def _wait(self, condition: Callable[[], bool], timeout_s: float, what: str) -> None:
        deadline = self._clock() + timeout_s
        while True:
            self._pump()
            if condition():
                return
            if self._clock() > deadline:
                raise RuntimeError(f"timed out waiting for {what}")
            self._sleep(0.05)

    # ---- BaseDriver: lifecycle --------------------------------------------
    def connect(self) -> None:
        self.conn.wait_heartbeat(timeout=10)
        m, c = self.mav, self.conn
        for msg_id in (m.MAVLINK_MSG_ID_LOCAL_POSITION_NED, m.MAVLINK_MSG_ID_ATTITUDE,
                       m.MAVLINK_MSG_ID_SYS_STATUS):
            c.mav.command_long_send(c.target_system, c.target_component,
                                    m.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                    msg_id, 100_000, 0, 0, 0, 0, 0)  # 10 Hz
        self._wait(lambda: self._mode is not None, 5, "a heartbeat from the flight controller")

    def arm(self) -> None:
        self.conn.set_mode("GUIDED")
        self._wait(lambda: self._mode == COPTER_GUIDED, 5, "GUIDED mode")
        c, m = self.conn, self.mav
        c.mav.command_long_send(c.target_system, c.target_component,
                                m.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
        try:
            self._wait(lambda: self._armed, 10, "arming")
        except RuntimeError:
            raise RuntimeError("the flight controller refused to arm: read its PreArm "
                               "messages in QGroundControl (GPS lock? compass? calibration?)")

    def takeoff(self) -> None:
        c, m = self.conn, self.mav
        c.mav.command_long_send(c.target_system, c.target_component,
                                m.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, self.takeoff_alt_m)
        self._wait(lambda: -self._down >= 0.9 * self.takeoff_alt_m, 20, "takeoff altitude")

    def send(self, cmd: DriveCommand) -> None:
        if not self.autonomy_permitted():
            return  # belt and braces: a human has the controls
        c, m = self.conn, self.mav
        c.mav.set_position_target_local_ned_send(
            0, c.target_system, c.target_component,
            m.MAV_FRAME_BODY_OFFSET_NED, VELOCITY_AND_YAW_RATE,
            0, 0, 0,                                          # position (ignored)
            cmd.forward_mps, cmd.right_mps, -cmd.up_mps,      # velocity: forward, right, down (NED: down = -up)
            0, 0, 0,                                          # acceleration (ignored)
            0.0, math.radians(cmd.yaw_rate_dps))  # yaw (ignored), yaw rate rad/s (+ = clockwise)

    def set_camera_pitch(self, pitch_down_deg: float) -> None:
        """Best-effort gimbal aim via the legacy DO_MOUNT_CONTROL command
        (MAV_MOUNT_MODE_MAVLINK_TARGETING). Widely supported by simple
        servo gimbals; verify the parameter order against your specific
        gimbal/autopilot combination (some expect DO_GIMBAL_MANAGER_PITCHYAW
        instead) before relying on it in flight.

        Called every control step, so it only sends when the angle moved by
        gimbal_min_step_deg or more since the last send, or gimbal_resend_s has
        passed. If the flight controller keeps refusing (no gimbal configured,
        as in SITL), it stops sending after gimbal_max_refusals in a row."""
        self._pump()  # pick up the flight controller's answer to the previous tilt command
        if not self.gimbal or self._mount_gave_up:
            return
        now = self._clock()
        if (self._mount_pitch_sent is not None
                and abs(pitch_down_deg - self._mount_pitch_sent) < self.gimbal_min_step_deg
                and now - self._mount_sent_at < self.gimbal_resend_s):
            return
        c, m = self.conn, self.mav
        c.mav.command_long_send(c.target_system, c.target_component,
                                m.MAV_CMD_DO_MOUNT_CONTROL, 0,
                                -pitch_down_deg, 0.0, 0.0, 0, 0, 0, 2)  # pitch, roll, yaw, -, -, -, MAVLINK_TARGETING
        self._mount_pitch_sent, self._mount_sent_at = pitch_down_deg, now

    def stop(self) -> None:
        if self.autonomy_permitted():
            self.send(DriveCommand())

    def land(self) -> None:
        self.conn.set_mode("LAND")

    def return_to_launch(self) -> None:
        self.conn.set_mode("RTL")

    def disconnect(self) -> None:
        self.conn.close()

    # ---- BaseDriver: state ------------------------------------------------
    def pose(self) -> Pose:
        self._pump()
        # MAVLink local frame is NED (x north, y east, z down): our x is east,
        # y is north, z (up) is the negative of NED down.
        return Pose(self._east, self._north, self._yaw, -self._down)

    def airborne(self) -> bool:
        self._pump()
        return self._armed and -self._down > 0.3   # NED: down is negative when above home

    def battery_pct(self) -> Optional[float]:
        self._pump()
        return self._battery

    def autonomy_permitted(self) -> bool:
        self._pump()
        fresh = self._clock() - self._last_heartbeat < self.heartbeat_timeout_s
        return fresh and self._mode == COPTER_GUIDED
