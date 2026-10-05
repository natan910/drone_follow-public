"""
A driver that prints what a real drone would be told to do, and dead-reckons
a pose (including altitude) from the commands so the rest of the stack has
something to work with. Use it to dry-run on a laptop with a webcam.
"""

import math
import time
from typing import Optional

from control.base_driver import BaseDriver
from datatypes import DriveCommand, Pose


class PrintDriver(BaseDriver):
    def __init__(self, print_interval: float = 0.25):
        self.print_interval = print_interval
        self._x = self._y = self._yaw = self._z = 0.0
        self._cmd = DriveCommand()
        self._pitch = 0.0
        self._last_print = 0.0
        self._last_tick: Optional[float] = None

    def _advance(self) -> None:
        now = time.monotonic()
        if self._last_tick is not None:
            dt = now - self._last_tick
            c = self._cmd
            self._yaw += math.radians(c.yaw_rate_dps) * dt
            self._x += (c.forward_mps * math.sin(self._yaw) + c.right_mps * math.cos(self._yaw)) * dt
            self._y += (c.forward_mps * math.cos(self._yaw) - c.right_mps * math.sin(self._yaw)) * dt
            self._z = max(0.0, self._z + c.up_mps * dt)
        self._last_tick = now

    def connect(self) -> None:
        print("[print-driver] connected")

    def arm(self) -> None:
        print("[print-driver] ARM")

    def takeoff(self) -> None:
        self._z = 1.0
        print("[print-driver] TAKEOFF")

    def send(self, cmd: DriveCommand) -> None:
        self._advance()
        self._cmd = cmd
        now = time.monotonic()
        if now - self._last_print >= self.print_interval:
            self._last_print = now
            print(f"[print-driver] yaw {cmd.yaw_rate_dps:+6.1f} deg/s  fwd {cmd.forward_mps:+5.2f}  "
                  f"right {cmd.right_mps:+5.2f}  up {cmd.up_mps:+5.2f} m/s  (z={self._z:4.2f}m)")

    def stop(self) -> None:
        self._advance()
        self._cmd = DriveCommand()
        print("[print-driver] STOP (hold position)")

    def land(self) -> None:
        self._z = 0.0
        print("[print-driver] LAND")

    def disconnect(self) -> None:
        print("[print-driver] disconnected")

    def set_camera_pitch(self, pitch_down_deg: float) -> None:
        if abs(pitch_down_deg - self._pitch) > 5:
            self._pitch = pitch_down_deg
            print(f"[print-driver] camera pitch -> {pitch_down_deg:.0f} deg down")

    def pose(self) -> Pose:
        self._advance()
        return Pose(self._x, self._y, self._yaw, self._z)
