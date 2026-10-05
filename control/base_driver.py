"""
The only interface the rest of the code uses to move (and read the state of)
a vehicle.

The autopilot never knows whether it is driving a print statement, the toy
simulator, a real flight controller, or a robot dog. To support a new body,
subclass BaseDriver and implement these methods; nothing else changes.
"""

from abc import ABC, abstractmethod
from typing import Optional

from datatypes import DriveCommand, Pose


class BaseDriver(ABC):
    # ---- lifecycle -------------------------------------------------------
    @abstractmethod
    def connect(self) -> None:
        """Open the link to the vehicle / simulator."""

    @abstractmethod
    def arm(self) -> None:
        """Enable the motors."""

    @abstractmethod
    def takeoff(self) -> None:
        """Climb to a hover. Should block until hovering."""

    @abstractmethod
    def send(self, cmd: DriveCommand) -> None:
        """Apply a body-frame velocity command. Called every step."""

    @abstractmethod
    def stop(self) -> None:
        """Zero all velocity and hold position. Must be safe to call anytime."""

    @abstractmethod
    def land(self) -> None:
        """Land at the current position."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the link."""

    # ---- state (override where the vehicle can report it) ----------------
    @abstractmethod
    def pose(self) -> Pose:
        """Position relative to home, and heading."""

    def set_camera_pitch(self, pitch_down_deg: float) -> None:
        """Tilt the camera (0 = level, 90 = straight down). No-op if there is
        no gimbal (a fixed camera): the autopilot then uses the fixed angle
        from CameraConfig instead of asking the driver to move anything."""

    def battery_pct(self) -> Optional[float]:
        """Remaining battery 0-100, or None if unknown."""
        return None

    def autonomy_permitted(self) -> bool:
        """False when a human has taken control (e.g. flipped the mode switch).
        The autopilot must then send nothing at all."""
        return True

    def airborne(self) -> bool:
        """True only when a REAL vehicle is off the ground. Ground-only
        procedures (camera calibration) refuse to run while this is True.
        Default False: dry-run and simulated drivers never really fly."""
        return False

    # Context manager: guarantees stop -> land -> disconnect however the
    # program ends (normal exit, Ctrl-C, or an exception mid-flight).
    def __enter__(self) -> "BaseDriver":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for step in (self.stop, self.land, self.disconnect):
            try:
                step()
            except Exception as e:  # keep going: every step must be attempted
                print(f"[driver] {step.__name__} failed during shutdown: {e}")
