"""
A Platform is everything outside the autopilot: it gathers an Observation from
sensors and applies a Decision to a driver. Swap the platform, keep the brain.

    SimPlatform   the toy world (tests, demos)
    RealPlatform  camera + face matcher + range sensors + a driver (webcam dry
                  run on a laptop, or the real drone)
"""

from abc import ABC, abstractmethod

from control.base_driver import BaseDriver
from datatypes import Decision, Mode, Observation


class Platform(ABC):
    driver: BaseDriver

    @abstractmethod
    def now(self) -> float:
        """Current time in seconds (simulated or real)."""

    @abstractmethod
    def observe(self) -> Observation:
        """Read all sensors."""

    def apply(self, decision: Decision) -> None:
        if decision.mode == Mode.IDLE:
            return  # the pilot has control: send nothing at all
        if decision.camera_pitch_deg is not None:
            self.driver.set_camera_pitch(decision.camera_pitch_deg)
        if decision.mode == Mode.LAND:
            self.driver.stop()
            self.driver.land()
            return
        self.driver.send(decision.cmd)

    def close(self) -> None:
        pass
