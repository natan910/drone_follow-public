"""BaseDriver for the toy world: lets the exact same autopilot code fly it."""

from control.base_driver import BaseDriver
from datatypes import DriveCommand, Pose
from sim.virtual_world import VirtualWorld


class SimDriver(BaseDriver):
    def __init__(self, world: VirtualWorld):
        self.world = world
        self.cmd = DriveCommand()
        self.armed = False

    def connect(self) -> None: pass
    def disconnect(self) -> None: pass

    def arm(self) -> None:
        self.armed = True

    def takeoff(self) -> None:
        self.world.landed = False

    def send(self, cmd: DriveCommand) -> None:
        self.cmd = cmd

    def stop(self) -> None:
        self.cmd = DriveCommand()

    def land(self) -> None:
        self.world.landed = True
        self.cmd = DriveCommand()

    def set_camera_pitch(self, pitch_down_deg: float) -> None:
        self.world.set_camera_pitch(pitch_down_deg)

    def pose(self) -> Pose:
        return self.world.pose

    def battery_pct(self) -> float:
        return self.world.battery_pct

    def autonomy_permitted(self) -> bool:
        return not self.world.pilot_override
