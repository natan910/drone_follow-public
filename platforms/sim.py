from control.base_driver import BaseDriver
from datatypes import Decision, DriveCommand, Observation
from platforms.base import Platform
from sim.sim_driver import SimDriver
from sim.virtual_world import VirtualWorld


class SimPlatform(Platform):
    def __init__(self, world: VirtualWorld, dt: float = 1 / 15):
        self.world, self.dt = world, dt
        self.driver: BaseDriver = SimDriver(world)

    def now(self) -> float:
        return self.world.t

    def observe(self) -> Observation:
        return Observation(
            now=self.world.t,
            pose=self.driver.pose(),
            detection=self.world.detection(),
            scan=self.world.scan(),
            battery_pct=self.driver.battery_pct(),
            autonomy_permitted=self.driver.autonomy_permitted(),
        )

    def apply(self, decision: Decision) -> None:
        super().apply(decision)
        self.world.step(self.driver.cmd, self.dt)  # advance simulated time
