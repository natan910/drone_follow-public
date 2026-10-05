"""
BaseDriver on top of flightcore (our own estimator + flight controller) instead of ArduPilot.

Same contract as MavlinkDriver, so the brain cannot tell the difference:
  * We only ever command body-frame velocity + yaw rate.
  * autonomy_permitted() is False whenever the flight core is not in HOLD/GUIDED (a failsafe took over:
    RTL, LAND, killed...). The autopilot then goes silent and send() drops any stray command.
  * arm() runs the flight core's own pre-arm checks and raises RuntimeError with the reasons if refused.
  * If the brain stops sending, the flight core itself holds, then lands (flightcore/config.py).

How time passes is injected (a "runner", see flightcore/runtime.py):
  * NativeDriver.simulated(): the vehicle is flightcore's own simulator, stepped in virtual time whenever
    the driver is called. No threads, no hardware, no ArduPilot. This is what `--driver native-sim` uses.
  * A real airframe needs a ThreadRunner around a FlightLoop fed by real sensor/motor drivers. Those do
    not exist yet, so there is no NativeDriver.open() and no `--driver native`.

Frames: the brain's Pose is x = east, y = north, z = up, yaw clockwise from north; flightcore is NED.
"""

import logging
import math
import threading
from typing import Callable, Optional

from control.base_driver import BaseDriver
from datatypes import DriveCommand, Pose
from flightcore.autopilot import FlightCore
from flightcore.config import FlightConfig
from flightcore.runtime import Runner
from flightcore.supervisor import Mode

log = logging.getLogger(__name__)

# The brain's own commands are only applied in these flight-core modes; anything else means a failsafe
# or an operator command has taken over.
_AUTONOMY_MODES = (Mode.HOLD, Mode.GUIDED)


class NativeDriver(BaseDriver):
    def __init__(self, core: FlightCore, runner: Runner, takeoff_alt_m: float = 1.5,
                 real_vehicle: bool = False,
                 ready_timeout_s: float = 30.0,
                 takeoff_timeout_s: float = 20.0,
                 land_timeout_s: float = 60.0,
                 poll_s: float = 0.05):
        self.core, self.runner = core, runner
        self.takeoff_alt_m = takeoff_alt_m
        self.real_vehicle = real_vehicle          # False: airborne() stays False, as BaseDriver asks of simulators
        self.ready_timeout_s = ready_timeout_s
        self.takeoff_timeout_s = takeoff_timeout_s
        self.land_timeout_s = land_timeout_s
        self.poll_s = poll_s
        self._lock = threading.RLock()            # the runner is not thread-safe; the brain should be the only caller

    @classmethod
    def simulated(cls, cfg: Optional[FlightConfig] = None, *, seed: int = 1, wind=(0.0, 0.0, 0.0),
                  gust_std: float = 0.0, spec=None, speed: float = 1.0,
                  clock: Optional[Callable[[], float]] = None, **kwargs) -> "NativeDriver":
        """flightcore flying flightcore's simulator. `clock` (default time.monotonic) says how much
        vehicle time passes between driver calls; `speed` scales it."""
        import time
        from flightcore.sim import SimWorld
        from flightcore.sim.harness import SimRunner
        cfg = cfg or FlightConfig()
        world = SimWorld(cfg.vehicle, spec, dt=cfg.dt, seed=seed, wind_mean=wind, gust_std=gust_std)
        core = FlightCore(cfg)
        runner = SimRunner(core, world, clock=clock or time.monotonic, speed=speed)
        driver = cls(core, runner, real_vehicle=False, **kwargs)
        driver.world = world                      # for tests and viewers
        return driver

    # ---- time -------------------------------------------------------------
    def _sync(self) -> None:
        with self._lock:
            self.runner.sync()

    def _wait(self, condition: Callable[[], bool], timeout_s: float, what: str) -> None:
        waited = 0.0
        while True:
            self._sync()
            if condition():
                return
            if waited >= timeout_s:
                raise RuntimeError(f"timed out waiting for {what}")
            with self._lock:
                self.runner.wait(self.poll_s)
            waited += self.poll_s

    # ---- BaseDriver: lifecycle --------------------------------------------
    def connect(self) -> None:
        with self._lock:
            self.runner.start()
        self._wait(lambda: self.core.est.initialized, self.ready_timeout_s,
                   "the estimator to align (vehicle must be still and sensors must be streaming)")

    def arm(self) -> None:
        try:
            self._wait(lambda: not self.core.prearm_check(), self.ready_timeout_s, "the pre-arm checks to pass")
        except RuntimeError:
            raise RuntimeError("the flight core refused to arm: " + ", ".join(self.core.prearm_check()))
        with self._lock:
            ok, why = self.core.arm()
        if not ok:
            raise RuntimeError("the flight core refused to arm: " + ", ".join(why))

    def takeoff(self) -> None:
        with self._lock:
            if not self.core.takeoff(self.takeoff_alt_m):
                raise RuntimeError(f"takeoff refused: the flight core is in mode '{self.core.mode.value}', not armed on the ground")
        self._wait(lambda: self.core.mode != Mode.TAKEOFF, self.takeoff_timeout_s, "takeoff altitude")
        if self.core.mode != Mode.HOLD:
            raise RuntimeError(f"takeoff aborted: mode '{self.core.mode.value}' ({self.core.sup.reason or 'no reason given'})")

    def send(self, cmd: DriveCommand) -> None:
        self._sync()                              # first fly out the time since the last call with the old command
        if not self.autonomy_permitted():
            return  # belt and braces: a failsafe or operator has the controls
        with self._lock:
            self.core.set_velocity_body(cmd.forward_mps, cmd.right_mps, -cmd.up_mps,   # NED: down = -up
                                        math.radians(cmd.yaw_rate_dps))                 # rad/s, + = clockwise

    def stop(self) -> None:
        if self.autonomy_permitted():
            self.send(DriveCommand())

    def land(self) -> None:
        self._sync()
        with self._lock:
            self.core.land()

    def return_to_launch(self) -> None:
        self._sync()
        with self._lock:
            self.core.rtl()

    def disconnect(self) -> None:
        """Let a landing in progress finish BEFORE the flight loop stops: a stopped loop means no
        control at all. If it will not land within land_timeout_s, log it and stop anyway."""
        try:
            if self.core.armed:
                try:
                    self._wait(lambda: not self.core.armed, self.land_timeout_s, "the vehicle to land and disarm")
                except RuntimeError as e:
                    log.error("%s; stopping the flight loop with the vehicle still armed", e)
        finally:
            with self._lock:
                self.runner.stop()

    # ---- BaseDriver: state ------------------------------------------------
    def pose(self) -> Pose:
        self._sync()
        n = self.core.nav
        # flightcore is NED (x north, y east, z down): our x is east, y is north, z is up.
        return Pose(float(n.p[1]), float(n.p[0]), float(n.yaw), -float(n.p[2]))

    def airborne(self) -> bool:
        self._sync()
        return self.real_vehicle and self.core.airborne and self.core.nav.altitude > 0.3

    def battery_pct(self) -> Optional[float]:
        self._sync()
        b = self.core.batt
        return None if b is None else 100.0 * float(b)

    def autonomy_permitted(self) -> bool:
        self._sync()
        return self.core.mode in _AUTONOMY_MODES and self.core.sup.latched is None
