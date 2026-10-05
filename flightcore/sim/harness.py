"""ClosedLoop: FlightCore + SimWorld wired together, with scripted events and a trace recorder."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..autopilot import FlightCore
from ..config import FlightConfig, VehicleParams
from ..runtime import FlightLoop
from .sensors import SensorSpec
from .world import SimWorld


@dataclass
class Trace:
    t: list = field(default_factory=list)
    p: list = field(default_factory=list)        # truth NED
    v: list = field(default_factory=list)
    p_est: list = field(default_factory=list)
    v_est: list = field(default_factory=list)
    rpy: list = field(default_factory=list)      # truth, rad
    mode: list = field(default_factory=list)
    cmd: list = field(default_factory=list)

    def arr(self, name: str) -> np.ndarray:
        return np.asarray(getattr(self, name))


class ClosedLoop:
    def __init__(
        self,
        cfg: FlightConfig | None = None,
        *,
        true_vehicle: VehicleParams | None = None,
        spec: SensorSpec | None = None,
        seed: int = 1,
        wind=(0.0, 0.0, 0.0),
        gust_std: float = 0.0,
        motor_scale=(1.0, 1.0, 1.0, 1.0),
        yaw0: float = 0.0,
        p0=(0.0, 0.0, 0.0),
        record_every: int = 25,
        recorder=None,
    ):
        self.recorder = recorder                 # optional callable(frame, airborne), e.g. CsvRecorder(f).record
        self.cfg = cfg or FlightConfig()
        self.world = SimWorld(
            true_vehicle or self.cfg.vehicle, spec, dt=self.cfg.dt, seed=seed, wind_mean=wind,
            gust_std=gust_std, motor_scale=motor_scale, yaw0=yaw0, p0=p0,
        )
        self.core = FlightCore(self.cfg)
        self.cmd = np.zeros(4)
        self.k = 0
        self.trace = Trace()
        self._every = max(1, record_every)
        self._events: list[tuple[float, Callable]] = []
        self.max_tilt = 0.0

    @property
    def t(self) -> float:
        return self.world.t

    @property
    def truth(self):
        return self.world.plant

    @property
    def altitude(self) -> float:
        return self.world.plant.altitude

    def at(self, t: float, fn: Callable[["ClosedLoop"], None]):
        self._events.append((t, fn))
        self._events.sort(key=lambda e: e[0])

    def tick(self):
        frame = self.world.step(self.cmd)
        if self.recorder is not None:
            self.recorder(frame, self.core.sup.airborne)
        out = self.core.step(frame)
        self.cmd = out.cmd if out.armed else np.zeros(4)
        self.k += 1
        while self._events and self._events[0][0] <= self.world.t:
            self._events.pop(0)[1](self)
        pl = self.world.plant
        self.max_tilt = max(self.max_tilt, math.acos(max(-1.0, min(1.0, 1 - 2 * (pl.q[1] ** 2 + pl.q[2] ** 2)))))
        if self.k % self._every == 0:
            tr = self.trace
            tr.t.append(self.world.t)
            tr.p.append(pl.p.copy())
            tr.v.append(pl.v.copy())
            tr.p_est.append(self.core.nav.p.copy())
            tr.v_est.append(self.core.nav.v.copy())
            tr.rpy.append(pl.euler())
            tr.mode.append(self.core.mode.value)
            tr.cmd.append(self.cmd.copy())

    def run(self, seconds: float):
        end = self.world.t + seconds
        while self.world.t < end - 1e-9:
            self.tick()

    def run_until(self, pred: Callable[["ClosedLoop"], bool], timeout: float) -> bool:
        end = self.world.t + timeout
        while self.world.t < end:
            self.tick()
            if pred(self):
                return True
        return False

    def fly_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float, seconds: float, resend: float = 0.1):
        """Send a body-frame velocity command repeatedly (like the brain does) for `seconds`."""
        end = self.world.t + seconds
        while self.world.t < end - 1e-9:
            self.core.set_velocity_body(vx, vy, vz, yaw_rate)
            self.run(min(resend, end - self.world.t))

    def wait_ready(self, timeout: float = 8.0) -> bool:
        """Run until the estimator is aligned and the pre-arm checks pass."""
        return self.run_until(lambda s: s.core.est.initialized and not s.core.prearm_check(), timeout)

    def arm_and_takeoff(self, alt: float, timeout: float = 15.0) -> bool:
        if not self.wait_ready():
            return False
        ok, _ = self.core.arm()
        if not ok:
            return False
        self.core.takeoff(alt)
        return self.run_until(lambda s: s.core.mode.value != "takeoff", timeout)


class SimSource:
    """SensorSource + MotorDriver backed by a SimWorld: lets FlightLoop run against the simulator."""

    def __init__(self, world):
        self.world = world
        self.cmd = np.zeros(4)
        self.drop_next = 0                       # test hook: raise TimeoutError this many times

    def read(self):
        if self.drop_next > 0:
            self.drop_next -= 1
            raise TimeoutError("imu")
        return self.world.step(self.cmd)

    def write(self, out) -> None:
        self.cmd = out.cmd if out.armed else np.zeros(4)


class SimRunner:
    """Virtual-time runner: the simulated vehicle advances only when the caller asks (sync / wait), by as
    much time as has passed on `clock` since the last call (times `speed`).  No threads, so it is
    deterministic under a fake clock.  Between calls the last command keeps being flown, like a real drone.

    sync():  advance by the wall time elapsed since the last call, at most `max_catchup_s` (a long stall in the
             caller does not make the simulator grind through minutes of ticks; it means the stall is
             under-simulated, which is optimistic for the link-loss failsafe).
    wait(s): advance `s` seconds of vehicle time immediately, without sleeping (startup/landing waits
             therefore run faster than real time)."""

    def __init__(self, core: FlightCore, world, *, clock: Callable[[], float] = time.monotonic,
                 speed: float = 1.0, max_catchup_s: float = 5.0):
        self.core, self.world = core, world
        self._clock = clock
        self.speed = speed
        self.max_catchup_s = max_catchup_s
        src = SimSource(world)
        self.loop = FlightLoop(core, src, src)
        self._last: Optional[float] = None
        self._owed = 0.0                               # vehicle seconds not yet simulated (< one tick)
        self.error = None

    def start(self) -> None:
        self._last = self._clock()

    def stop(self) -> None:
        pass

    def _advance(self, seconds: float) -> None:
        dt = self.world.dt
        self._owed += seconds
        n = int(self._owed / dt + 1e-9)
        self._owed -= n * dt
        if n:
            self.loop.run(max_ticks=n)

    def sync(self) -> None:
        now = self._clock()
        if self._last is None:
            self._last = now
            return
        owed = min(max(now - self._last, 0.0) * self.speed, self.max_catchup_s)
        self._last = now
        self._advance(owed)

    def wait(self, seconds: float) -> None:
        self.sync()
        self._advance(seconds)
        self._last = self._clock()
