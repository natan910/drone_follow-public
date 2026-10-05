"""Real-time loop glue: SensorSource -> FlightCore -> MotorDriver, with timing statistics and a
sensor-loss watchdog.  Everything hardware-facing is injected (clock, source, motors), so the same
class runs against the simulator in tests and against real drivers on the vehicle.

Honest limits: this is Python.  On a Linux companion computer expect ~ms scheduling jitter, so run the
loop at 100-250 Hz there (set FlightConfig.loop_hz and re-tune the rate loop) or keep the rate/attitude
loops on an MCU.  `stats` tells you what you actually got.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np

from .autopilot import FlightCore
from .hal import MotorDriver, MotorOutput, SensorFrame


class SensorSource(Protocol):
    def read(self) -> Optional[SensorFrame]:
        """Block until the next IMU tick.  Return None when the stream has ended;
        raise TimeoutError when the IMU stopped delivering."""


@dataclass
class LoopStats:
    ticks: int = 0
    overruns: int = 0                # step() took longer than the tick period
    source_timeouts: int = 0
    consecutive_timeouts: int = 0
    max_step_s: float = 0.0
    total_step_s: float = 0.0
    killed_by_watchdog: bool = False

    @property
    def mean_step_s(self) -> float:
        return self.total_step_s / self.ticks if self.ticks else 0.0


class FlightLoop:
    def __init__(
        self,
        core: FlightCore,
        source: SensorSource,
        motors: MotorDriver,
        *,
        clock: Callable[[], float] = time.perf_counter,
        max_consecutive_timeouts: int = 25,
        recorder: Optional[Callable[[SensorFrame, bool], None]] = None,
    ):
        self.core = core
        self.source = source
        self.motors = motors
        self.clock = clock
        self.max_timeouts = max_consecutive_timeouts
        self.recorder = recorder          # e.g. CsvRecorder(f).record; called as recorder(frame, airborne)
        self.stats = LoopStats()
        self._period = core.cfg.dt
        self._stop = threading.Event()

    def run_once(self) -> bool:
        """One tick.  False when the source has ended."""
        try:
            frame = self.source.read()
        except TimeoutError:
            return self._on_timeout()
        if frame is None:
            return False
        self.stats.consecutive_timeouts = 0
        if self.recorder is not None:
            self.recorder(frame, self.core.sup.airborne)
        t0 = self.clock()
        try:
            out = self.core.step(frame)
        except Exception:
            self._motors_off()                 # a crashed controller must not leave the last command applied
            raise
        dt = self.clock() - t0
        self.motors.write(out)
        st = self.stats
        st.ticks += 1
        st.total_step_s += dt
        st.max_step_s = max(st.max_step_s, dt)
        if dt > self._period:
            st.overruns += 1
        return True

    def _motors_off(self) -> None:
        try:
            self.motors.write(MotorOutput(t=self.core.t, cmd=np.zeros(4), armed=False))
        except Exception:                      # noqa: BLE001 - already failing; nothing better to do
            pass

    def request_stop(self) -> None:
        """Ask run() to return after the current tick (safe from any thread)."""
        self._stop.set()

    def clear_stop(self) -> None:
        self._stop.clear()

    def _on_timeout(self) -> bool:
        """No IMU data.  Without it nothing can be controlled: outputs go to zero at once and, if it
        persists, the core is killed so it cannot restart the motors when data returns."""
        st = self.stats
        st.source_timeouts += 1
        st.consecutive_timeouts += 1
        self.motors.write(MotorOutput(t=self.core.t, cmd=np.zeros(4), armed=False))
        if st.consecutive_timeouts >= self.max_timeouts and not st.killed_by_watchdog:
            st.killed_by_watchdog = True
            self.core.kill()
        return True

    def run(self, max_ticks: Optional[int] = None) -> LoopStats:
        n = 0
        while (max_ticks is None or n < max_ticks) and not self._stop.is_set() and self.run_once():
            n += 1
        return self.stats


class Runner(Protocol):
    """How a vehicle's time passes.  NativeDriver talks only to this."""

    error: Optional[BaseException]

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def sync(self) -> None:
        """Bring the vehicle up to 'now'.  Raises RuntimeError if the flight loop is dead."""
    def wait(self, seconds: float) -> None:
        """Let `seconds` of vehicle time pass (a real runner sleeps; a simulated one just steps)."""


class ThreadRunner:
    """Runs a FlightLoop on its own thread (real hardware: the blocking IMU read paces the loop).

    If the thread dies, sync() raises so the caller cannot keep flying blind.  On the vehicle itself
    the motor/ESC layer still needs its own watchdog: this thread dying, or the whole process dying,
    cannot be trusted to zero the motors by itself."""

    def __init__(self, loop: FlightLoop, *, sleep: Callable[[float], None] = time.sleep, join_timeout_s: float = 2.0):
        self.loop = loop
        self._sleep = sleep
        self._join_timeout = join_timeout_s
        self._thread: Optional[threading.Thread] = None
        self._stopping = False
        self.error: Optional[BaseException] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.error, self._stopping = None, False
        self.loop.clear_stop()
        self._thread = threading.Thread(target=self._main, name="flight-loop", daemon=True)
        self._thread.start()

    def _main(self) -> None:
        try:
            self.loop.run()
        except BaseException as e:             # noqa: BLE001 - handed to sync()
            self.error = e

    def stop(self) -> None:
        self._stopping = True
        self.loop.request_stop()
        if self._thread is not None:
            self._thread.join(self._join_timeout)

    def sync(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"the flight loop died: {self.error!r}") from self.error
        t = self._thread
        if t is not None and not t.is_alive() and not self._stopping:
            raise RuntimeError("the flight loop stopped (sensor stream ended)")

    def wait(self, seconds: float) -> None:
        self._sleep(seconds)
        self.sync()
