"""Closed-loop harness: plant + sensors, one tick at a time."""
from __future__ import annotations

import numpy as np

from ..config import VehicleParams
from ..hal import SensorFrame
from .dynamics import Plant, WindModel
from .sensors import SensorSpec, SensorSuite


class SimWorld:
    def __init__(
        self,
        vehicle: VehicleParams | None = None,
        spec: SensorSpec | None = None,
        *,
        dt: float = 0.002,
        seed: int = 1,
        wind_mean=(0.0, 0.0, 0.0),
        gust_std: float = 0.0,
        motor_scale=(1.0, 1.0, 1.0, 1.0),
        yaw0: float = 0.0,
        p0=(0.0, 0.0, 0.0),
    ):
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.plant = Plant(
            vehicle or VehicleParams(),
            motor_scale=motor_scale,
            wind=WindModel(mean=np.asarray(wind_mean, dtype=float), gust_std=gust_std),
            rng=self.rng,
            yaw0=yaw0,
            p0=p0,
        )
        self.sensors = SensorSuite(spec or SensorSpec(), dt, self.rng)
        self.t = 0.0
        self._last_cmd = np.zeros(4)

    @property
    def truth(self) -> Plant:
        return self.plant

    def step(self, motor_cmd=None) -> SensorFrame:
        """Advance one tick with the given ESC commands (held for the whole tick)."""
        if motor_cmd is not None:
            self._last_cmd = np.asarray(motor_cmd, dtype=float)
        self.plant.step(self._last_cmd, self.dt)
        self.t += self.dt
        return self.sensors.sample(self.plant, self.t)
