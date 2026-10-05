"""Append-only JSON-lines flight recorder: one line per step, easy to load in
pandas or grep. Enough to replay what the autopilot saw and decided."""

import json
from typing import Optional

from datatypes import Decision, Observation


class FlightLog:
    def __init__(self, path: str, flush_every_s: float = 1.0):
        self._f = open(path, "w", buffering=1)
        self._flush_every_s = flush_every_s
        self._last_flush = 0.0

    def log(self, obs: Observation, decision: Decision) -> None:
        d = obs.detection
        record = {
            "t": round(obs.now, 3),
            "mode": decision.mode.name,
            "note": decision.note,
            "x": round(obs.pose.x, 2), "y": round(obs.pose.y, 2), "z": round(obs.pose.z, 2),
            "yaw": round(obs.pose.yaw, 3),
            "yaw_cmd": round(decision.cmd.yaw_rate_dps, 1),
            "fwd_cmd": round(decision.cmd.forward_mps, 2),
            "right_cmd": round(decision.cmd.right_mps, 2),
            "up_cmd": round(decision.cmd.up_mps, 2),
            "gimbal": None if decision.camera_pitch_deg is None else round(decision.camera_pitch_deg, 1),
            "battery": obs.battery_pct,
            "seen": None if d is None else [round(d.offset_x, 2), round(d.offset_y, 2), round(d.size, 3), d.source],
            "ranges": None if obs.scan is None else
                      [None if b.distance is None else round(b.distance, 2) for b in obs.scan.beams],
            "down": None if obs.scan is None or obs.scan.down is None else round(obs.scan.down, 2),
        }
        self._f.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._f.close()
