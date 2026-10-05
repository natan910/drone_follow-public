"""
Safety supervisor: watches vital signs and can override whatever the drone was
about to do.

    OK      carry on
    HOLD    stand still until the problem clears (sensor stalls)
    RETURN  go home and land (low or unknown battery, geofence, altitude, time limit, camera lost)
    LAND    land right here, right now (critical battery, range sensors lost)

RETURN and LAND are latched: once triggered they stay triggered for the rest of
the flight. This is a second layer. The flight controller's own failsafes
(battery, radio loss, geofence) and the pilot's RC switch remain the first.

Heights are metres above the launch point (platforms/real.py makes the pose
launch-relative). The ceiling itself is enforced by the autopilot on every
command (Autopilot._limit_climb); this only catches "above it anyway".
"""

import math
from enum import IntEnum
from typing import Optional, Tuple

from config import SafetyConfig
from datatypes import Observation


class SafetyAction(IntEnum):
    OK = 0
    HOLD = 1
    RETURN = 2
    LAND = 3


class SafetySupervisor:
    def __init__(self, config: Optional[SafetyConfig] = None, home=(0.0, 0.0)):
        self.cfg = config or SafetyConfig()
        self.home = home
        self.reset()

    def reset(self) -> None:
        self._start: Optional[float] = None
        self._latched = SafetyAction.OK
        self._latched_reason = ""
        self._battery_unknown_since: Optional[float] = None

    def check(self, obs: Observation) -> Tuple[SafetyAction, str]:
        c = self.cfg
        if self._start is None:
            self._start = obs.now

        action, reason = SafetyAction.OK, ""

        def raise_to(level: SafetyAction, why: str) -> None:
            nonlocal action, reason
            if level > action:
                action, reason = level, why

        # -- things that stall us in place
        if obs.frame_age > c.frame_hold_s:
            raise_to(SafetyAction.HOLD, "camera stalled")
        if c.require_scan and (obs.scan is None or obs.scan_age > c.scan_hold_s):
            raise_to(SafetyAction.HOLD, "range sensors stalled")

        # -- things that send us home
        if obs.frame_age > c.frame_return_s:
            raise_to(SafetyAction.RETURN, "camera lost")
        if obs.now - self._start > c.max_flight_s:
            raise_to(SafetyAction.RETURN, "flight time limit")
        if math.hypot(obs.pose.x - self.home[0], obs.pose.y - self.home[1]) > c.geofence_radius_m:
            raise_to(SafetyAction.RETURN, "outside geofence")
        if obs.pose.z > c.max_altitude_m + c.altitude_return_margin_m:
            raise_to(SafetyAction.RETURN, "above altitude ceiling")
        if obs.battery_pct is not None and obs.battery_pct <= c.battery_return_pct:
            raise_to(SafetyAction.RETURN, "battery low")
        if self._battery_unknown_for(obs) > c.battery_unknown_return_s:
            raise_to(SafetyAction.RETURN, "battery level unknown")

        # -- things that put us on the ground
        if c.require_scan and obs.scan_age > c.scan_land_s:
            raise_to(SafetyAction.LAND, "range sensors lost")
        if obs.battery_pct is not None and obs.battery_pct <= c.battery_land_pct:
            raise_to(SafetyAction.LAND, "battery critical")

        if action >= SafetyAction.RETURN and action > self._latched:
            self._latched, self._latched_reason = action, reason
        if self._latched > action:
            return self._latched, self._latched_reason
        return action, reason

    def _battery_unknown_for(self, obs: Observation) -> float:
        """Seconds the battery level has been unknown in a row (0 when known,
        or when this vehicle is not required to report it)."""
        if not self.cfg.require_battery or obs.battery_pct is not None:
            self._battery_unknown_since = None
            return 0.0
        if self._battery_unknown_since is None:
            self._battery_unknown_since = obs.now
        return obs.now - self._battery_unknown_since
