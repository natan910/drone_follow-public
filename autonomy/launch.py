"""
Launch gate: the drone stays on the ground, disarmed, until someone asks it to
launch AND the preflight checks pass.

    gate = LaunchGate(cfg.safety, driver, auto=args.auto_launch)
    if gate.wait(platform, autopilot, on_step):   # loops: camera, photos, tasks, status
        driver.arm(); ...; driver.takeoff(); run(...)

Before launch the loop runs as usual (camera, matcher, phone page, fleet
console, calibration), but the autopilot is not stepped and nothing is sent to
the vehicle. Launch comes from:
    the phone page / fleet console "Launch" button: command {"launch": true}
    main.py --auto-launch (SITL, dry runs): as soon as preflight passes

Preflight (launch_problems), all must pass:
    - the flight controller's heartbeat is fresh (MAVLink drivers only)
    - battery known (when SafetyConfig.require_battery) and above battery_return_pct
The flight controller's own pre-arm checks (GPS, EKF, compass) still run at
arming and can still refuse; that failure comes back here (LaunchGate.failed)
and the gate waits again instead of exiting.

Threading: everything here runs on the main loop. The HTTP threads only queue
the command; RealPlatform checks it (with launch_problems) and answers.
"""

import time
from typing import Callable, List, Optional

from config import SafetyConfig


def launch_problems(battery_pct: Optional[float], link_ok: bool, cfg: SafetyConfig) -> List[str]:
    """Why the drone must not launch right now. Empty list = go."""
    problems = []
    if not link_ok:
        problems.append("no heartbeat from the flight controller")
    if battery_pct is None:
        if cfg.require_battery:
            problems.append("battery level unknown (the flight controller reports none: "
                            "check its BATT_* settings)")
    elif battery_pct <= cfg.battery_return_pct:
        problems.append(f"battery {battery_pct:.0f} % is at or below the return level "
                        f"({cfg.battery_return_pct:.0f} %)")
    return problems


def link_ok(driver) -> bool:
    """Is the vehicle link alive? MAVLink drivers (ArduPilot and PX4) keep the time
    of the last autopilot heartbeat; other drivers (print, native-sim) have no link
    that can drop, so they always count as alive. Read before arming, when
    autonomy_permitted() is still False (not in GUIDED / OFFBOARD yet)."""
    last = getattr(driver, "_last_heartbeat", None)
    clock = getattr(driver, "_clock", None)
    if last is None or clock is None:
        return True
    return clock() - last < getattr(driver, "heartbeat_timeout_s", 2.0)


class LaunchGate:
    def __init__(self, cfg: SafetyConfig, driver, auto: bool = False,
                 clock: Callable[[], float] = time.monotonic, retry_s: float = 5.0,
                 say: Callable[[str], None] = print):
        self.cfg, self.driver, self.auto = cfg, driver, auto
        self.clock, self.retry_s, self.say = clock, retry_s, say
        self.last_error: Optional[str] = None     # the last refusal / failed arming, for the status page
        self._next_try = -1e18
        self._said: Optional[str] = None

    def problems(self) -> List[str]:
        return launch_problems(self.driver.battery_pct(), link_ok(self.driver), self.cfg)

    def status(self, problems: Optional[List[str]] = None, state: str = "waiting") -> dict:
        """JSON-friendly, for the phone page / console ("launch" key of the status)."""
        problems = self.problems() if problems is None else problems
        return {"state": state, "ready": not problems, "problems": problems, "auto": self.auto,
                "last_error": self.last_error}

    def failed(self, error: str) -> None:
        """Arming or takeoff was refused after the gate opened: show why, wait again."""
        self.last_error = error
        self._next_try = self.clock() + self.retry_s
        self.say(f"Launch failed: {error}. Waiting for launch again.")

    def wait(self, platform, autopilot, on_step: Optional[Callable] = None) -> bool:
        """Loop on the ground until launch. True = launch now; False = told to stop
        (on_step returned False). on_step(obs, status_dict) runs every loop."""
        platform.launch_check = self.problems   # a Launch tap is refused at once, with the reason
        self.say("Waiting for launch: " + ("automatic (--auto-launch) once preflight passes"
                                           if self.auto else "tap Launch on the phone page or fleet console"))
        while True:
            obs = platform.observe()
            autopilot.apply_operator(obs)       # tasks / hover height / calibration sent on the ground
            problems = self.problems()
            requested = platform.take_launch_request()
            if self.auto and self.clock() >= self._next_try:
                requested = True
            go = requested and not problems
            if requested and problems and not self.auto:
                self.last_error = "launch refused: " + "; ".join(problems)   # changed since it was accepted
            self._announce(problems)
            status = self.status(problems, "launching" if go else "waiting")
            if on_step is not None and on_step(obs, status) is False:
                return False
            if go:
                self.say("Launching.")
                return True

    def _announce(self, problems: List[str]) -> None:
        """Print the preflight state when it changes (not every loop)."""
        text = "; ".join(problems) if problems else "preflight OK"
        if text != self._said:
            self._said = text
            self.say(("Preflight: " + text) if problems else "Preflight OK.")
