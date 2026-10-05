"""The observe -> decide -> act loop, shared by main.py, the demos, and tests."""

from typing import Callable, Optional

from autonomy.autopilot import Autopilot
from datatypes import Decision, Mode, Observation
from platforms.base import Platform

StepHook = Callable[[Observation, Decision], Optional[bool]]


def run(platform: Platform, autopilot: Autopilot, seconds: Optional[float] = None,
        max_steps: Optional[int] = None, on_step: Optional[StepHook] = None) -> int:
    """Loop until told to stop; returns the number of steps taken. Stops after a
    LAND decision, when `seconds` of platform time have passed, after `max_steps`,
    or when `on_step` returns False."""
    start, steps = platform.now(), 0
    while True:
        obs = platform.observe()
        decision = autopilot.step(obs)
        platform.apply(decision)
        steps += 1
        if on_step is not None and on_step(obs, decision) is False:
            break
        if decision.mode == Mode.LAND:
            break
        if max_steps is not None and steps >= max_steps:
            break
        if seconds is not None and platform.now() - start >= seconds:
            break
    return steps
