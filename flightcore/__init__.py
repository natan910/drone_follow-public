"""flightcore: own flight stack (estimator + control + modes/failsafes) with a closed-loop simulator.

Pure numpy, written from textbook math (no ArduPilot / PX4 code).  Reference implementation:
runs the same code in simulation and on a companion computer; hard-real-time inner loops
belong on an MCU port (see docs in HANDOFF.md).
"""
