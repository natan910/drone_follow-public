# flightcore: our own estimator + flight controller

Pure numpy quadrotor flight stack: sensors in, motor commands out. Built as a replacement for
ArduPilot GUIDED mode behind the driver seam. **Not adopted, not flown.** It has run only against
the simulator in this folder. The simulator shares assumptions with the controller (same author,
same model family), so passing it is necessary, not sufficient. Treat the first hardware hover as a
tethered test, props on a bench first.

## What is in it

    flightcore/
      config.py        every tunable (VehicleParams, ControlConfig, EstimatorConfig, SupervisorConfig)
      hal.py           sample types (ImuSample, GpsSample, ...), SensorFrame, MotorOutput, driver Protocols
      mathutil.py      quaternion / rotation helpers, frame conventions
      estimation/      eskf.py (16-state error-state Kalman filter), mahony.py (independent backup attitude)
      control/         position -> velocity -> acceleration -> attitude -> rate -> mixer
      supervisor.py    modes (DISARMED ARMED TAKEOFF HOLD GUIDED GOTO LAND RTL KILLED) + failsafes
      autopilot.py     FlightCore: estimator + supervisor + controller, thread-safe command API
      runtime.py       FlightLoop: source -> core -> motors, timing stats, IMU-loss watchdog;
                       ThreadRunner (loop on its own thread, for hardware)
      log.py           CSV flight log + offline replay through the estimator
      sim/             rigid-body plant, sensor models (noise, bias walk, latency, outages), ClosedLoop harness,
                       SimRunner (virtual-time runner behind NativeDriver.simulated)

    control/native_driver.py   (repo-level, next to mavlink_driver.py) BaseDriver on top of FlightCore

## Conventions (whole package)

- World NED (x north, y east, z down). Body FRD (x forward, y right, z down).
- Quaternion Hamilton `[w, x, y, z]`, rotates body to world. Yaw positive clockwise seen from above.
- Accelerometer is specific force: level and still reads `[0, 0, -9.81]`.
- Motor order FR, BL, FL, BR (ArduPilot order). Motor command 0..1 (ESC), thrust curve `(1-e)u + e u^2`.
- All timestamps are seconds on the IMU clock, and are the time the quantity was MEASURED. Late
  GPS/baro samples are fused by replaying the filter from the sample time (tested equal to on-time fusion).

## How a tick flows

    SensorFrame (IMU + whatever else arrived)
      -> Eskf.step           propagate, then fuse GPS pos/vel, baro (+bias), rangefinder (tilt-compensated),
                             mag heading, airborne drag, on-ground ZUPT/gravity; gate every innovation
      -> Mahony (shadow)     tracks the ESKF while GPS vouches for it; if they disagree without GPS, alarm
      -> Supervisor.update   mode logic, failsafes, landing detection  ->  Setpoint
      -> FlightController    position P -> velocity PI -> accel -> thrust vector -> SO(3) attitude P
                             -> rate PID -> torques -> mixer (priority roll/pitch > yaw > collective, airmode)
      -> MotorOutput

Failsafes (latching ones clear only on disarm): command silence -> hold -> land; battery warn -> RTL,
critical -> land; fence radius/altitude; flip > 80 deg for 0.4 s -> kill; estimator degradation ->
velocity hold / level hold / emergency level descent on the backup attitude; IMU stream lost (FlightLoop)
-> motors zero at once, kill after 25 missed ticks.

## Use

    from flightcore.autopilot import FlightCore
    from flightcore.runtime import FlightLoop

    core = FlightCore()                       # FlightConfig() defaults; pass your own for real hardware
    loop = FlightLoop(core, source, motors)   # source.read() -> SensorFrame; motors.write(MotorOutput)

    # from another thread (the brain): all commands are thread-safe
    core.prearm_check()                       # [] means OK, else a list of reasons
    core.arm(); core.takeoff(2.0)
    core.set_velocity_body(vx, vy, vz, yaw_rate)   # forward, right, down m/s; rad/s clockwise. Re-send at >= 2 Hz
    core.land(); core.status()                # status() is JSON-serialisable

    loop.run()                                # blocks; loop.stats has overruns and max step time

Through the brain's driver seam (`control/native_driver.py`, same contract as `MavlinkDriver`):

    python main.py --platform real --driver native-sim --backend opencv --fixed-camera --fixed-pitch 0 --phone

That is the brain (webcam, matcher, autopilot) commanding flightcore, which flies its own simulator.
No ArduPilot, no SITL. `NativeDriver.simulated()` moves the vehicle forward by the wall time that passed
between driver calls; startup/landing waits run faster than real time.

Simulator and log tools (no hardware, no network):

    python tools/flightcore_sim.py all                            # 7 scenarios, PASS/FAIL + numbers
    python tools/flightcore_sim.py hover --csv t.csv --log s.csv  # trace + raw sensor log
    python tools/flightcore_replay.py s.csv --out est.csv         # re-run the estimator on a log

## Tests

    python -m unittest discover -s tests -t . -p "test_flightcore_*.py"      # default set
    FLIGHTCORE_SLOW=1 python -m unittest discover -s tests -t . -p "test_flightcore_*.py"   # every scenario

Default set flies the essentials in the sim. The slow set adds battery RTL, fence, GPS glitch,
mag disturbance, gyro/accel bias steps, vibration burst, baro drift, compass yaw jump, gross
attitude corruption, backup-filter takeover, NaN emergency, and a determinism check.
No `unittest.mock`: fakes and injected clocks/sources only.

## Tuning

Everything is in `config.py`. `VehicleParams` is used twice: as the model the controller assumes and
as the true plant in the sim, so give the sim a different one to test robustness (the tests do:
+25 % mass, one weak motor). Order of work on a real vehicle:

1. Measure: mass, motor-to-motor arm, thrust vs command on a thrust stand (fills `max_thrust`,
   `thrust_expo`, `hover_frac`), motor lag (`motor_tau`). Inertia from a bifilar swing or CAD.
2. Rate loop first (`rate_kp/ki/kd`), props on, tethered, small steps. Then attitude, then velocity.
3. Calibrate the compass on the airframe; set `mag_declination` for the site.
4. Record a log every flight (`CsvRecorder`), replay after each filter change.

The hover-thrust learner adapts to mass/battery drift over `hover_lpf_tau` seconds. There is no
battery-sag compensation: a sagging pack lowers thrust per command until the learner catches up.

## Licensing (read before going commercial)

- Written from textbook math (error-state Kalman filter, geometric SO(3) control, Mahony filter,
  quad mixing). **No ArduPilot or PX4 code was copied or ported.** ArduPilot is GPL-3: copying even
  small parts would pull the whole product under GPL-3. Keep it that way: do not paste code from it,
  and do not translate its functions line by line. PX4's flight stack is BSD-3, which is permissive,
  but the same "write it yourself" habit keeps this clean.
- Only third-party dependency here is numpy (BSD-3).
- The licence is not the hard part of selling a flight controller: product liability, and rules for
  aircraft that fly near people (this is a person-following drone), are. Get advice before you sell.
  This is a note, not legal advice.

## Real-time and hardware path

The loop is 500 Hz (`FlightConfig.loop_hz`); the sim costs about 0.8 ms per tick on a laptop core, so
the compute fits, but Python on Linux has scheduling jitter of milliseconds and pauses (GC, other
processes). That is fine for the estimator and position loops, not great for the rate loop. Options:

1. Companion computer runs all of it at 100-250 Hz (set `loop_hz`, retune the rate loop). Simplest.
2. Recommended for a real product: an MCU (STM32/RP2040 class) runs IMU -> attitude -> rate -> mixer -> ESC
   in C at 1 kHz+, the companion runs the estimator, supervisor and position loops and sends attitude/thrust
   setpoints. The math in `control/attitude.py`, `control/rate.py`, `control/mixer.py` is short and
   ports directly; keep the Python as the reference and test the C port against it for equal outputs.

Still needed for a real vehicle: IMU/baro/mag/GPS drivers that produce `SensorFrame`s (with correct
measurement timestamps, GPS origin = home), an ESC output driver with its own hardware watchdog (a dead
Python process must not leave the last motor command applied), a `NativeDriver.open(...)` that wraps a
`ThreadRunner(FlightLoop(...))` around those (the adapter and runner exist; the drivers do not), and
bench identification of the vehicle model.

## Known limits

- No optical flow: indoors or under GPS loss the estimator falls back to drag-model velocity and holds
  poorly; it flags this and degrades to velocity/level hold, then lands.
- Rangefinder over a person or other unmodelled surface is handled by gating and holdoff only.
- Compass yaw reset after more than 5 s of disturbance can adopt a bad heading if the field norm looks plausible.
- Gross attitude corruption (40-60 deg) is survivable in the sim but violent.
- Sim-only tuning; no wind-tunnel or flight data behind any gain.
- Python loop is not hard real-time (see above).
