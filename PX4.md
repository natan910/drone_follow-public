# PX4 support

`main.py` flies ArduPilot or PX4. Same brain, same commands; only the driver differs.

    --driver mavlink --autopilot auto        # default: read the FC's heartbeat, pick the driver
    --driver mavlink --autopilot ardupilot   # MavlinkDriver, GUIDED; refuses to start on PX4
    --driver mavlink --autopilot px4         # Px4Driver, OFFBOARD; refuses to start on ArduPilot

Files: `control/px4_driver.py` (driver), `control/flight_controller.py` (picks the driver),
`tools/px4_hover_test.py` (first test), `tests/test_px4_driver.py` (39 tests, fakes).

## How PX4 differs (what the driver handles)

| | ArduPilot (`MavlinkDriver`) | PX4 (`Px4Driver`) |
|---|---|---|
| our mode | GUIDED | OFFBOARD |
| mode switch | `set_mode("GUIDED")` | `DO_SET_MODE` main/sub (`custom_mode = main<<16 \| sub<<24`) |
| command stream | whenever the brain sends | must be > 2 Hz or PX4 leaves OFFBOARD: thread resends at 20 Hz |
| program stalls | GUID_TIMEOUT, then hold | after 1 s without `send()`: thread streams zero velocity (hold) |
| program dies | GUID_TIMEOUT, then hold | stream stops, PX4 offboard-loss failsafe after `COM_OF_LOSS_T` (1 s): `COM_OBL_RC_ACT` |
| velocity frame | `BODY_OFFSET_NED` (9) | `BODY_NED` (8). PX4 rejects 9 |
| takeoff altitude (param7) | above home | above mean sea level: driver adds ground AMSL |
| arming mode | GUIDED | Hold (AUTO.LOITER) |
| land / return | LAND / RTL | AUTO.LAND / AUTO.RTL, resent until PX4 confirms |
| pose origin | EKF origin | where it armed (PX4's EKF origin can be metres away) |

Not used on purpose: PX4's `HOME_POSITION` and `GLOBAL_POSITION_INT.relative_alt` for height.
PX4 moves home's altitude to correct baro drift; in SITL that made the height read +1.0 m on the ground.

## PX4 SITL (headless, no Gazebo)

Clone as a sibling, never inside drone_follow (same reason as ArduPilot):

    git clone --depth 1 --branch v1.17.0 https://github.com/PX4/PX4-Autopilot.git
    cd PX4-Autopilot
    python3.11 -m venv .venv-px4 && source .venv-px4/bin/activate
    pip install cmake ninja "empy>=3.3,<4" kconfiglib jinja2 pyyaml jsonschema pyros-genmsg packaging toml future numpy lxml
    ulimit -S -n 2048                  # macOS: PX4 needs more open files than the default 256
    make px4_sitl sihsim_quadx         # first build 5-15 min; then the pxh> shell

No Homebrew needed (cmake + ninja come from pip, the compiler from Xcode Command Line Tools).
`make` fetches the submodules it needs. Stop PX4 with `shutdown` (or Ctrl+C) in `pxh>`.

Then, in drone_follow (`.venv` active, pymavlink installed):

    python tools/px4_hover_test.py --mavlink udpin:127.0.0.1:14540 --signs
    python -u main.py --platform real --driver mavlink --autopilot px4 --mavlink udpin:127.0.0.1:14540 \
        --backend opencv --camera webcam --fixed-camera --fixed-pitch 0 --phone --log run_px4.jsonl

Ports: PX4 SITL sends to 14540 (onboard/API link: us) and 14550 (ground station: QGroundControl).
Monitor and preflight work unchanged against PX4:

    python tools/mav_watch.py --csv ~/px4_run1.csv     # listens on 14550: close QGroundControl first
    python tools/preflight_check.py --driver mavlink --autopilot px4 --mavlink udpin:127.0.0.1:14540 --skip-camera

`mav_watch` shows PX4 mode names (LOITER = Hold, TAKEOFF, OFFBOARD, LAND, RTL) and, on PX4 only, takes
`alt` from LOCAL_POSITION_NED relative to where it armed (PX4's relative_alt jumps). `preflight_check
--autopilot px4|ardupilot` fails if the flight controller runs the other firmware.

SITL notes:
- SIH battery drains in about a minute and stops at 50 % (`SIM_BAT_MIN_PCT`). Our 30 %/15 % battery
  logic never fires unless `param set SIM_BAT_MIN_PCT 10` in `pxh>`.
- SIH height estimate wanders ~0.3-0.5 m and landing detection can take ~30 s. Sim artefact.
- After an auto-disarm PX4 reports OFFBOARD again. `autonomy_permitted()` also requires armed, so
  nothing is sent then.
- PX4 SITL has no gimbal: tilt commands get refused 3 times, then the driver stops sending (same as ArduPilot).

## Real PX4 flight controller (bench, props OFF first)

Set in QGroundControl:
- Companion link: `MAV_1_CONFIG` = TELEM2, `MAV_1_MODE` = Onboard, `SER_TEL2_BAUD` = 921600.
  `main.py --mavlink /dev/serial0 --baud 921600`.
- Offboard loss: `COM_OF_LOSS_T` 1.0 (default), `COM_OBL_RC_ACT` = 5 (Hold). Same as ArduPilot's
  GUID_TIMEOUT behaviour. Default 0 (Position mode) needs an RC transmitter; without one PX4 falls back to RTL.
- PX4's own fence as a backstop (wider than ours): `GF_MAX_HOR_DIST` 40, `GF_MAX_VER_DIST` 15, `GF_ACTION` 2 (Hold) or 5 (Land).
- `RTL_RETURN_ALT`: default 60 m. RTL (a failsafe, or `return_to_launch()`) climbs to it first. Set it low (e.g. 10) for a camera drone near people.
- Test the pilot takeover: flip the RC mode switch out of OFFBOARD mid-flight; the brain must go silent.

## Licence

PX4 is BSD-3 (permissive). The driver contains no PX4 code: it speaks MAVLink (pymavlink, LGPL-3,
already a dependency). Nothing here changes `flightcore/`.
