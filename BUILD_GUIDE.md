# Building the real drone: parts list and step-by-step

This is a from-parts build (no DJI/prebuilt), aimed at "cheap and functional
enough to test `drone_follow` for real." Budget roughly $450-650 all-in if you
buy nothing you don't need. It assumes zero prior drone-building experience,
but does assume basic soldering.

**Read this whole section before ordering anything** — a few parts have to
match each other (frame size ↔ prop size ↔ motor kv ↔ battery).

---

## 1. Parts list

### Airframe & propulsion (~$110-150)
| Part | Why this one | Approx. price |
|---|---|---|
| F450 (or F550 if you want more lift for the Pi + gimbal) quad frame kit | Cheap, huge community, plenty of room underneath for a downward camera/gimbal | $20-30 |
| 4x 920KV brushless motors (e.g. "A2212 1000KV" clones or better, a matched 4-pack) | 920-1000KV suits a 10" prop on 3S, good thrust margin for extra payload | $30-40 for 4 |
| 4x 30A ESCs (SimonK or BLHeli firmware), or a 4-in-1 30A ESC board | 4-in-1 is neater to wire | $20-30 |
| 2x pairs 10x4.5" props (CW+CCW) + 1 spare pair | Match the frame's motor spacing | $10 |
| 3S 5200mAh LiPo battery (XT60 connector) + a second one | ~12-15 min flight time each; you'll want two so one charges while you fly | $40-60 for two |
| LiPo balance charger | Never charge a LiPo without one | $25-35 |

### Flight controller (~$50-60)
| Part | Why | Price |
|---|---|---|
| **Holybro Kakute H7 Mini** (or the full-size Kakute H7, or Matek H7A3-SLIM) | Runs ArduCopter, has enough UARTs for the companion computer + 2-3 range sensors + a GPS, built-in OSD not needed but harmless | $45-55 |
| M8N GPS + compass module | ArduPilot's GUIDED mode (which this project uses) wants a good position source; GPS outdoors, or see §7 for indoor alternatives | $15-20 |

### Companion computer & camera (~$90-130)
| Part | Why | Price |
|---|---|---|
| Raspberry Pi 4B (4GB) or Pi 5 (4GB) | Runs `drone_follow`'s Python. 2026 pricing on Pi boards has crept up due to LPDDR4 cost — a Pi 4B 4GB is the better value pick right now if you find one in stock; a Pi 5 is faster and fine if that's what's available | $55-80 |
| Raspberry Pi Camera Module 3 (standard, not wide) | Matches the 66° HFOV assumed in `config.py`'s `CameraConfig` — if you use a different lens, measure its FOV and update `hfov_deg`/`aspect` | $25 |
| 32GB+ microSD card | OS + code | $10 |
| USB-to-serial adapter (CP2102, if the Pi's own UART isn't used for MAVLink) or use the Pi's GPIO UART directly | Talks MAVLink to the flight controller | $0-8 |

### Range sensors (~$60-90 for 3, more if you add a downward one)
| Part | Why | Price |
|---|---|---|
| 3x TF-Luna LiDAR range sensors | `perception/range_sensors.py` already has a `TFLunaParser`/`TFLunaRing`; cheap (~$20 each), 8m range, works outdoors in sun (LiDAR, not ultrasonic) | $60 |
| 1x extra TF-Luna, pointed straight down | Used for the "hover 30cm above the target's head" height control (`RangeScan.down`) — mount it looking down, ideally with a narrow enough beam it mostly sees the person and not the floor around them | $20 |
| (Optional) 1x more TF-Luna pointed up | Only if you want the ceiling/ceiling-clearance protection in `AvoidConfig`; skip for outdoor flying | $20 |

### Gimbal (optional but recommended for the "aim the camera" behavior)
| Part | Why | Price |
|---|---|---|
| 1-axis (pitch) micro servo gimbal for a Pi Camera, or just a metal-gear micro servo (e.g. MG90S) and 3D-printed/improvised bracket | Lets the autopilot tilt the camera per `CameraConfig.gimbal=True`; without one, set `gimbal=False` and bolt the camera at a fixed downward angle (`fixed_pitch_deg`) | $10-25 |

### Misc
- XT60 pigtails, bullet connectors, heat-shrink, zip ties, a soldering iron, a multimeter.
- A way to power the Pi from the flight battery: a 5V/3A+ UBEC (not the flight controller's onboard 5V rail — that's usually not rated for a whole Raspberry Pi).

**Total: roughly $450-650** depending on what you already have and Pi
availability. This is a *test platform*, not a polished product — expect to
tune things.

---

## 2. Assemble the airframe

1. Bolt the four arms to the frame's center plates per the kit's instructions.
2. Mount a motor on each arm. **Prop rotation matters**: front-left and
   rear-right spin one way (CW), front-right and rear-left the other way
   (CCW) — the frame kit's instructions show which. Get this wrong and the
   quad will flip on spin-up.
3. Mount the 4-in-1 ESC board (or 4 individual ESCs) on the center plate,
   solder each ESC's 3 motor wires to its motor (order doesn't matter yet —
   you fix rotation direction later via ESC/motor settings, not by re-wiring).
4. Mount the flight controller on top, on vibration-dampening standoffs
   (usually included with the FC), oriented with its arrow pointing to the
   front of the frame (this matters — wrong orientation = uncontrollable).
5. Mount the GPS/compass on a mast above the frame, away from the power
   wiring (compass is very sensitive to nearby current).
6. Mount the Pi, camera (on the gimbal if you have one, facing forward and
   down), and range sensors underneath/around the frame. Downward range
   sensor needs a clear view of the ground/person, not blocked by landing
   gear.
7. Wire ESCs to the flight controller's motor outputs (M1-M4).
8. Wire GPS/compass to the FC's dedicated GPS port (usually a single JST-GH
   cable).
9. Wire a UBEC from the battery's power distribution to the Pi's 5V/GND (via
   USB or the Pi's GPIO 5V pin — GPIO is fine if you're careful about
   polarity; there's no reverse-polarity protection).
10. Wire the FC's UART (say UART1, a 4-pin TX/RX/5V/GND) to the Pi: FC-TX →
    Pi RX, FC-RX → Pi TX, GND → GND (don't connect the FC's 5V to the Pi's
    UART pins if you're powering the Pi separately — just TX/RX/GND).
11. Wire each range sensor's UART (they're 3.3V TTL, one wire pair each) to
    free UARTs on the flight controller (a Kakute H7 has several) or to the
    Pi's serial-to-USB adapters — whichever `perception/range_sensors.py`'s
    `TFLunaRing` config expects (this repo assumes they arrive as strings
    like `"/dev/ttyUSB0"`, so USB-serial adapters into the Pi are the
    simplest path if the FC runs low on spare UARTs).
12. If you have a gimbal servo: wire its signal wire to the FC's servo
    output pin normally reserved for a camera gimbal (check your FC's pinout
    for a "servo"/"aux out" pin), and see §5 for whether `set_camera_pitch`'s
    `DO_MOUNT_CONTROL` command actually reaches it — you may instead wire the
    servo straight to the Pi's own PWM/GPIO and bypass MAVLink for this one
    axis if it's simpler (would need a small code change in
    `control/mavlink_driver.py`'s `set_camera_pitch`).

---

## 3. Flash and configure ArduPilot

1. Download **Mission Planner** (Windows) or **QGroundControl** (Mac/Linux)
   on a laptop.
2. Connect the flight controller via USB, flash **ArduCopter** firmware for
   your specific board (Kakute H7 has a dedicated build on ardupilot.org's
   firmware downloads).
3. Run the mandatory setup wizard: accelerometer calibration, compass
   calibration (rotate the drone through all axes away from metal/motors),
   radio calibration (see §4), ESC calibration (throttle range).
4. Set these parameters (Mission Planner: Config → Full Parameter List):
   - `SERIALx_PROTOCOL = 1` (MAVLink2) on whichever UART goes to the Pi, and
     `SERIALx_BAUD = 921` (921600, matching `MavlinkDriver.open`'s default).
   - `ARMING_CHECK` — keep this at its default (all checks on) until you've
     flown successfully a few times; don't disable checks to "make it arm."
   - `FENCE_ENABLE = 1` and set a `FENCE_RADIUS`/`FENCE_ALT_MAX` as an
     extra hardware-level safety net *in addition to* this project's own
     software geofence (`SafetyConfig.geofence_radius_m`) — belt and braces.
   - `FLTMODE1..6` — set at least one mode to **Stabilize** or **Loiter**
     (manual pilot control) and one to **GUIDED** (what this software uses).
     You need a physical mode switch on your RC transmitter to flip between
     them in flight.

---

## 4. You still need an RC transmitter/receiver

Autonomy here means "the software flies it," but ArduPilot's whole safety
model (and this project's `autonomy_permitted()` check) assumes **a human
with a physical radio can take back control instantly** by flipping the
flight mode switch. Get a basic RC transmitter/receiver pair (e.g. a
FlySky FS-i6 + receiver, ~$40-60) bound to the flight controller, with one
channel wired as the mode switch. **Do not skip this** — it's your emergency
stop.

---

## 5. Software on the Raspberry Pi

Proven on a Pi 5 8GB + Camera Module 3 Wide, 2026-10-02 (30 brain steps/s, `--driver print`).

1. Flash Raspberry Pi OS Lite (64-bit) with Raspberry Pi Imager (Trixie-based; its system
   Python is 3.13). Set hostname `drone`, user, Wi-Fi and SSH in Imager's settings.
2. Camera: Camera Module 3 Wide on CAM1 via the 22-to-15-pin cable. Check it:
   `rpicam-hello` (logs `imx708_wide`). Serial for the flight controller:
   `sudo raspi-config` -> Interface Options -> Serial Port (login shell: no, hardware: yes).
3. `sudo apt update && sudo apt install -y git python3-picamera2`
4. `git clone https://github.com/<you>/drone_follow && cd drone_follow && ./setup.sh --full --models`.
   On the Pi `setup.sh` sees picamera2 in the system Python and builds `.venv` from it with
   `--system-site-packages`. Why: picamera2 exists only as that apt package, compiled for the
   system Python; a venv from pyenv or another Python cannot import it, so `--camera pi` fails.
   By hand, the same thing is `/usr/bin/python3 -m venv --system-site-packages .venv`.
5. A dry run, **no flight controller, props off**:
   ```
   python main.py --platform real --driver print --auto-launch --camera pi --backend opencv \
     --fixed-camera --fixed-pitch 30 --no-geofence --no-window --log pi.jsonl
   python tools/flight_summary.py pi.jsonl      # steps/s well above 5 = the Pi keeps up
   ```
   `--no-window`: there is no screen over SSH. `--no-geofence` and `--auto-launch`: dry runs only.
6. Bench, flight controller wired (TELEM2 -> Pi UART), **props off**:
   `python tools/preflight_check.py --driver mavlink --mavlink /dev/serial0 --camera pi`, then
   `python tools/mav_watch.py --mavlink /dev/serial0` while you move the drone by hand.
7. First real flight, **outdoors, open field, you at the transmitter**:
   ```
   python main.py --platform real --camera pi --driver mavlink --mavlink /dev/serial0 \
       --backend opencv --fixed-camera --fixed-pitch 30 --phone --log flight.jsonl
   ```
   It waits on the ground until you tap **Launch** on the phone page (never use
   `--auto-launch` on a real drone), then arms and takes off. Hold the transmitter, mode switch
   on Loiter, ready to flip it if anything looks wrong.
8. Set the program to launch automatically at boot (a `systemd` service unit
   running `main.py` with your chosen flags) once you trust it, so you don't
   need a laptop plugged into the Pi in the field — just power it on and
   connect to its phone page.

---

## 6. Calibrate `CameraConfig.face_height_m`

The whole distance/height estimate in `perception/geometry.py` depends on
one number: the real height of a face bounding box. Point the camera at
someone standing at a known distance, read the reported `size` (fraction of
frame height), and solve `face_height_m = size * distance_m * 2 * tan(vfov/2)`
— or just fly it and empirically nudge `face_height_m` up/down until the
reported hover height (`down` sensor reading vs. what you see) matches
reality.

---

## 7. If you don't have reliable GPS (flying indoors)

ArduPilot's GUIDED velocity control (what `MavlinkDriver` sends) technically
needs *some* position/velocity source for the EKF to stay happy, even though
this project only ever commands body-frame velocities, never GPS
waypoints. Indoors, options in rough order of cost:
- Optical flow sensor + downward rangefinder (e.g. a PMW3901-based flow
  sensor, ~$25) — ArduPilot fuses this instead of GPS for position.
- A motion-capture or UWB indoor positioning system (Vicon, or cheaper
  UWB tag systems) — expensive, only worth it for serious indoor work.
- Simplest: fly outdoors first. Get the whole pipeline validated with GPS
  before investing in an indoor positioning solution.
