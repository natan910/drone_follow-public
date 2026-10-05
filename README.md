# drone_follow

An autonomous drone brain: patrols and maps an area on its own, avoids obstacles,
finds and follows one specific person, and takes that person's photo from your
phone while it is flying.

```
              +-------------------- Autopilot ---------------------+
 camera  -->  | tracker --> map --> safety --> behaviour --> avoid  |  --> driver
 (face, else body re-ID)
 range   -->  |                                       --> shape     |      (sim / print / MAVLink)
 pose    -->  +-----------------------------------------------------+
 phone photo --> EnrollmentBox --> matcher (between frames)
```

## Behaviour (highest priority first)
| Mode | When |
|---|---|
| IDLE | the pilot flipped the mode switch: nothing is sent at all |
| LAND / RETURN / HOLD | safety: critical battery or lost range sensors / low battery, geofence, time limit, lost camera / sensor stall |
| TRACK | the enrolled person is visible (face, or body when the face isn't): fly to and hover `hover_height_above_target_m` above them, following as they walk |
| LOST | they just vanished: keep going the way they were walking, turning to face it (or turn toward the side they left on) |
| SEARCH | fly to where they were last seen, then spin to scan |
| EXPLORE / PATROL | otherwise: go to whichever reachable place the camera has gone longest without looking at (never-seen space first, then oldest-seen, forever) |

Obstacle avoidance and speed/acceleration limits apply to every mode.

## Run
    ./setup.sh                                                  # or: make setup  -- venv + deps + tests
    python main.py --platform sim --show --seconds 300        # toy world, no hardware
    python main.py --platform real --phone                     # laptop webcam dry run + phone page
    python -m unittest discover -s tests -t .                  # ~825 tests, ~3 min (or: pytest)
    python tools/bench_perception.py                           # per-stage timings on this machine

On the drone: see the flags at the top of `main.py` (`--camera pi --driver mavlink --ranger ...`).
First flights: `python tools/preflight_check.py ...` (same flags as main.py) to catch a broken
setup on the ground, `tools/hover_test.py`, then `tools/check_ranges.py` on the bench.
Once per camera, on the ground: send your photo, then tap **Calibrate** on the phone page or
fleet console and stand 2 m from the camera until it says done. It measures the lens's real
field of view (the default is a Pi Camera Module 3's) and saves it to `camera_calibration.json`,
which every later start loads automatically.

## Simulated flight (ArduPilot SITL)
Before real hardware, fly the real code against ArduPilot's simulator. ArduPilot lives in
its own folder **next to** this repo, never inside it, with its own venv. From the parent
directory, clone it with `git clone` then `git submodule update --init --recursive` (rerun the
second command if GitHub times out). One-time setup, from inside `ardupilot/`:

    python3.11 -m venv .venv-ap && source .venv-ap/bin/activate
    pip install "empy==3.3.4" pyserial pexpect future lxml MAVProxy gnureadline
    ./waf configure --board sitl && ./waf copter

Every time, two terminals:

    # terminal 1, in ardupilot/ with .venv-ap active
    Tools/autotest/sim_vehicle.py -v ArduCopter --out=udp:127.0.0.1:14551
    # terminal 2, in drone_follow/ with .venv active
    python tools/hover_test.py --mavlink udpin:127.0.0.1:14551

Wait for `EKF3 IMU0 is using GPS` in terminal 1 before arming. Add `-w` to `sim_vehicle.py`
only to reset the simulated drone's parameters. `--map`/`--console` need wxPython (not
installed). Take control back any time at the MAVProxy prompt: `mode loiter` or `mode land`.

## Automation
| Need | Use |
|---|---|
| One-command setup | `./setup.sh` (`--full` for insightface/onnxruntime/pymavlink, `--models` to download the body re-ID / face ONNX files) or `make setup` / `make setup-full` |
| Reproducible environment | `Dockerfile` (`docker build -t drone_follow .`, `--build-arg EXTRAS=full`) and `.devcontainer/` for VS Code |
| CI | `.github/workflows/tests.yml` runs the full suite on every push/PR, Python 3.11, 3.12 and 3.13 |
| Ground checks before a flight | `python tools/preflight_check.py` -- model files, camera, range sensors, MAVLink link, all independently, one report |
| Auto-start on the Pi | `deploy/drone-follow.service` (systemd; runs preflight first, restarts on crash, never respawns into a crash loop) |
| Camera field of view | one tap on **Calibrate** (phone page or console); `tools/calibrate_camera.py` is the manual bench alternative |
| Watching and tasking every drone | `python -m fleet.server` (ops console: tactical map, tasking, target photos, calibration, operator log), plus each drone's `main.py --fleet-url ... --drone-id ... --fleet-token ...`. Drones pull their tasking in the reply to their own reports, so it works behind NAT/LTE. `python tools/fleet_demo.py` shows it with simulated drones |

`make help` lists the shortcuts for all of the above.

## Sending a photo while it flies
`--phone` prints a URL like `http://<drone-host>:8080/?token=<generated-token>`. Open it on your
phone (same Wi-Fi), pick or take a photo, tap Send. Replies: `200` enrolled, `422` no
face in the photo, `401` wrong token. "Forget target" sends the drone back to patrolling.
Plain HTTP plus a token: use a private network you control, never the open internet.
Only the face embedding is kept, in memory. Photos are never written to disk.

## Recognising them from behind (body re-ID)
The face is only visible from the front and not from above, so on its own the
drone loses you the moment you turn around or it hovers over you. `perception/
target_finder.py` adds a second stage: while your face is visible it studies
your body (clothes, build) every few frames; when the face disappears it finds
the people in view and picks you by that look.

- **Only face-confirmed frames teach it.** Body-only matches go into a short
  memory that expires after 20 s, and the tracker stops trusting body-only
  tracking 45 s after the last face (`track_only_timeout_s`). It cannot slowly
  drift onto someone else.
- **It never guesses.** Two similar-looking people and no recent track: it
  reports nobody. A lookalike far from where you were predicted to be is
  ignored.
- **A full-length photo** from the phone teaches your body immediately, so it
  can pick you up from behind before it has ever seen your face live. The
  phone reply says `"body_looks": 1` when that happened.
- **Cheap:** it searches a crop around where you are predicted to be first
  (HOG: 44 ms full frame vs 7 ms crop on a laptop core), embeds at most 4
  people a frame, and does no body work at all while the face is visible
  except every 5th frame.

| `--reid` | what it compares | download |
|---|---|---|
| `color` (default) | clothing colours, torso and legs | nothing |
| `onnx` | a re-ID network: [OpenCV Zoo YouTu re-ID](https://github.com/opencv/opencv_zoo/tree/main/models/person_reid_youtureid), an OSNet export from torchreid, or your own distilled model (128x256 RGB input, ImageNet normalisation) | `models/person_reid_youtu_2021nov.onnx` |
| `fused` | 70 % network + 30 % colour | same |

| `--person-detector` | notes |
|---|---|
| `hog` | built into OpenCV, no download; weak on close, partial, turned-around or top-down people |
| `yolo` (default) | much better. Default model: OpenCV Zoo's YOLOX (Apache-2.0, 640 px) at `models/person_yolo.onnx` (`./setup.sh --models`). YOLOv8/YOLO11 exports also work (`--yolo-format v8`; AGPL-3.0) |
| `own` | our own detector, trained on our own footage (`--own-model models/person_own.onnx`). See [TRAINING.md](TRAINING.md) |

Our own re-ID net is an `onnx` model: `--reid fused --reid-model models/reid_own.onnx`.
Record, label, train and benchmark both: [TRAINING.md](TRAINING.md).

Tune thresholds in `config.py` → `BodyReIDConfig`; `--phone`'s `/status`
shows the live similarity (`reid.best_sim`) so you can see how close calls are.

## Layout
| Path | Job |
|---|---|
| `datatypes.py` `config.py` | shared types; every tunable number |
| `perception/` | face matcher (InsightFace or OpenCV backend), body re-ID (person detector, appearance embedders, TargetFinder), camera geometry, one-tap optics calibration, cameras, TF-Luna range sensors |
| `tracking/` | SEARCHING / TRACKING / LOST state machine |
| `mapping/` | occupancy grid + "last viewed" clock, save/load |
| `navigation/` | Dijkstra planner, patrol/explore goal picker, waypoint follower, obstacle avoider |
| `safety/` | supervisor: battery, geofence, watchdogs, latched RETURN/LAND |
| `control/` | driver interface; print, MAVLink (ArduPilot) drivers; follow controller; command shaper |
| `autonomy/` | `Autopilot` (the brain) and the run loop |
| `platforms/` `sim/` | where observations come from: toy world, or real sensors |
| `comms/` | the drone's own phone page (`phone_page.html`) and its server; shared command validation |
| `telemetry/` | JSON-lines flight log |
| `fleet/` | ops console (`server.py` + `console.html`) and the drone-side link that reports to it and collects tasking (`reporter.py`) |
| `dataset/` | our own training data: recorder, teacher auto-labels, review tool, splits, export |
| `training/` | PyTorch training of our own detector and re-ID net, ONNX export (training machine only) |
| `evaluation/` | benchmark on the frozen eval split: AP, recall by size/altitude, re-ID rank-1/mAP |
| `tools/` | benchmarking, range-sensor check, hover test, preflight check, manual camera calibration, fleet demo, `data.py` (record/label/export), `benchmark.py` (score + gate models) |
| `deploy/` | systemd unit for auto-starting on the Pi |

## Know the limits
- Tested in the toy world and against fakes. The MAVLink driver, TF-Luna reader and Pi
  camera code have never touched hardware. Verify each on the bench, then in ArduPilot
  SITL, before a real flight.
- Obstacles are sensed in 2-D (a horizontal ring plus a down/up sensor). Anything
  between those beams is invisible.
- Hovering 30 cm over someone with one forward-tilting camera leaves little margin:
  when they step off from directly underneath, the lock blinks for a second or so
  before the drone catches up. In the toy world it keeps up with walkers to ~0.8 m/s
  and trails a brisk 1.2 m/s walk by 2-3 m.
- Body re-ID with `color` is fooled by two people dressed alike; with `onnx` it is
  much better but still not a face. The 45 s identity window is the backstop.
- Single-point range sensors have a narrow beam and miss thin things (poles, chair legs).
  Treat autonomy as pilot-supervised: keep the RC transmitter in hand.
- Outdoors the pose is GPS-grade (metres), so the map is coarse. Resolution is 0.5 m.
- Copter velocity control needs a position source: GPS outdoors, or optical flow + a
  downward rangefinder indoors.
