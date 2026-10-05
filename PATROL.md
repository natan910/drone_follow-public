# Perimeter watch: how it works, how to run it (2026-09-27 evening)

`--mission perimeter` + task PATROL. The drone patrols, tracks strangers, goes to look (R1), answers
fixed sensors (R4), notices objects that appeared or went missing (R7), pushes alerts to a phone.
Detection runs off the flight loop (R9). Every event is logged against the entities involved (R6 ontology).
Status: unit + toy-world tested (sandbox + your Mac suite). Not flown. Hardware not here yet.

---

## 1. Run it

    # toy world, no hardware (this is the sim to use for patrol; `main.py --platform sim` is the FOLLOW demo)
    python tools/patrol_sim.py --show                  # q quits. ~20 min of sim time, faster than real time
    python tools/patrol_sim.py --trigger-at 60         # + a door sensor fires in the "hall" at 60 s

    # phone pushes, no drone
    python tools/ntfy_test.py --url https://ntfy.sh/<your-topic>

    # indoor, on the drone (tap Patrol, then Launch)
    python main.py --platform real --driver mavlink --camera pi --fixed-camera --fixed-pitch 30 --phone \
        --token "$DRONE_TOKEN" --mission perimeter --indoor --zones zones.json --investigate --baseline baseline.json \
        --alert-url "$NTFY_URL" [--responder-port 8091 --phone-url http://drone.local:8080]

    # outdoor: same without --indoor, zones.json with a "property"

**Tap Patrol, not Follow.** PATROL ignores the enrolled person. FOLLOW would chase that person instead
of watching. Their photo is still useful: it keeps the enrolled person from being tracked as a stranger.

### What you see in `patrol_sim.py`
Grey = walls/obstacles. Grey outlines = zones (red while occupied). Triangle = drone, lines = its
camera's view. Blue dot = the intruder (truth). Orange ring + "T1" = the drone's track of them.
Magenta X = the spot being investigated. Green squares = objects (grey once removed).
Default timeline: patrol from 0 s | 150 s intruder enters from the east, walks into the study,
stays 40 s, leaves | 600 s the laptop in the study disappears. Expected: `stranger T1` investigated,
an `intrusion` alert in the study, one `laptop missing` alert about 1-4 min after 600 s, no collisions.

## 2. One step (main loop, ~10 Hz)

    camera + rangers + FC  -> Observation
        -> Autopilot.step -> Decision -> driver                     FLIGHT (pure, no I/O)
        -> mission.step(frame, obs, decision, board)                WATCHING (never steers directly)
             source.step        frame -> Scene now and then (R9: detector thread; sim: ground truth)
             tracker.update     ground points -> Tracks, zone events, intrusion alerts (R6)
             baseline.observe   objects per viewpoint -> appeared / missing alerts (R7)
             _direct            decides where to look -> autopilot.investigate(x, y) (R1, R4)
             alerts             snapshot, events.jsonl, status, push (missions/notify.py)

A crash inside the mission switches the mission off; flight continues. An investigation already
running still ends at its deadline.

### Flight side (autonomy/autopilot.py)
Priority: pilot (IDLE) > safety (LAND / RETURN / HOLD) > operator task > TRACK (FOLLOW only) >
**INVESTIGATE** > SEARCH / EXPLORE / PATROL. Every command: obstacle avoider, then the altitude limit
(the lower of the 15 m ceiling and the **headroom** limit, below), then the command shaper.

Patrol ("go where you looked longest ago"): map 60 x 60 m around launch, 0.5 m cells, each with
occupied/free evidence (rangers) and last-viewed time (camera wedge, `view_range_m` deep). Every
1 s: score = staleness (max 900 s) - 5 s per metre of path; never-seen first (EXPLORE), then
oldest-seen forever (PATROL). 30 s hysteresis, stuck guard (8 s no progress -> blacklist 60 s).

### R1 investigate (autonomy/investigate.py, Autopilot.investigate, PerimeterMission._direct)
- Who: with `--investigate`, the mission watches (a) a person in a zone alert, else (b) any confirmed
  stranger just seen, else (c) the zone of a sensor trigger (R4). A stranger replaces a sensor check;
  a zone alert replaces a plain stranger.
- How: fly (planner route, avoider on) to the standoff point: on the line spot->drone,
  `investigate_standoff_m` from the spot (8 m outdoors, 2.5 m `--indoor`). Never overhead. Then hold
  there facing the spot (yaw servo). The spot follows the track as they walk.
- Until: 60 s, extended to 20 s after the last sighting while they stay in view; the track ending
  (30 s unseen); the operator choosing HOVER / HOLD / RETURN / LAND; pilot takeover; battery/geofence
  (safety always wins). Goals are clamped 2 m inside the geofence. Then back to patrol.
- A saved photo every 5 s while investigating (not pushed: the first alert already was).
- Toy world: standoff reached within 0.1 m, facing within 3 deg, back to EXPLORE after the deadline.

### R4 responder (missions/responder.py)
A PIR / door contact / camera calls `GET http://<drone>:8091/trigger?token=T&zone=hall&source=pir-1`.
- Drone on the ground: push "pir-1 triggered at hall" with **[Launch + check]** and **[Open drone page]**
  buttons. Tapping Launch makes *your phone* POST `{"launch": true}` to the drone's phone page (same
  Wi-Fi). The drone never launches itself; the tap is the human decision.
- In the air: push without buttons; the drone investigates the zone's centre within a second.
- A trigger waits 180 s for the drone to be airborne; one sensor re-triggers at most every 30 s.
- Needs `--token` (fixed) and ideally `--phone-url http://drone.local:8080`.
- Example: `curl -H "X-Token: $DRONE_TOKEN" "http://drone.local:8091/trigger?zone=hall&source=test"`; ESPHome
  `http_request.get`; Shelly "actions -> URL"; Home Assistant `rest_command`.

### R6 tracks + ontology (missions/tracks.py, missions/entities.py)
- Ground points -> anonymous Tracks ("T3"): predicted with their velocity, linked within a gate
  (2 m + 2 m/s unseen, max 8 m), closest pairs first. Confirmed after 2 sightings within 5 s
  (a single false detection is dropped silently). Ended after 30 s unseen: `track.end` with entry,
  exit, metres walked, zones visited, duration.
- Alerts are track events: a confirmed track inside a zone for `confirm_s` (1.5 s) alerts once per
  track per zone; any zone at most every 15 s. A second person alerts too (before: one alert per zone
  per minute, whoever it was).
- Entities: Sortie, Zone, Place (viewpoint), Track, Object, Sensor, Alert. Every event in
  `<alerts-dir>/events.jsonl` names them: `sortie.start/end`, `track.new/confirmed/zone/end`,
  `alert`, `change`, `trigger`, `investigate.start/end`. Anonymous by design: no names, the owner is
  skipped, not tracked. Push-button tokens never reach disk.

### R7 learn normal (missions/baseline.py, `--baseline baseline.json`)
- A Place = where the drone was (2 m cell, 1 m `--indoor`) x which way it faced (8 sectors) x camera
  tilt. Per Place: how often each object label is in view (chair, couch, tv, laptop, backpack,
  handbag, suitcase, bicycle, car, motorcycle, truck, potted plant, bench; no animals, no people).
- One pass through a Place = one visit; "there" = in at least half its frames (a person walking in
  front for one frame changes nothing).
- After 5 visits: `missing` = usually there (>= 80 % of visits), absent 2 visits in a row;
  `appeared` = almost never there (<= 10 %), present 2 visits in a row. While a change is being
  confirmed it is not learned; once reported it is learned quietly as the new normal (a new sofa
  stops being news). One report per label per 5 min across all Places, per Place per 10 min.
- Needs `--person-detector yolo` (a COCO model). One network pass gives people + objects (R9).
- Saved on exit, loaded at start. Launch from the same spot (like `--map`).
- Toy world: laptop removed at 600 s -> one `laptop missing` ~250 s later, no false alarms over 20 min.

### R9 detection off the loop (missions/detect.py)
- The detector runs in its own thread. The main loop hands it a frame copy + the pose and camera tilt
  at that instant, picks the result up later, never waits; if still busy, the frame is skipped
  (`detector.skipped` in status). People are placed with the pose AT CAPTURE (the drone moved meanwhile).
- One YOLOX pass for people and objects; NMS per class (a person on a chair keeps both boxes).
- `detector.avg_ms` in status. `--detect-sync` puts it back on the loop (debugging only).
- Next steps (not built): our own 320 px detector (TRAINING.md) ~4x fewer pixels; a Hailo AI HAT+ on
  the Pi 5 (hardware NPU) for every-frame detection.

### Headroom: lower ceilings (mapping/headroom.py)
- With an **upward** range sensor, every reading stores "ceiling here = height + reading" in a
  0.5 m map. The altitude limit near the drone (1 m around) = lowest remembered ceiling - 0.5 m.
  Patrol descends to stay under it, this pass and every later one in the same flight.
- Limits: one TF-Luna looking up sees a 2 deg spot over the drone's centre; the props reach ~0.4 m
  further. The first crossing under a lower section is reactive (toy world: 0.1 m clearance at a
  2.1 m beam when patrolling at 2.0 m; 0.5 m after that). Patrol well below the lowest ceiling.
- Needs hardware: a third TF-Luna pointing up, and `RangeScan.up` filled by `perception/range_sensors.py`
  (upload that file + `navigation/avoidance.py` so I can check how "up" is configured).

## 3. Numbers (missions/mission_config.py)

| What | Outdoor | `--indoor` |
|---|---|---|
| Patrol height | 7 m | 2 m (doors ~2.0-2.1 m: use `--patrol-altitude 1.5` if it flies through doors) |
| View range (patrol "seen") | 15 m | 8 m |
| Cruise | 1.0 m/s | 0.5 m/s |
| Investigate standoff | 8 m | 2.5 m |
| Privacy mask on snapshots | on | off (your home; it would black out the walls) |
| Baseline viewpoint cell | 2 m | 1 m |
| Detection | start one every 1 s (thread) | same |
| Zone alert | confirmed track inside 1.5 s | same |

## 4. zones.json

    {"home": [0.00000, 0.00000],                          <- only if you use latlon
     "property": {"latlon": [[..], [..], [..], [..]]},     <- outdoors: your land
     "areas": [{"name": "study", "polygon": [[-9, 1], [-4, 1], [-4, 6], [-9, 6]]},
               {"name": "hall",  "polygon": [[4, -6], [9, -6], [9, -1], [4, -1]]}]}

Metres east / north of the **launch point** (indoors: measure with a tape from where the drone takes
off; north = where the drone's compass says). Launch from the same spot every time. Zone names are
what the sensors send (`zone=hall`).

## 5. Push alerts (ntfy)

1. Phone: install **ntfy** (App Store / Play Store). Tap + and subscribe to a unique, unpredictable
   topic that you create for yourself. A public topic name is its only lock on the public server.
2. Set `DRONE_TOKEN` to a long random secret and set `NTFY_URL` to
   `https://ntfy.sh/<your-unique-topic>`. Test the alert topic with
   `curl -d "hello" "$NTFY_URL"` -> the phone buzzes.
3. Run `python tools/ntfy_test.py --url "$NTFY_URL"` -> 2 pushes: a person
   alert with a test photo, a "laptop missing" (quieter). Add
   `--launch-button --phone-url http://<drone-host>:8080 --phone-token "$DRONE_TOKEN"` for the button push
   (tap it only with main.py running as a dry run: `--driver print`).
4. Drone: `--alert-url "$NTFY_URL"`. The drone needs internet access.
5. Later, private: run your own ntfy server (open source; one binary or Docker, on a small VPS or a
   home box), create a user + access token (`ntfy token add`), use `--alert-url https://<yours>/<topic>
   --alert-token tk_...`. iPhone: a self-hosted server needs `upstream-base-url: "https://ntfy.sh"` in its
   config for instant pushes. The public ntfy.sh server sees every photo and the Launch button's token.

## 6. Phone page, console, main.py (applied in the repo 2026-10-03)

Nothing to do by hand any more: `main.py` calls `mission.bind(autopilot)` in `run_real`; the phone page
shows the Perimeter panel (tracks, LOOKING AT, last alert, baseline, detector ms, push count) and
remembers the token; the console logs alerts and marks a drone INVESTIGATE / occupied as "alert".

Use the same long random `DRONE_TOKEN` for the phone page and sensor triggers; bookmark `http://drone.local:8080/` (Pi hostname `drone`;
Mac dry run: `http://<mac-name>.local:8080/`). The first visit asks for the token once and the phone
remembers it. A wrong/old token: open `.../?token=<new>` once.
JS syntax-checked with node; not yet run in a browser.

## 7. Flying indoors: read before the first indoor flight

- **No GPS indoors.** ArduPilot GUIDED velocity control needs a position estimate. Indoors that means
  an **optical-flow sensor + downward rangefinder wired to the flight controller** (EKF3 flow), e.g. a
  combined flow + lidar module on a Pixhawk UART. The TF-Lunas on the Pi feed our brain, not the FC's
  EKF. Without flow the drone drifts and GUIDED refuses or wanders. Needs an EKF origin set (the driver
  will need to send one: not built yet).
- **Compass indoors** is unreliable (steel, wiring): expect yaw drift; flow + a good yaw source matter.
- **X500 indoors**: 61 cm frame, 10" props, ~1.5 kg. Our planner keeps 0.35 m radius + 0.4 m margin,
  so it will not route through a normal 80 cm door (it needs ~1.5 m). Fine for one big room; for a
  house, a smaller frame (see chat: 3-3.5" ArduPilot whoop, remote brain).
- **Legal**: EU drone rules do not cover flights inside a building (your home, with consent of the
  people in it). Outdoors, patrol over a house in a town is not open category A3 (see earlier notes).
- **Ceiling**: add the upward sensor (section 2, Headroom) before patrolling at 2 m.

## 8. Remaining roadmap (not built)

- R2 property geofence (supervisor keep-in polygon + FC polygon fence) · R3 night (NoIR + IR light /
  thermal) · R5 household gallery (several known people, pets suppress alarms) · R8 projection with FC
  attitude + terrain · R10 recorder privacy mask · R11 health pushes (sortie start, battery return, link
  lost) · Dock with auto-charging (the real answer to 10-15 min batteries) · EKF origin + flow setup
  for indoor flight (section 7).

## 9. Check on the Mac (or the Pi)

    git pull
    python -m pytest -q --tb=short          # 824 passed, 20 skipped in the sandbox (Python 3.11), 2026-10-03
    python tools/patrol_sim.py --show       # watch one run; the last line should read about:
    # Done at t=1200 s: alerts {'intrusion': 1, 'object_missing': 1}, investigations started 1, collisions 0, events 11

## 10. Files added or changed by perimeter v2 (commit of 2026-10-03)

| File | What |
|---|---|
| `datatypes.py` | `Mode.INVESTIGATE` (one line) |
| `autonomy/autopilot.py` | `investigate()` / `stop_investigating()`, INVESTIGATE behaviour, headroom altitude limit |
| `autonomy/investigate.py` | NEW: standoff geometry |
| `mapping/headroom.py` | NEW: lowest-ceiling map |
| `missions/entities.py` | NEW: ontology records + event log |
| `missions/tracks.py` | NEW: R6 tracker + zone events |
| `missions/baseline.py` | NEW: R7 learn normal |
| `missions/detect.py` | NEW: R9 detector thread, COCO one-pass detector, frame -> Scene |
| `missions/responder.py` | NEW: R4 sensor server |
| `missions/simsource.py` | NEW: toy-world people/objects + top-down picture |
| `missions/perimeter.py` | tracks-based watch; ZoneMonitor kept |
| `missions/notify.py` | kinds, buttons (Actions header) |
| `missions/wiring.py` | flags, `bind()`, R1/R4 decisions |
| `missions/mission_config.py` | new configs + `indoor_preset` |
| `tools/patrol_sim.py`, `tools/ntfy_test.py` | NEW |
| `tests/test_perimeter_upgrades.py` (replaced), `test_tracks_baseline.py`, `test_detect_responder.py`, `test_investigate_headroom.py`, `test_patrol_sim.py` | 66 tests |
| `PATROL.md`, `CLAUDE.md` | docs |
