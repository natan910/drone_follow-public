# Missions + curated recording (added 2026-09-25)

New files (all tested with fakes, `tests/test_curator.py`, `tests/test_missions.py`, 51 tests):

| File | What |
|---|---|
| `dataset/curator.py` | `--curate`: record hard moments, not 4 fps. `--record-blur`: pixelate bystanders' faces |
| `missions/mission_config.py` | every tunable number for the below (move into `config.py` later) |
| `missions/geo.py` | pixel -> ground point (flat ground), polygons, lat/lon <-> metres, zones/field JSON loader |
| `missions/wiring.py` | `--mission ...` flags, `make_mission`, the 6 edits in `main.py` call only this |
| `missions/perimeter.py` | #1 perimeter: person in a restricted zone -> alert + snapshot |
| `perception/pet_finder.py` | #2 pet: follow one enrolled dog/cat (COCO YOLOX classes + fur colour) |
| `missions/coverage.py` + `perception/veg_index.py` | #5 survey: lawnmower -> Mission Planner file, VARI/NDVI field map |
| `perception/thermal.py` | #3 thermal spotter (FLIR Lepton 3.5 + PureThermal, not bought) |
| `fleet/sectors.py` | #6 groundwork: split an area between drones, too-close warnings. Not wired |

#4 warehouse: not built. Needs GPS-free positioning (optical flow / VIO hardware) and a smaller frame.

## Map memory (already existed)
`main.py --map area.npz`: loads the occupancy grid at start, saves it on exit (`make_autopilot`, `finally:`).
Grid is launch-relative: launch from the same spot (+-1 m) or the saved map is shifted.
Unchecked: which clock `viewed_t` uses. If monotonic, "last seen" times are wrong after a reboot
(patrol then treats old areas as fresh or stale at random). Check `mapping/occupancy_grid.py` save/load.

## Learning from flights (already existed, never run end to end)
    python main.py ... --record datasets --record-subject subject-01 [--curate] [--record-blur]
    python tools/data.py autolabel        # teacher YOLOX-S boxes every person
    python tools/data.py review           # you fix boxes (space = accept)
    python tools/data.py export           # train / eval lists, split by session
    python -m training.train detector --data exports/v1 --name run1      # Mac, torch, mps
    python -m training.train reid --data exports/v1 --name run1
    python tools/benchmark.py detector --data exports/v1 --model <the run's .onnx> --gate --promote models/person_own.onnx
    python main.py ... --person-detector own        # fly the promoted model
`--curate` rows carry `"why"` in frames.jsonl (lost, reacquired, face_to_body, body_confirmed_by_face,
reid_borderline, far_target, close_obstacle, new_place, burst:*, background): review those first.

## Running the missions
    # 1 perimeter: tap Patrol on the phone page
    python main.py --platform real --driver mavlink --camera pi --phone --mission perimeter --zones zones.json
    # 2 pet: then send the dog's photo from the phone. Hover height forced to 4 m
    python main.py --platform real --driver mavlink --camera pi --phone --mission pet --pet-species dog --pet-height 0.5
    # 5 survey: writes survey.waypoints next to field.json; tap Launch, then RC mode switch -> AUTO
    python main.py --platform real --driver mavlink --camera pi --fixed-camera --fixed-pitch 90 --phone \
        --mission survey --survey-area field.json --home-latlon 0.0,0.0 --field-map field_map.json
    # 3 thermal
    python main.py --platform real --driver mavlink --camera pi --phone --mission thermal --thermal-index 1

zones.json / field.json (metres east/north of the launch point, or Google Maps corners + home):

    {"home": [0.00000, 0.00000],
     "areas": [{"name": "back gate", "polygon": [[-5, 5], [5, 5], [5, 15], [-5, 15]]},
               {"name": "shed", "latlon": [[0.00010, 0.00010], [0.00010, 0.00030], [0.00020, 0.00030]]}]}

Survey: stay inside our geofence (`SafetyConfig.geofence_radius_m`, 30 m) and the FC fence
(`FENCE_RADIUS`), or raise both. In AUTO the brain is IDLE (sends nothing) and only logs.

## Not verified
Nothing here has run on the Mac, the Pi, SITL or real hardware. Sandbox only, against stubs of
`dataset/layout.py`, `datatypes.py`, `perception/person_detector.py` (the repo is private).
`perception/body_reid.py` ColorHistogramEmbedder assumed: `embed(frame, [ltrb boxes]) -> N x D`,
`default_acquire` / `default_keep` (as TargetFinder uses them).
Perimeter: YOLOX-640 on the Pi 5 CPU ~0.3-0.5 s per run (every 1 s): loop rate will dip.
