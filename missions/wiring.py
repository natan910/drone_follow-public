"""
How main.py runs a mission. One object, four calls:

    mission = make_mission(args, cfg)          # after configure_reid(args, cfg)
    matcher = mission.finder or make_finder(cfg.matcher, cfg.reid)
    mission.attach(matcher, recorder)
    mission.bind(autopilot)                    # right after autopilot = make_autopilot(args, cfg)
    ...every flying step:  mission.step(platform.last_frame, obs, decision, board)
    ...on exit:            mission.close()

--mission follow     (default) nothing changes
--mission perimeter  --zones zones.json [--indoor] [--alerts-dir alerts] [--alert-url URL [--alert-token T]]
                     [--patrol-altitude M] [--investigate] [--baseline baseline.json]
                     [--responder-port 8091 --token T [--phone-url http://drone.local:8080]]
                     tap "Patrol" on the phone (PATROL.md)
--mission pet        --pet-species dog [--pet-height 0.5]          send the pet's photo from the phone
--mission survey     --survey-area field.json [--home-latlon LAT,LON] [--field-map field_map.json]
--mission thermal    [--thermal-index 1] [--alerts-dir alerts]     needs the Lepton (not bought)

A mission never takes the flight down: any error inside step() is printed once
and the mission switches itself off; the drone keeps flying its normal brain
(an investigation already under way still ends at its timeout).
"""

import math
import os
import time
from typing import Callable, Optional

from missions.mission_config import (COCO_CLASSES, BaselineConfig, PerimeterConfig, PetConfig, SurveyConfig,
                                     ThermalConfig, indoor_preset)


def add_mission_args(p) -> None:
    g = p.add_argument_group("missions (missions/wiring.py)")
    g.add_argument("--mission", choices=["follow", "perimeter", "pet", "survey", "thermal"], default="follow")
    g.add_argument("--zones", help="perimeter: zones + optional property JSON (missions/perimeter.py)")
    g.add_argument("--indoor", action="store_true",
                   help="perimeter: rooms, not a yard (2 m patrol, 8 m view, no privacy mask, 1 m viewpoints)")
    g.add_argument("--alerts-dir", default="alerts", help="perimeter / thermal: alert snapshots go here")
    g.add_argument("--events", help="perimeter: event log (JSON lines). Default: <alerts-dir>/events.jsonl")
    g.add_argument("--alert-url", help="perimeter: push each alert (+ snapshot) here, ntfy protocol "
                                       "(missions/notify.py). Your own server, https")
    g.add_argument("--alert-token", help="perimeter: access token for --alert-url (sent as Bearer)")
    g.add_argument("--patrol-altitude", type=float,
                   help="perimeter: patrol height in metres above launch (default 7, --indoor 2)")
    g.add_argument("--investigate", action="store_true",
                   help="perimeter: on an intrusion alert, fly to a standoff point and watch (R1)")
    g.add_argument("--baseline", metavar="FILE",
                   help="perimeter: learn what each viewpoint usually shows, alert on changes (R7). "
                        "Loaded at start, saved on exit. Needs --person-detector yolo")
    g.add_argument("--responder-port", type=int,
                   help="perimeter: listen here for sensor triggers (R4, missions/responder.py). Needs --token")
    g.add_argument("--phone-url", help="perimeter: the phone page address the push's Launch button calls "
                                       "(default http://<this machine's IP>:<--phone-port>)")
    g.add_argument("--detect-sync", action="store_true",
                   help="perimeter: run the detector on the main loop (debugging; slows the loop)")
    g.add_argument("--pet-species", default="dog",
                   help=f"pet: comma list of {', '.join(k for k in COCO_CLASSES if k != 'person')}")
    g.add_argument("--pet-height", type=float, help="pet: your animal's height in metres (dog ~0.5, cat ~0.25)")
    g.add_argument("--survey-area", help="survey: field polygon JSON")
    g.add_argument("--home-latlon", help="survey: launch spot as LAT,LON -> writes survey.waypoints")
    g.add_argument("--field-map", help="survey: vegetation map JSON, loaded at start and saved on exit")
    g.add_argument("--survey-index", choices=["vari", "ndvi"], default="vari",
                   help="survey: vari = normal camera; ndvi = NoIR camera + blue filter")
    g.add_argument("--thermal-index", type=int, default=1, help="thermal: /dev/videoN of the Lepton")
    g.add_argument("--curate", action="store_true",
                   help="with --record: keep hard moments (lost / reacquired / borderline re-ID ...) "
                        "plus a slow background trickle, instead of a steady 4 fps (dataset/curator.py)")
    g.add_argument("--record-blur", action="store_true",
                   help="with --record: pixelate every face except the target's before saving")


class Mission:
    """--mission follow: does nothing. The base of the others."""
    name = "follow"
    finder = None
    autopilot = None

    def __init__(self):
        self.failed: Optional[str] = None

    def attach(self, matcher, recorder) -> None:
        """Give the recording curator (if any) the matcher's live stats (borderline re-ID)."""
        curator = getattr(recorder, "curator", None)
        if curator is not None and hasattr(matcher, "stats"):
            curator.stats_fn = lambda: getattr(matcher, "stats", None)

    def bind(self, autopilot) -> None:
        """The autopilot, for missions that ask it to go somewhere (Autopilot.investigate)."""
        self.autopilot = autopilot

    def step(self, frame, obs, decision, board: dict) -> None:
        if self.failed:
            return
        try:
            st = self._step(frame, obs, decision)
            if st is not None:
                board["mission"] = st
        except Exception as e:
            self.failed = f"{type(e).__name__}: {e}"
            board["mission"] = {"mission": self.name, "failed": self.failed}
            print(f"Mission {self.name}: switched off after an error ({self.failed}). Flight is unaffected.")

    def _step(self, frame, obs, decision) -> Optional[dict]:
        return None

    def close(self) -> None:
        pass


class PerimeterMission(Mission):
    """Watch (missions/perimeter.py) + the decisions that move the drone:
    R1 investigate an intruder's track, R4 investigate a sensor's zone."""
    name = "perimeter"

    def __init__(self, watch, notifier=None, profile: Optional[dict] = None, responder=None,
                 trigger_server=None, investigate: bool = False, baseline_path: Optional[str] = None,
                 clock: Callable[[], float] = time.monotonic):
        super().__init__()
        self.watch, self.notifier, self.profile = watch, notifier, profile or {}
        self.responder, self.trigger_server = responder, trigger_server
        self.investigate_on, self.baseline_path = investigate, baseline_path
        self._clock = clock
        self._last_step = -math.inf
        self._last_shot = -math.inf
        self.inv: Optional[dict] = None
        self.sortie: Optional[str] = None
        self._t = 0.0

    def airborne(self) -> bool:
        """step() runs only while flying: a recent step = in the air (read by the sensor thread)."""
        return self._clock() - self._last_step < 2.0

    def _step(self, frame, obs, decision):
        from missions.entities import new_sortie_id
        self._last_step, self._t = self._clock(), obs.now
        if self.sortie is None:
            self.sortie = new_sortie_id()
            self.watch.events.write("sortie.start", obs.now, sortie=self.sortie,
                                    x=round(obs.pose.x, 1), y=round(obs.pose.y, 1))
        alerts = self.watch.step(frame, obs)
        self._direct(obs, alerts)
        st = self.watch.status(obs.now)
        st["sortie"] = self.sortie
        st["investigating"] = None if self.inv is None else {k: v for k, v in self.inv.items() if k != "seen"}
        if self.notifier is not None:
            st["notify"] = self.notifier.stats()
        if self.responder is not None:
            st["responder"] = self.responder.status()
        if self.profile:
            st["patrol"] = self.profile
        return st

    # ---- R1 / R4: where to look ---------------------------------------------------------
    def _direct(self, obs, alerts) -> None:
        ap = self.autopilot
        if ap is None:
            return
        pc = self.watch.cfg
        ev = self.watch.events
        if self.inv is not None and not ap.investigating:
            ev.write("investigate.end", obs.now, why=getattr(ap, "last_investigation", "") or "ended", **self._ids())
            self.inv = None
        if self.investigate_on:
            # 1. a zone alert wins: watch that person (unless already watching an alerted one)
            for a in alerts:
                if a.kind == "intrusion" and a.track and not (self.inv and self.inv.get("alerted")):
                    self._switch(ap, obs.now, {"kind": "track", "track": a.track, "zone": a.zone, "alerted": True},
                                 a.x, a.y, f"{a.track} in {a.zone}")
            # 2. otherwise any stranger just seen (a confirmed track) beats a sensor check or nothing
            if self.inv is None or self.inv["kind"] == "sensor":
                fresh = [t for t in self.watch.tracker.confirmed() if obs.now - t.last_t <= 2.0]
                if fresh:
                    tr = max(fresh, key=lambda t: t.last_t)
                    self._switch(ap, obs.now, {"kind": "track", "track": tr.id}, tr.x, tr.y, f"stranger {tr.id}")
            # 3. keep eyes on them: follow their moves, extend while they stay in view
            if self.inv is not None and self.inv["kind"] == "track":
                tr = self.watch.tracker.get(self.inv["track"])
                if tr is None:
                    ap.stop_investigating(f"{self.inv['track']} left")
                    ev.write("investigate.end", obs.now, why="track ended", **self._ids())
                    self.inv = None
                elif tr.last_t > self.inv.get("seen", -math.inf):
                    self.inv["seen"] = tr.last_t
                    ap.investigate(tr.x, tr.y, obs.now, pc.investigate_standoff_m, pc.investigate_timeout_s,
                                   extend_s=pc.investigate_keep_s)
        if self.responder is not None and self.inv is None:
            p = self.responder.take_pending()
            if p is not None:
                self.inv = {"kind": "sensor", "sensor": p["source"], "zone": p["zone"], "since": round(obs.now, 1)}
                self._look(ap, obs.now, p["x"], p["y"], f"sensor {p['source']} at {p['zone']}")
        if self.inv is not None and obs.now - self._last_shot >= pc.investigate_snapshot_s:
            self._last_shot = obs.now
            self.watch.snapshot("investigate", self.inv.get("track") or self.inv.get("sensor", ""))

    def _switch(self, ap, now: float, inv: dict, x: float, y: float, reason: str) -> None:
        if self.inv is not None:
            if self.inv.get("track") == inv.get("track"):
                self.inv.update(inv)                   # same person: now also alerted
                return
            self.watch.events.write("investigate.end", now, why=f"switched to {inv.get('track')}", **self._ids())
            ap.stop_investigating(f"switched to {inv.get('track')}")
        self.inv = {**inv, "since": round(now, 1)}
        self._look(ap, now, x, y, reason)

    def _look(self, ap, now: float, x: float, y: float, reason: str) -> None:
        pc = self.watch.cfg
        ap.investigate(x, y, now, pc.investigate_standoff_m, pc.investigate_timeout_s, reason)
        self.watch.events.write("investigate.start", now, x=round(x, 1), y=round(y, 1), reason=reason,
                                standoff_m=pc.investigate_standoff_m, **self._ids())

    def _ids(self) -> dict:
        i = self.inv or {}
        return {k: i[k] for k in ("track", "sensor", "zone") if k in i}

    def close(self) -> None:
        if self.trigger_server is not None:
            self.trigger_server.stop()
        if self.notifier is not None:
            self.notifier.close()
        if self.watch.baseline is not None and self.baseline_path:
            try:
                self.watch.baseline.save(self.baseline_path)
                print(f"Saved the learned baseline to {self.baseline_path}")
            except OSError as e:
                print(f"Baseline not saved: {e}")
        if self.sortie is not None:
            self.watch.events.write("sortie.end", self._t, sortie=self.sortie, alerts=len(self.watch.recent),
                                    events=self.watch.events.count)
        self.watch.close()
        self.watch.events.close()


class PetMission(Mission):
    name = "pet"

    def __init__(self, finder):
        super().__init__()
        self.finder = finder

    def _step(self, frame, obs, decision):
        return {"mission": "pet", "enrolled": self.finder.has_target, **self.finder.stats}


class SurveyMission(Mission):
    name = "survey"

    def __init__(self, logger, map_path: Optional[str] = None):
        super().__init__()
        self.logger, self.map_path = logger, map_path

    def _step(self, frame, obs, decision):
        self.logger.step(frame, obs)
        return self.logger.status()

    def close(self) -> None:
        if self.map_path:
            self.logger.map.save(self.map_path)
            self.logger.map.to_csv(os.path.splitext(self.map_path)[0] + ".csv")
            print(f"Saved field map to {self.map_path} (+ .csv)")


class ThermalMission(Mission):
    name = "thermal"

    def __init__(self, watch):
        super().__init__()
        self.watch = watch

    def _step(self, frame, obs, decision):
        self.watch.step(obs)
        return self.watch.status()

    def close(self) -> None:
        self.watch.src.close()


def apply_patrol_profile(cfg, pc: PerimeterConfig, property_polygon=None) -> dict:
    """Perimeter watching needs a different patrol than face-following: 'seen' means
    'a person detector could have spotted someone there', not 'a face was
    recognisable' (4 m). Edits cfg.patrol in place, before make_autopilot reads
    it. Returns what it set (shown in the mission status)."""
    patrol, safety = getattr(cfg, "patrol", None), getattr(cfg, "safety", None)
    if patrol is None:
        return {}
    ceiling = getattr(safety, "max_altitude_m", None)
    alt = pc.patrol_altitude_m if ceiling is None else min(pc.patrol_altitude_m, ceiling - pc.ceiling_margin_m)
    patrol.altitude_m = alt
    patrol.view_range_m = max(patrol.view_range_m, pc.patrol_view_range_m)
    patrol.cruise_mps = pc.patrol_cruise_mps
    if property_polygon:
        reach = max(math.hypot(x, y) for x, y in property_polygon) + pc.patrol_radius_margin_m
        fence = getattr(safety, "geofence_radius_m", None)
        if fence is not None:
            if reach > fence - pc.geofence_margin_m:
                print(f"Perimeter: the property reaches {reach:.0f} m from launch, the geofence is {fence:g} m. "
                      f"Patrol stops {pc.geofence_margin_m:g} m inside the fence; launch nearer the middle, "
                      "or raise SafetyConfig.geofence_radius_m AND the FC's FENCE_RADIUS together.")
            reach = min(reach, fence - pc.geofence_margin_m)
        patrol.patrol_radius_m = reach
    return {"altitude_m": round(patrol.altitude_m, 1), "view_range_m": patrol.view_range_m,
            "cruise_mps": patrol.cruise_mps, "radius_m": round(patrol.patrol_radius_m, 1)}


def _perimeter_detector(cfg, baseline_on: bool, detector_factory):
    """One network pass: people + the R7 objects when --baseline, else the usual person detector."""
    if detector_factory is not None:
        return detector_factory(), baseline_on
    if baseline_on:
        if getattr(cfg.reid, "detector", None) == "yolo":
            from missions.detect import CocoSceneDetector
            return CocoSceneDetector(cfg.reid, BaselineConfig().labels), True
        print("--baseline needs --person-detector yolo (a COCO model knows objects): baseline off.")
    from perception.target_finder import make_person_detector
    return make_person_detector(cfg.reid), False


def make_mission(args, cfg, detector_factory: Optional[Callable] = None,
                 lepton_factory: Optional[Callable] = None,
                 embedder_factory: Optional[Callable] = None,
                 notifier_factory: Optional[Callable] = None) -> Mission:
    """Factories are injectable for tests (no model files, no thermal camera, no network)."""
    kind = getattr(args, "mission", None) or "follow"
    cam = cfg.camera
    if kind == "follow":
        return Mission()

    if kind == "perimeter":
        return _make_perimeter(args, cfg, detector_factory, notifier_factory)

    if kind == "pet":
        from perception.pet_finder import make_pet_finder
        species = tuple(s.strip() for s in (getattr(args, "pet_species", None) or "dog").split(",") if s.strip())
        pc = PetConfig(species=species)
        if getattr(args, "pet_height", None):
            pc.body_height_m = args.pet_height
        finder = make_pet_finder(cfg.reid, pc, cam.face_height_m,
                                 detector=detector_factory() if detector_factory else None,
                                 embedder=embedder_factory() if embedder_factory else None)
        cfg.control.hover_height_above_target_m = pc.hover_height_m   # never low over an animal
        print(f"Mission pet ({', '.join(species)}): send a photo of your animal from the phone. "
              f"Hover height {pc.hover_height_m:g} m.")
        return PetMission(finder)

    if kind == "survey":
        from missions.coverage import lawnmower, parse_latlon, path_length, swath_width, write_waypoints
        from missions.geo import file_home, load_polygons
        from perception.veg_index import FieldMap, SurveyLogger
        sc = SurveyConfig(index=getattr(args, "survey_index", None) or "vari")
        home = parse_latlon(args.home_latlon) if getattr(args, "home_latlon", None) else None
        poly = None
        if getattr(args, "survey_area", None):
            home = home or file_home(args.survey_area)
            poly = load_polygons(args.survey_area, home)[0]["polygon"]
            if home is not None:
                wps = lawnmower(poly, swath_width(sc.altitude_m, cam.hfov_deg, sc.overlap))
                out = os.path.join(os.path.dirname(os.path.abspath(args.survey_area)), "survey.waypoints")
                write_waypoints(out, wps, home[0], home[1], sc.altitude_m)
                print(f"Survey: {len(wps) // 2} passes, {path_length(wps) / 1000:.2f} km -> {out} "
                      "(load it in Mission Planner / QGroundControl, fly in AUTO)")
        map_path = getattr(args, "field_map", None)
        fmap = FieldMap.load(map_path) if map_path and os.path.exists(map_path) else None
        return SurveyMission(SurveyLogger(cam.hfov_deg, cam.aspect, sc, fmap, poly), map_path)

    if kind == "thermal":
        from perception.thermal import LeptonSource, ThermalWatch
        tc = ThermalConfig(camera_index=getattr(args, "thermal_index", None) or 1)
        src = lepton_factory() if lepton_factory else LeptonSource(tc.camera_index)
        return ThermalMission(ThermalWatch(src, tc, alerts_dir=getattr(args, "alerts_dir", None)))

    raise SystemExit(f"unknown --mission {kind!r}")


def _make_perimeter(args, cfg, detector_factory, notifier_factory, source=None) -> PerimeterMission:
    from missions.baseline import SceneBaseline
    from missions.entities import EventLog
    from missions.perimeter import PerimeterWatch, load_watch_file
    if not getattr(args, "zones", None):
        raise SystemExit("--mission perimeter needs --zones zones.json")
    zones, prop = load_watch_file(args.zones)
    indoor = bool(getattr(args, "indoor", False))
    pc = indoor_preset(PerimeterConfig()) if indoor else PerimeterConfig()
    if getattr(args, "patrol_altitude", None):
        pc.patrol_altitude_m = args.patrol_altitude
    profile = apply_patrol_profile(cfg, pc, prop)
    alerts_dir = getattr(args, "alerts_dir", None)
    events_path = getattr(args, "events", None) or (os.path.join(alerts_dir, "events.jsonl") if alerts_dir else None)
    events = EventLog(events_path)

    baseline_path = getattr(args, "baseline", None)
    detector, baseline_on = (None, bool(baseline_path)) if source is not None else \
        _perimeter_detector(cfg, bool(baseline_path), detector_factory)
    baseline = None
    if baseline_on:
        bc = BaselineConfig(cell_m=1.0) if indoor else BaselineConfig()
        baseline = SceneBaseline(bc)
        if os.path.exists(baseline_path):
            try:
                print(f"Baseline: {baseline.load(baseline_path)} viewpoint(s) from {baseline_path}")
            except (ValueError, KeyError, OSError) as e:
                print(f"Baseline: {e}. Starting a new one (it will overwrite the file on exit).")

    notifier = None
    drone_id = getattr(args, "drone_id", None) or "drone"
    if getattr(args, "alert_url", None):
        from missions.notify import AlertNotifier
        make = notifier_factory or AlertNotifier
        notifier = make(args.alert_url, token=getattr(args, "alert_token", None), config=pc, drone_id=drone_id)
    watch = PerimeterWatch(detector, zones, cfg.camera.hfov_deg, cfg.camera.aspect, pc, alerts_dir=alerts_dir,
                           property_polygon=prop, on_alert=[notifier.submit] if notifier is not None else [],
                           source=source, baseline=baseline, events=events,
                           threaded=not getattr(args, "detect_sync", False))
    mission = PerimeterMission(watch, notifier, profile, investigate=bool(getattr(args, "investigate", False)),
                               baseline_path=baseline_path)

    port = getattr(args, "responder_port", None)
    if port:
        from missions.responder import Responder, TriggerServer, local_ip
        token = getattr(args, "token", None)
        if not token:
            raise SystemExit("--responder-port needs --token (sensors and the push's Launch button use it)")
        phone_url = getattr(args, "phone_url", None) or f"http://{local_ip()}:{getattr(args, 'phone_port', None) or 8080}"
        mission.responder = Responder(zones, notifier, phone_url, token, is_airborne=mission.airborne, events=events)
        mission.trigger_server = TriggerServer(mission.responder, token, port=port)
        mission.trigger_server.start()
        print(f"Sensors: http://{local_ip()}:{mission.trigger_server.port}/trigger?token=...&zone=<name>"
              f"{'' if notifier else '  (no --alert-url: nobody gets the Launch button)'}")

    print(f"Mission perimeter{' (indoor)' if indoor else ''}: {len(zones)} zone(s) from {args.zones}"
          f"{', property set' if prop else ''}; patrol {profile.get('altitude_m', '?')} m high"
          f"{'; investigate on' if mission.investigate_on else ''}{'; baseline on' if baseline else ''}"
          f"{'; alerts pushed' if notifier is not None else ''}. Tap Patrol on the phone.")
    if not prop and not indoor:
        print("Perimeter: no \"property\" in the zones file: people on the street or next door can alert.")
    return mission
