"""
The perimeter watch in the toy world: patrol, an intruder walks in, the drone
tracks and investigates them, a laptop goes missing, a door sensor fires.
No hardware, no camera, no models.

    python tools/patrol_sim.py --show                 # watch it (q to quit)
    python tools/patrol_sim.py --seconds 600           # headless, prints the events
    python tools/patrol_sim.py --show --no-investigate # alerts only, the drone keeps patrolling
    python tools/patrol_sim.py --trigger-at 60         # a PIR in the "hall" zone fires at t=60 s

The drone is the real autopilot (same code that flies), task PATROL. People and
objects come from missions/simsource.py (ground truth, with a little noise)
instead of the camera detector; everything after that (tracks, zones, alerts,
baseline, investigate, events) is the real mission code. The baseline learns
faster here than in flight (3 visits per viewpoint instead of 5), so a change shows within the run.

Timeline (default): 0 s patrol starts | 150 s intruder enters from the east,
walks into the study, stays 40 s, leaves | 600 s the laptop in the study is gone.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autonomy.autopilot import Autopilot          # noqa: E402
from autonomy.runner import run                   # noqa: E402
from config import AppConfig                      # noqa: E402
from datatypes import Task                        # noqa: E402
from missions.baseline import SceneBaseline       # noqa: E402
from missions.entities import EventLog            # noqa: E402
from missions.mission_config import BaselineConfig, PerimeterConfig, indoor_preset   # noqa: E402
from missions.perimeter import PerimeterWatch     # noqa: E402
from missions.responder import Responder          # noqa: E402
from missions.simsource import SimObject, SimSceneSource, Walker, render_topdown   # noqa: E402
from missions.wiring import PerimeterMission, apply_patrol_profile   # noqa: E402
from platforms.sim import SimPlatform             # noqa: E402
from sim.scenarios import demo_world              # noqa: E402

ZONES = [{"name": "study", "polygon": [(-9.0, 1.0), (-4.0, 1.0), (-4.0, 6.0), (-9.0, 6.0)]},
         {"name": "hall", "polygon": [(4.0, -6.0), (9.0, -6.0), (9.0, -1.0), (4.0, -1.0)]}]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--seconds", type=float, default=1200)
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-investigate", action="store_true")
    p.add_argument("--intruder-at", type=float, default=150)
    p.add_argument("--laptop-gone-at", type=float, default=600)
    p.add_argument("--trigger-at", type=float, help="a sensor fires in the hall zone at this time")
    p.add_argument("--events", help="also write the events here (JSON lines)")
    p.add_argument("--quiet", action="store_true", help="no per-event lines")
    return p.parse_args(argv)


def build(args, world=None):
    world = world or demo_world(person=None)
    cfg = AppConfig()
    cfg.safety.max_flight_s = args.seconds + 60      # the toy battery never runs out; let the demo run
    pc = indoor_preset(PerimeterConfig(confirm_s=1.5, detect_every_s=0.5))
    apply_patrol_profile(cfg, pc)
    walkers = [Walker([(9.0, -5.0), (5.0, -3.0), (-1.0, 3.5), (-6.0, 4.0)], args.intruder_at, 1.0, 40.0)]
    objects = [SimObject("laptop", -7.5, 4.5, until_t=args.laptop_gone_at), SimObject("chair", 7.0, -4.5)]
    source = SimSceneSource(world, walkers, objects, cfg.camera.hfov_deg, view_range_m=pc.patrol_view_range_m,
                            every_s=pc.detect_every_s)
    baseline = SceneBaseline(BaselineConfig(cell_m=2.0, min_visits=3, learn_visits=3, revisit_gap_s=5.0))
    events = EventLog(args.events)
    watch = PerimeterWatch(zones=ZONES, config=pc, source=source, baseline=baseline, events=events,
                           hfov_deg=cfg.camera.hfov_deg)
    mission = PerimeterMission(watch, investigate=not args.no_investigate)
    mission.responder = Responder(ZONES, is_airborne=mission.airborne, events=events)
    ap = Autopilot(cfg)
    ap.set_task(Task.PATROL)
    mission.bind(ap)
    return world, SimPlatform(world), ap, mission, walkers, objects


def main(argv=None) -> int:
    args = parse(argv)
    world, platform, ap, mission, walkers, objects = build(args)
    watch = mission.watch
    board, printed = {}, [0]
    fired = [args.trigger_at is None]

    def hook(obs, decision):
        if not fired[0] and obs.now >= args.trigger_at:
            fired[0] = True
            mission.responder.trigger("hall", "pir-hall")
        mission.step(None, obs, decision, board)
        new = watch.events.count - printed[0]
        if new and not args.quiet:
            for e in list(watch.events.recent)[-new:]:
                keys = {k: v for k, v in e.items() if k not in ("type", "t", "wall")}
                print(f"t={'' if e['t'] is None else format(e['t'], '6.1f'):>6}  {e['type']:<18} {keys}")
        printed[0] = watch.events.count
        if args.show and int(obs.now * 10) % 3 == 0:
            import cv2
            st = board.get("mission", {})
            lines = [f"t={obs.now:5.0f}s  {decision.mode.name}  {decision.note}"[:90],
                     f"tracks {len(st.get('tracks', []))}  alerts {st.get('alert_count', 0)}  "
                     f"baseline {st.get('baseline', {})}"[:90]]
            img = render_topdown(world, obs.pose, watch, walkers, objects, obs.now, ap.status()["investigating"],
                                 lines=lines)
            cv2.imshow("patrol sim", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

    run(platform, ap, seconds=args.seconds, on_step=hook)
    mission.close()
    kinds = {}
    for a in watch.recent:
        kinds[a.kind] = kinds.get(a.kind, 0) + 1
    starts = sum(1 for e in watch.events.recent if e["type"] == "investigate.start")
    print(f"\nDone at t={world.t:.0f} s: alerts {kinds or 'none'}, investigations started {starts}, "
          f"collisions {world.collisions}, events {watch.events.count}")
    if args.show:
        import cv2
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
