"""
The autopilot: one object that turns an Observation into a Decision.

    Observation --> tracker --> map --> safety --> dispatch on Task --> avoid --> ceiling --> shape --> Decision

Priority, highest first:
    pilot has control            -> IDLE    (send nothing)
    safety says land / return    -> LAND / RETURN
    safety says hold             -> HOLD
    operator Task.LAND/RETURN/HOLD -> that, directly
    operator Task.FOLLOW/HOVER, target visible   -> TRACK (vision-servo toward/above the target)
    operator Task.FOLLOW/HOVER, target just lost -> LOST  (turn toward where it went)
    operator Task.HOVER, arrived and dwelled     -> HOVER (parked above where the target was;
                                                            keeps that spot even if they walk off)
    a mission asked for a look (investigate())   -> INVESTIGATE (standoff point, face the spot)
    otherwise                     -> SEARCH / EXPLORE / PATROL (cover the map, oldest-seen first)

Whatever the mode, no command climbs above SafetyConfig.max_altitude_m (metres above
the launch point), nor closer than AvoidConfig.ceiling_clearance_m to the lowest
ceiling the upward range sensor has seen near here (mapping/headroom.py); above
that limit, the command descends.

Task is set live by the operator (typically from the phone — see comms/phone_server.py)
and persists across frames until changed; it defaults to Task.FOLLOW. The
adjustable hover height and every other tunable lives in config.py / ControlConfig.

It does no I/O and never sleeps, so the same code runs in unit tests, in the
toy simulator, and on the drone.
"""

import math
from dataclasses import replace
from typing import Dict, Optional, Tuple

from autonomy.investigate import Investigation, face_rate, standoff_point
from config import AppConfig
from control.command_shaper import CommandShaper
from control.follow_controller import FollowController, FollowOutput
from control.position_hold import PositionHold
from control.target_motion import TargetVelocity
from datatypes import (Decision, DriveCommand, Mode, Observation, Pose, Task, TargetEstimate,
                       TrackerOutput, TrackState)
from mapping.headroom import HeadroomMap
from mapping.occupancy_grid import OccupancyGrid
from navigation.avoidance import ObstacleAvoider
from navigation.follower import WaypointFollower
from navigation.patrol import PatrolPlanner
from perception.geometry import CameraModel
from safety.supervisor import SafetyAction, SafetySupervisor
from tracking.target_tracker import TargetTracker

HOME = (0.0, 0.0)
MIN_CEILING_M = 0.5   # a remembered low ceiling never pushes the altitude limit below this

# (mode, cmd, note, gimbal pitch this step or None to leave the gimbal where it is)
Behaviour = Tuple[Mode, DriveCommand, str, Optional[float]]


class Autopilot:
    def __init__(self, config: Optional[AppConfig] = None,
                 grid: Optional[OccupancyGrid] = None):
        self.cfg = config or AppConfig()
        self.grid = grid or OccupancyGrid(self.cfg.map)
        self.headroom = HeadroomMap(self.cfg.map.size_m, self.cfg.map.resolution_m)
        self.tracker = TargetTracker(self.cfg.tracker)
        self.follow = FollowController(self.cfg.control, self.cfg.camera)
        self.hold = PositionHold(self.cfg.control)
        self.target_motion = TargetVelocity()
        self.planner = PatrolPlanner(self.cfg.patrol)
        self.follower = WaypointFollower(self.cfg.patrol)
        self.avoider = ObstacleAvoider(self.cfg.avoid)
        self.supervisor = SafetySupervisor(self.cfg.safety, HOME)
        self.shaper = CommandShaper(self.cfg.shaping)

        self.mode = Mode.HOLD
        self.task = Task.FOLLOW
        self._last_target: Optional[Tuple[float, float, float]] = None  # x, y, time
        self._spin_until: Optional[float] = None
        self._camera_pitch = self.follow.aim_for_search()
        self._seen_pitch = self._camera_pitch  # camera tilt when the target was last actually seen
        self._hover_anchor: Optional[Pose] = None
        self._arrive_since: Optional[float] = None
        self._pitch_t: Optional[float] = None
        self._investigation: Optional[Investigation] = None
        self.last_investigation = ""          # why the last investigation ended
        self._ceiling_now = self.cfg.safety.max_altitude_m

    # ---- operator interface -------------------------------------------------
    def set_task(self, task: Task) -> None:
        """Called live (typically from the phone) to change what the drone should
        be doing. Changing away from HOVER releases any parked position; HOVER,
        HOLD, RETURN and LAND also end an investigation."""
        if task != self.task:
            self.task = task
            if task != Task.HOVER:
                self._forget_hover()
            if task in (Task.HOVER, Task.HOLD, Task.RETURN, Task.LAND):
                self.stop_investigating(f"operator chose {task.name}")

    def set_hover_height(self, metres: float) -> None:
        """The adjustable "how high above the target" filter, changeable mid-flight."""
        self.cfg.control.hover_height_above_target_m = max(0.0, metres)

    def set_camera_fov(self, hfov_deg: float, aspect: float) -> None:
        """Apply a measured field of view (see perception/calibration.py).
        CameraModel caches its tangents, so it is rebuilt, not patched."""
        self.cfg.camera.hfov_deg, self.cfg.camera.aspect = hfov_deg, aspect
        self.follow.model = CameraModel(self.cfg.camera)

    def apply_operator(self, obs: Observation) -> None:
        """Take the operator's task / hover height / calibration from an
        Observation. step() does this itself; the launch gate (autonomy/launch.py)
        calls it on the ground, before the first step, so nothing sent before
        launch is lost."""
        if obs.task is not None:
            self.set_task(obs.task)
        if obs.hover_height_m is not None:
            self.set_hover_height(obs.hover_height_m)
        if obs.camera_fov is not None:
            self.set_camera_fov(*obs.camera_fov)

    # ---- mission interface (R1: missions/wiring.py) ---------------------------
    def investigate(self, x: float, y: float, now: float, standoff_m: float = 3.0,
                    timeout_s: float = 60.0, reason: str = "", extend_s: Optional[float] = None) -> None:
        """Go and look at (x, y) for timeout_s. Calling it again while it runs moves the
        spot (a walking person); extend_s also pushes the deadline to at least now + extend_s
        (keep watching while they are still in view)."""
        inv = self._investigation
        if inv is not None and now < inv.until:
            inv.x, inv.y = x, y
            inv.reason = reason or inv.reason
            if extend_s is not None:
                inv.until = max(inv.until, now + extend_s)
            return
        self._investigation = Investigation(x, y, max(0.5, standoff_m), now + max(0.0, timeout_s), reason, now)
        self.planner.reset()

    def stop_investigating(self, why: str = "stopped") -> None:
        if self._investigation is not None:
            self._investigation = None
            self.last_investigation = why
            self.planner.reset()

    @property
    def investigating(self) -> bool:
        return self._investigation is not None

    # ---- main entry point -----------------------------------------------------
    def step(self, obs: Observation) -> Decision:
        self.apply_operator(obs)
        if not obs.autonomy_permitted:
            self._handover()
            return self._finish(Decision(DriveCommand(), Mode.IDLE, "pilot has control"))

        out = self.tracker.update(obs.detection, obs.now)
        if out.bbox is not None:            # a trusted sighting this frame
            self._seen_pitch = self._camera_pitch
        self._update_map(obs)
        action, why = self.supervisor.check(obs)
        mode, desired, note, pitch = self._behave(obs, out, action, why)

        flying = mode not in (Mode.LAND, Mode.IDLE)
        ceiling = self._ceiling_now = self._ceiling(obs.pose)
        if flying:
            desired = self.avoider.filter(desired, obs.scan)
            desired = self._limit_climb(desired, obs.pose.z, ceiling)
        cmd = self.shaper.shape(desired, obs.now)
        if flying and obs.pose.z >= ceiling and cmd.up_mps > 0:
            cmd = replace(cmd, up_mps=0.0)   # the shaper eases off a climb; at the ceiling, no easing
        if pitch is not None:
            self._camera_pitch = self._slew_pitch(pitch, obs.now)
        return self._finish(Decision(cmd, mode, note, self._camera_pitch))

    def status(self) -> Dict[str, object]:
        """Small, JSON-friendly snapshot for the phone page and logs."""
        target = self._last_target
        inv = self._investigation
        return {"mode": self.mode.name, "task": self.task.name, "tracker": self.tracker.state.name,
                "hover_height_m": round(self.cfg.control.hover_height_above_target_m, 2),
                "parked": self._hover_anchor is not None,
                "map_known": round(self.grid.known_fraction(), 3),
                "hfov_deg": round(self.cfg.camera.hfov_deg, 1),
                "ceiling_m": self.cfg.safety.max_altitude_m,
                "headroom_m": (round(self._ceiling_now, 2)
                               if self._ceiling_now < self.cfg.safety.max_altitude_m else None),
                "investigating": None if inv is None else {
                    "x": round(inv.x, 1), "y": round(inv.y, 1), "reason": inv.reason,
                    "holding": inv.holding},
                # where the target was last seen, in the same home-relative metres as the pose
                "target_xy": [round(target[0], 1), round(target[1], 1)] if target else None}

    # ---- behaviours -------------------------------------------------------
    def _behave(self, obs: Observation, out: TrackerOutput, action: SafetyAction,
                why: str) -> Behaviour:
        if action == SafetyAction.LAND:
            return Mode.LAND, DriveCommand(), why, None
        if action == SafetyAction.RETURN:
            return self._return_home(obs, why)
        if action == SafetyAction.HOLD:
            return Mode.HOLD, DriveCommand(), why, None

        if self.task == Task.LAND:
            return Mode.LAND, DriveCommand(), "operator requested landing", None
        if self.task == Task.RETURN:
            return self._return_home(obs, "operator requested return")
        if self.task == Task.HOLD:
            return Mode.HOLD, DriveCommand(), "operator requested hold", None
        if self.task == Task.PATROL:
            self._forget_hover()
            return self._patrol(obs)

        # Task.FOLLOW or Task.HOVER: both need the target
        if self._hover_anchor is not None:
            return (Mode.HOVER, self.hold.command(obs.pose, self._hover_anchor),
                    "holding position above where the target was", None)

        if out.state == TrackState.TRACKING and out.target is not None:
            self._remember_target(obs.pose, out.target, obs.now)
            self._spin_until = None
            self.planner.reset()
            down = obs.scan.down if obs.scan is not None else None
            fout = self.follow.desired(out, self._camera_pitch, down)
            cmd = self._with_feedforward(fout, obs, fresh=out.bbox is not None)
            if self.task == Task.HOVER and self._arrived(fout):
                if self._arrive_since is None:
                    self._arrive_since = obs.now
                elif obs.now - self._arrive_since >= self.cfg.control.arrive_dwell_s:
                    self._hover_anchor = obs.pose
                    return Mode.HOVER, DriveCommand(), "arrived: parking here", fout.camera_pitch_deg
            else:
                self._arrive_since = None
            return Mode.TRACK, cmd, "", fout.camera_pitch_deg

        if out.state == TrackState.LOST:
            self._arrive_since = None
            down = obs.scan.down if obs.scan is not None else None
            # The held estimate is in the image as it was when last seen: read it
            # with the camera tilt of that moment, not today's (the gimbal moves on).
            fout = self.follow.desired(out, self._seen_pitch, down)
            chase = self._chase_while_lost(fout, obs)
            if chase is not None:
                return Mode.LOST, chase, "target lost, following where they were heading", fout.camera_pitch_deg
            return Mode.LOST, fout.cmd, "target lost, turning to find it", fout.camera_pitch_deg

        # SEARCHING: no lock at all (or identity fully given up on) -> go find them
        self._arrive_since = None
        self.target_motion.reset()
        return self._patrol(obs)

    def _with_feedforward(self, fout: FollowOutput, obs: Observation, fresh: bool) -> DriveCommand:
        """Add the target's own walking velocity (see control/target_motion.py)."""
        if fresh and fout.rel is not None:
            self.target_motion.sighting(obs.pose, fout.rel, obs.now)
        k = self.cfg.control.velocity_feedforward
        if k <= 0:
            return fout.cmd
        fwd, right = self.target_motion.body_frame(obs.pose, obs.now)
        return replace(fout.cmd, forward_mps=fout.cmd.forward_mps + k * fwd,
                       right_mps=fout.cmd.right_mps + k * right)

    def _chase_while_lost(self, fout: FollowOutput, obs: Observation) -> Optional[DriveCommand]:
        """If they were walking when we lost them, keep going the way they were
        going (their velocity fades out over a couple of seconds) and turn to
        face that way, rather than stopping to spin: stopping is what lets a
        walker get away."""
        fwd, right = self.target_motion.body_frame(obs.pose, obs.now)
        if math.hypot(fwd, right) < 0.2:
            return None
        c = self.cfg.control
        heading_err = math.degrees(math.atan2(right, fwd))
        yaw = max(-c.max_yaw_rate_dps, min(c.max_yaw_rate_dps, c.yaw_kp * heading_err))
        k = c.velocity_feedforward
        return DriveCommand(yaw_rate_dps=yaw, forward_mps=k * fwd, right_mps=k * right,
                            up_mps=fout.cmd.up_mps)

    def _arrived(self, fout: FollowOutput) -> bool:
        c = self.cfg.control
        return (fout.horizontal_m is not None and fout.horizontal_m <= c.arrive_radius_m
                and fout.height_above_target_m is not None
                and abs(fout.height_above_target_m - c.hover_height_above_target_m) <= c.arrive_height_tol_m)

    def _return_home(self, obs: Observation, why: str) -> Behaviour:
        route = self.planner.route_home(self.grid, obs.pose, obs.now, HOME)
        pitch = self.follow.aim_for_search()
        dist = math.hypot(obs.pose.x - HOME[0], obs.pose.y - HOME[1])
        if route is None:
            # The planner gave us nothing. Never land here pretending it's home:
            # fly the straight line; the avoider still guards the way.
            if dist <= self.cfg.patrol.goal_tolerance_m:
                return Mode.LAND, DriveCommand(), f"{why}: home reached ({dist:.1f} m)", None
            return (Mode.RETURN, self.follower.command(obs.pose, [HOME]),
                    f"{why}: no route home, flying straight ({dist:.1f} m to go)", pitch)
        if route.arrived:
            return Mode.LAND, DriveCommand(), f"{why}: home reached ({dist:.1f} m)", None
        return Mode.RETURN, self.follower.command(obs.pose, route.path), why, pitch

    def _patrol(self, obs: Observation) -> Behaviour:
        c = self.cfg.patrol
        pitch = self.follow.aim_for_search()
        looking = self._investigate(obs, pitch)
        if looking is not None:
            return looking
        if self._spin_until is not None:
            if obs.now < self._spin_until:
                return Mode.SEARCH, DriveCommand(yaw_rate_dps=c.search_spin_dps), "scanning the area", pitch
            self._spin_until = None

        hint = self._target_hint(obs.now)
        route = self.planner.update(self.grid, obs.pose, obs.now, hint)
        if route is None:
            return Mode.HOLD, DriveCommand(), "nowhere to go", pitch
        if route.kind == Mode.SEARCH and route.arrived:
            self._last_target = None
            self._spin_until = obs.now + c.search_spin_s
            self.planner.reset()
            return Mode.SEARCH, DriveCommand(yaw_rate_dps=c.search_spin_dps), "scanning the area", pitch
        return route.kind, self.follower.command(obs.pose, route.path), "", pitch

    def _investigate(self, obs: Observation, pitch: float) -> Optional[Behaviour]:
        """R1 (autonomy/investigate.py): reach the standoff point, then hold it facing the spot."""
        inv = self._investigation
        if inv is None:
            return None
        if obs.now >= inv.until:
            self.stop_investigating("time up")
            return None
        c = self.cfg.patrol
        p = obs.pose
        spot = (inv.x, inv.y)
        dist = math.hypot(spot[0] - p.x, spot[1] - p.y)
        tol = max(c.goal_tolerance_m, 0.2 * inv.standoff_m)
        if inv.holding and abs(dist - inv.standoff_m) > 2.5 * tol:
            inv.holding = False                         # they walked off: close in again
        if not inv.holding and abs(dist - inv.standoff_m) <= tol:
            inv.holding, inv.anchor = True, (p.x, p.y)
        yaw = face_rate(p.x, p.y, p.yaw, spot, c.yaw_kp, c.max_yaw_dps)
        if inv.holding:
            ax, ay = self._inside_fence(standoff_point(inv.anchor, spot, inv.standoff_m))
            inv.anchor = (ax, ay)
            cmd = replace(self.hold.command(p, Pose(ax, ay, p.yaw, c.altitude_m)), yaw_rate_dps=yaw)
            return Mode.INVESTIGATE, cmd, f"watching {inv.reason or 'the spot'} ({dist:.1f} m)", pitch
        goal = self._inside_fence(standoff_point((p.x, p.y), spot, inv.standoff_m))
        route = self.planner.route_home(self.grid, p, obs.now, goal)
        cmd = self.follower.command(p, route.path if route is not None else [goal])
        return Mode.INVESTIGATE, cmd, f"going to look: {inv.reason or 'the spot'} ({dist:.1f} m)", pitch

    # ---- helpers ----------------------------------------------------------
    def _inside_fence(self, pt: Tuple[float, float], margin_m: float = 2.0) -> Tuple[float, float]:
        """A goal outside the geofence would latch RETURN: pull it back inside."""
        r = math.hypot(pt[0] - HOME[0], pt[1] - HOME[1])
        lim = self.cfg.safety.geofence_radius_m - margin_m
        if r <= lim or not math.isfinite(lim) or r < 1e-9:
            return pt
        k = max(0.0, lim) / r
        return HOME[0] + (pt[0] - HOME[0]) * k, HOME[1] + (pt[1] - HOME[1]) * k

    def _ceiling(self, pose: Pose) -> float:
        """Altitude limit here: the configured ceiling, or lower under a remembered low ceiling."""
        top = self.cfg.safety.max_altitude_m
        low = self.headroom.ceiling_near(pose, self.cfg.patrol.lookahead_m)
        if low is not None:
            top = min(top, max(MIN_CEILING_M, low - self.cfg.avoid.ceiling_clearance_m))
        return top

    def _limit_climb(self, cmd: DriveCommand, z: float, ceiling: Optional[float] = None) -> DriveCommand:
        """The altitude ceiling, in every flying mode. Climbing slows as the ceiling
        nears (ceiling_kp x metres left) and stops at it; above it the command
        descends, at up to ceiling_descend_mps. Descending is never limited here."""
        s = self.cfg.safety
        top = s.max_altitude_m if ceiling is None else ceiling
        allowed = max(-s.ceiling_descend_mps, s.ceiling_kp * (top - z))
        if cmd.up_mps > allowed:
            return replace(cmd, up_mps=allowed)
        return cmd

    def _slew_pitch(self, target_deg: float, now: float) -> float:
        """Rate-limit the gimbal so it moves at a plausible servo speed; this
        rate-limited value is also what's fed back into the geometry as
        "where the camera is currently pointed" next step."""
        dt = 0.1 if self._pitch_t is None else max(0.0, min(0.25, now - self._pitch_t))
        self._pitch_t = now
        max_step = self.cfg.camera.gimbal_max_rate_dps * dt
        cur = self._camera_pitch
        return cur + max(-max_step, min(max_step, target_deg - cur))

    def _update_map(self, obs: Observation) -> None:
        if obs.scan is not None:
            self.grid.integrate(obs.pose, obs.scan)
            self.headroom.observe(obs.pose, obs.scan.up)
        if obs.frame_age < 1.0:  # only count what the camera could really see
            fov = math.radians(self.cfg.camera.hfov_deg)
            reach = self.cfg.patrol.view_range_m
            if obs.scan is not None:
                ahead = obs.scan.clearance_at(0.0, fov / 2)
                if ahead is not None:
                    reach = min(reach, ahead)  # walls block the view
            self.grid.mark_viewed(obs.pose, fov, reach, obs.now)

    def _remember_target(self, pose: Pose, t: TargetEstimate, now: float) -> None:
        half = math.radians(self.cfg.camera.hfov_deg) / 2
        bearing = math.atan(t.offset_x * math.tan(half))
        rng = self.follow.model.size_at_1m / max(t.size, 1e-3)
        a = pose.yaw + bearing
        self._last_target = (pose.x + rng * math.sin(a), pose.y + rng * math.cos(a), now)

    def _target_hint(self, now: float) -> Optional[Tuple[float, float]]:
        if self._last_target is None:
            return None
        x, y, t = self._last_target
        if now - t > self.cfg.patrol.search_memory_s:
            self._last_target = None
            return None
        return (x, y)

    def _forget_hover(self) -> None:
        self._hover_anchor = None
        self._arrive_since = None

    def _handover(self) -> None:
        """The pilot took over: forget transient state so we resume cleanly."""
        self.shaper.reset()
        self.avoider.reset()
        self.planner.reset()
        self.tracker.reset()
        self.target_motion.reset()
        self._spin_until = None
        self._forget_hover()
        self.stop_investigating("pilot took over")

    def _finish(self, d: Decision) -> Decision:
        self.mode = d.mode
        return d
