"""
All tunable numbers live here, so behaviour changes never require hunting
through logic code. Every value is a starting point to tune.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MatcherConfig:
    backend: str = "insightface"            # "insightface" or "opencv"
    reference_image: Optional[str] = None   # None = wait for a photo from the phone
    camera_index: int = 0
    threshold: Optional[float] = None       # cosine similarity; None = backend default
    yunet_model: str = "models/face_detection_yunet_2023mar.onnx"
    sface_model: str = "models/face_recognition_sface_2021dec.onnx"


@dataclass
class BodyReIDConfig:
    """Keep following someone whose face isn't visible (back turned, seen from
    above) by the look of their whole body. Only ever LEARNS from frames where
    the face confirmed who it is, so it can't slowly drift onto someone else."""
    enabled: bool = True
    detector: str = "yolo"                 # "yolo" (ONNX, needs a download), "own" (ours, trained with
                                            # training/train.py) or "hog" (built in, weak)
    yolo_model: str = "models/person_yolo.onnx"   # download: see README "Recognising them from behind"
    yolo_format: str = "yolox"             # default model below is OpenCV Zoo's YOLOX (Apache-2.0);
                                            # "v8"/"v5"/"auto" for a YOLOv5/v8/11 export instead (AGPL-3.0)
    yolo_input_size: int = 640             # YOLOX wants 640; a v8/v5 export usually wants 320
    yolo_rgb: bool = False                 # YOLOv5/8/11: True; OpenCV Zoo YOLOX: False
    yolo_scale_01: bool = False            # YOLOv5/8/11: True; OpenCV Zoo YOLOX: False
    detector_conf: float = 0.4
    own_model: str = "models/person_own.onnx"  # our detector (TRAINING.md); RGB 0..255 in, CenterNet out
    own_input_size: int = 320              # must match the size it was exported at
    embedder: str = "color"                # "color" (built in), "onnx" (a re-ID network), "fused" (both)
                                            # our own re-ID net (TRAINING.md) is an "onnx" model:
                                            # --reid onnx|fused --reid-model models/reid_own.onnx
    reid_model: str = "models/person_reid_youtu_2021nov.onnx"  # OpenCV Zoo; any OSNet-style export works
    reid_input_w: int = 128
    reid_input_h: int = 256
    fused_deep_weight: float = 0.7        # "fused": share of the score from the network (rest: colour)
    # decisions (None = the embedder's own default, calibrated for it)
    acquire_threshold: Optional[float] = None  # similarity needed to pick them up with no recent track
    keep_threshold: Optional[float] = None     # lower bar while the track is fresh (hysteresis)
    margin: float = 0.05                   # best must beat the runner-up by this much, or we don't guess
    # memory
    gallery_size: int = 12                 # long-term looks, taught ONLY by face-confirmed frames
    merge_above: float = 0.93              # a look this similar to a stored one refines it instead of adding
    short_term_size: int = 6               # looks learned from confident body-only matches...
    short_term_ttl_s: float = 20.0         # ...forgotten after this long (covers turning around)
    self_update_margin: float = 0.05       # body-only match must beat its threshold by this to teach
    # effort
    enroll_every_n: int = 5                # while the face is visible, learn the body every Nth frame
    max_candidates: int = 4                # embed at most this many people per frame (closest first)
    roi_scale: float = 2.5                 # search a crop this many body-sizes around the prediction first
    # motion gate (units: body heights; the prediction is constant-velocity)
    gate_body_heights: float = 0.8
    gate_growth_per_s: float = 1.0
    keep_window_s: float = 1.5             # track counts as "fresh" (keep_threshold) this long after a hit
    motion_max_age_s: float = 4.0          # after this, no prediction: search the whole frame


@dataclass
class CameraConfig:
    hfov_deg: float = 66.0             # Pi Camera Module 3 standard lens (wide is ~102)
    aspect: float = 0.75               # frame height / width (4:3)
    face_height_m: float = 0.22        # real height of a face box; sets the distance estimate. CALIBRATE.
    face_to_head_top_m: float = 0.12   # from the middle of the face box up to the top of the head
    gimbal: bool = True                # True: the camera tilts on a servo and the autopilot aims it
    fixed_pitch_deg: float = 30.0      # used when there is no gimbal (camera is bolted at this tilt)
    search_pitch_deg: float = 25.0     # gimbal angle while patrolling / searching
    gimbal_kp: float = 0.8             # how hard the gimbal corrects the target's vertical offset
    gimbal_max_rate_dps: float = 120.0 # a hobby servo manages 300+; keep some margin for smoothness
    aim_below_face_m: float = 0.30     # aim at the chest, not the face: keeps the face in the upper
                                       # part of the frame AND the body in view for body re-ID
    min_pitch_deg: float = 0.0         # 0 = level
    max_pitch_deg: float = 90.0        # 90 = looking straight down


@dataclass
class TrackerConfig:
    lock_frames: int = 5          # consecutive FACE detections needed to trust a match
    grace_period: float = 0.5     # seconds of dropouts tolerated while TRACKING
    give_up_timeout: float = 5.0  # seconds in LOST before going back to SEARCHING
    smoothing: float = 0.4        # 0..1, weight of newest sample (lower = smoother)
    track_only_timeout_s: float = 45.0  # how long to keep trusting a visual-only ("track")
                                        # detection after the last confirmed face (e.g. hovering
                                        # directly overhead, where no face is visible)


@dataclass
class ControlConfig:
    """Following behaviour: approach the target, hover above it, then keep station."""
    hover_height_above_target_m: float = 0.30  # <-- the adjustable clearance above the top of the head
    standoff_m: float = 0.0            # horizontal distance to keep from directly overhead; 0 = centred
    horizontal_kp: float = 1.0         # m/s per metre of horizontal error
    max_horizontal_mps: float = 1.5    # a brisk walk is ~1.4 m/s; slower than this and they walk away
    velocity_feedforward: float = 1.0  # add this share of the target's own walking velocity (0 = off)
    vertical_kp: float = 1.0           # m/s per metre of height error
    max_climb_mps: float = 0.5
    max_descend_mps: float = 0.3
    height_deadband_m: float = 0.03
    horizontal_deadband_m: float = 0.10
    approach_radius_m: float = 1.5     # inside this horizontal range, do not close further while too high
    down_valid_radius_m: float = 0.6   # trust the downward rangefinder as "target height" only this close
    yaw_kp: float = 1.0                # deg/s per degree of bearing error
    max_yaw_rate_dps: float = 45.0
    yaw_deadband_deg: float = 3.0
    yaw_release_m: float = 0.8         # inside this, turning toward it fades out (0 when overhead)
    lost_search_yaw_dps: float = 20.0
    lost_hold_within_m: float = 1.0    # lost this close and in front (probably right below us): hold, don't spin
    arrive_radius_m: float = 0.25      # HOVER: counts as "over the target" inside this horizontal distance...
    arrive_height_tol_m: float = 0.10  # ...and this close to the desired hover height...
    arrive_dwell_s: float = 1.5        # ...continuously for this long before parking
    hold_kp: float = 0.6               # HOVER once parked: m/s per metre of world-frame position error
    hold_deadband_m: float = 0.10
    hold_max_mps: float = 0.6


@dataclass
class ShapingConfig:
    """Final limits applied to every command, whatever produced it."""
    max_yaw_rate_dps: float = 45.0
    max_forward_mps: float = 1.5
    max_backward_mps: float = 0.8
    max_right_mps: float = 1.5
    max_up_mps: float = 0.5
    max_down_mps: float = 0.4
    max_yaw_accel_dps2: float = 180.0
    max_forward_accel_mps2: float = 1.0
    max_right_accel_mps2: float = 1.0
    max_vertical_accel_mps2: float = 0.8
    brake_multiplier: float = 3.0  # slowing down is allowed this much faster


@dataclass
class MapConfig:
    size_m: float = 60.0           # square map centred on home
    resolution_m: float = 0.5      # GPS-grade pose does not justify finer
    l_free: float = 0.6            # log-odds added per "free" observation (subtracted)
    l_occ: float = 0.9             # log-odds added per "hit"
    l_min: float = -3.0
    l_max: float = 3.0
    free_below: float = -0.4       # cell counts as free under this
    occupied_above: float = 0.8    # ...and as occupied over this
    drone_radius_m: float = 0.35   # half-width of the drone including props
    safety_margin_m: float = 0.4


@dataclass
class PatrolConfig:
    cruise_mps: float = 0.6
    altitude_m: float = 2.0        # cruise height while patrolling / searching
    altitude_kp: float = 1.0
    lookahead_m: float = 1.0
    goal_tolerance_m: float = 0.6
    slow_radius_m: float = 1.5
    turn_in_place_deg: float = 35.0
    yaw_kp: float = 1.5            # deg/s per degree of heading error
    max_yaw_dps: float = 40.0
    replan_interval_s: float = 1.0
    unknown_cost: float = 2.0      # planning through unseen space costs this much per metre
    view_range_m: float = 4.0      # how far the camera can recognise a face
    staleness_cap_s: float = 900.0
    distance_weight_s_per_m: float = 5.0   # how much a far goal is penalised
    switch_margin_s: float = 30.0          # hysteresis when changing goals
    patrol_radius_m: float = 20.0          # never pick goals farther than this from home
    stuck_window_s: float = 8.0
    stuck_distance_m: float = 0.3
    blacklist_s: float = 60.0
    search_memory_s: float = 30.0          # how long to remember where the target was
    search_spin_s: float = 12.0            # spin in place this long on arrival
    search_spin_dps: float = 30.0


@dataclass
class AvoidConfig:
    stop_distance_m: float = 1.0
    slow_distance_m: float = 3.0
    escape_distance_m: float = 2.0  # head-on this close: commit to turning one way
    front_half_angle_deg: float = 45.0
    push_gain_dps: float = 30.0
    max_push_dps: float = 40.0
    rear_clear_m: float = 1.0      # reversing needs a rear sensor reading at least this
    # vertical protection (applies in every mode, including hovering over the target)
    min_clearance_below_m: float = 0.30   # hard floor: never let the downward reading get closer than this
    blind_descend_limit_m: float = 1.0    # floor used when there is no downward reading at all
    ceiling_clearance_m: float = 0.5      # keep at least this far below whatever the upward sensor sees
    rescue_climb_mps: float = 0.4         # climb speed used to back off from the hard floor


@dataclass
class SafetyConfig:
    geofence_radius_m: float = 30.0
    battery_return_pct: float = 30.0
    battery_land_pct: float = 15.0
    max_flight_s: float = 600.0
    require_scan: bool = True      # False only for the webcam dry run
    scan_hold_s: float = 0.5
    scan_land_s: float = 3.0
    frame_hold_s: float = 1.0
    frame_return_s: float = 10.0
    # altitude ceiling, metres above the launch point (every mode; the autopilot never commands a climb
    # above it). Keep the flight controller's own fence (ArduPilot FENCE_ALT_MAX) a little higher as backstop.
    max_altitude_m: float = 15.0
    ceiling_kp: float = 1.0                # climb allowed = this x metres left below the ceiling (slows the approach)
    ceiling_descend_mps: float = 0.3       # above the ceiling: come back down at up to this speed
    altitude_return_margin_m: float = 3.0  # this far above the ceiling anyway (e.g. wind): RETURN (latched)
    # battery level unknown. main.py sets require_battery for --driver mavlink (a real flight controller or
    # SITL must report it); print / native-sim report None legitimately, so the default is off.
    require_battery: bool = False          # True: refuse launch while unknown, RETURN if unknown in flight...
    battery_unknown_return_s: float = 10.0  # ...for longer than this


@dataclass
class PhoneConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    token: Optional[str] = None    # None = generate a random one at startup
    max_upload_mb: float = 8.0


@dataclass
class AppConfig:
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    reid: BodyReIDConfig = field(default_factory=BodyReIDConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    shaping: ShapingConfig = field(default_factory=ShapingConfig)
    map: MapConfig = field(default_factory=MapConfig)
    patrol: PatrolConfig = field(default_factory=PatrolConfig)
    avoid: AvoidConfig = field(default_factory=AvoidConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    phone: PhoneConfig = field(default_factory=PhoneConfig)
    show_window: bool = True
