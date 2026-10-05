"""
Tunable numbers for the recording curator and the extra missions
(missions/*.py, perception/pet_finder.py, perception/veg_index.py,
perception/thermal.py, fleet/sectors.py). Same idea as config.py: every number
lives here, none in the logic. Move these into config.py when convenient.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CuratorConfig:
    """What counts as a frame worth keeping for training (dataset/curator.py)."""
    background_fps: float = 0.2        # a steady trickle of ordinary frames (easy negatives, variety)
    event_fps: float = 2.0             # max rate while something interesting is happening
    burst_s: float = 2.0               # keep recording this long after an event
    cooldown_s: float = 5.0            # the same reason re-triggers at most this often
    max_events_per_min: int = 60       # budget: SD card and later labelling time
    far_target_size: float = 0.03      # face height / frame height below this = "far target"
    borderline_band: float = 0.05      # body re-ID score within this of its threshold = "borderline"
    obstacle_m: float = 2.0            # a range reading closer than this with no target in view
    new_place_cell_m: float = 5.0      # first visit to a cell this big = "new place"


@dataclass
class PerimeterConfig:
    """Mission 1: patrol, alert when a person is inside a restricted zone (missions/perimeter.py)."""
    detect_every_s: float = 1.0       # start a detection at most this often. Runs in a background thread
                                      # (missions/detect.py); YOLOX-640 on a Pi 5 CPU takes ~0.3-0.5 s
    min_score: float = 0.5
    object_min_score: float = 0.45    # R7: objects (laptop, bike ...) below this are not counted
    confirm_s: float = 1.5            # a person inside a zone this long (one track) before an alert
    gap_s: float = 3.0                # a zone counts as occupied this long after its last sighting
    cooldown_s: float = 60.0          # ZoneMonitor (the simple point-based monitor): one alert per zone per this long
    zone_cooldown_s: float = 15.0     # tracks: each track alerts once per zone; any zone alerts at most this often
    max_range_m: float = 40.0         # ground points farther than this are too inaccurate to use
    min_down_deg: float = 5.0         # ray must point at least this far below the horizon
    ignore_target: bool = True        # the enrolled person (the owner) never triggers an alert
    keep_alerts: int = 20             # how many recent alerts the status page shows
    # property ("property" in zones.json, optional)
    ignore_outside_property: bool = True   # feet outside the property (street, neighbours) never alert
    privacy_mask: bool = True              # snapshots: black out sky + ground outside the property
    mask_cell_px: int = 16                 # mask resolution (pixels per tested cell)
    # patrol profile: perimeter replaces the face-follow patrol numbers (config.py PatrolConfig)
    patrol_altitude_m: float = 7.0         # above the launch point. Down TF-Luna reads to ~8 m; trees!
    patrol_view_range_m: float = 15.0      # person-detector reach. Face-follow patrol uses 4 m
    patrol_cruise_mps: float = 1.0
    patrol_radius_margin_m: float = 2.0    # patrol goals: property's farthest corner + this
    geofence_margin_m: float = 3.0         # ...but always this far inside SafetyConfig.geofence_radius_m
    ceiling_margin_m: float = 1.0          # patrol altitude stays this far under SafetyConfig.max_altitude_m
    # R1 investigate (--investigate): on an intrusion alert, go and look
    investigate_standoff_m: float = 8.0    # horizontal distance kept from the person. Never overhead
    investigate_timeout_s: float = 60.0    # then back to patrol...
    investigate_keep_s: float = 20.0       # ...but a person still in view keeps it going until this long after
                                           # the last sighting (the battery supervisor still wins)
    investigate_snapshot_s: float = 5.0    # one saved photo this often while investigating (not pushed)
    # notifications (missions/notify.py, --alert-url)
    notify_queue: int = 20                 # alerts waiting to be sent; the oldest is dropped beyond this
    notify_timeout_s: float = 5.0          # one HTTP attempt
    notify_retries: int = 3                # attempts per alert
    notify_retry_s: float = 5.0            # wait between attempts


def indoor_preset(pc: PerimeterConfig) -> PerimeterConfig:
    """--indoor: rooms, not a yard. Your own home: no privacy mask (it would black out the walls)."""
    pc.patrol_altitude_m = 2.0        # doorways are ~2.0-2.1 m high: 1.5 through doors (--patrol-altitude 1.5)
    pc.patrol_view_range_m = 8.0
    pc.patrol_cruise_mps = 0.5
    pc.max_range_m = 12.0
    pc.min_down_deg = 3.0
    pc.privacy_mask = False
    pc.investigate_standoff_m = 2.5
    pc.ceiling_margin_m = 0.5
    return pc


@dataclass
class TrackConfig:
    """R6: anonymous people tracks on the ground (missions/tracks.py). Metres, seconds."""
    gate_m: float = 2.0                # a sighting this close to a track's prediction can be that track...
    gate_speed_mps: float = 2.0        # ...plus this much per second since it was last seen (walking)
    max_gate_m: float = 8.0
    max_speed_mps: float = 3.0         # velocity estimates are clipped to this (jumps are detector noise)
    velocity_alpha: float = 0.5        # weight of the newest velocity sample
    max_extrapolate_s: float = 2.0     # predict with the velocity for at most this long
    confirm_hits: int = 2              # sightings before a track is real (one false detection is not a person)
    confirm_window_s: float = 5.0      # ...within this long, or the tentative track is dropped
    forget_s: float = 30.0             # a confirmed track unseen this long has left: "track.end"
    path_points: int = 300             # path points kept per track


# the 80 COCO classes, in model output order (YOLOX from OpenCV Zoo)
COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush")
COCO_CLASSES = {"person": 0, "bird": 14, "cat": 15, "dog": 16, "horse": 17, "sheep": 18, "cow": 19}


@dataclass
class BaselineConfig:
    """R7 learn normal: which objects each viewpoint usually sees (missions/baseline.py)."""
    labels: Tuple[str, ...] = ("bicycle", "car", "motorcycle", "truck", "backpack", "handbag", "suitcase",
                               "chair", "couch", "potted plant", "tv", "laptop", "bench")   # things that
                               # stay put. No animals, no people, nothing small (cups: noise)
    cell_m: float = 2.0                # a viewpoint = this big a square... (--indoor: 1 m)
    headings: int = 8                  # ...x this many compass sectors...
    pitch_bucket_deg: float = 15.0     # ...x camera tilt in steps of this
    present_frac: float = 0.5          # in a visit, seen in at least this share of frames = "there"
    min_frames: int = 2                # a visit with fewer detection frames teaches nothing
    revisit_gap_s: float = 10.0        # away from a viewpoint this long = the next look is a new visit
    learn_visits: int = 5              # first visits: plain average. After: moving average
    alpha: float = 0.2                 # moving-average weight of a new visit
    min_visits: int = 5                # no change is reported before a viewpoint has this many visits
    rare_p: float = 0.1                # "never there" = presence below this...
    common_p: float = 0.8              # ..."always there" = presence above this
    confirm_visits: int = 2            # a change must hold this many visits in a row
    cooldown_s: float = 600.0          # the same (viewpoint, object) change is reported at most this often
    label_cooldown_s: float = 300.0    # and "laptop missing" from ANY viewpoint at most this often (several
                                       # viewpoints see the same laptop: one alert, not three)


@dataclass
class ResponderConfig:
    """R4: fixed sensors (PIR, door contact) call the drone (missions/responder.py)."""
    pending_s: float = 180.0           # a trigger waits this long for the drone to be airborne
    source_cooldown_s: float = 30.0    # the same sensor re-triggers at most this often
    max_body_bytes: int = 10_000


@dataclass
class PetConfig:
    """Mission 2: follow one enrolled animal (perception/pet_finder.py)."""
    species: Tuple[str, ...] = ("dog",)   # keys of COCO_CLASSES
    detector_conf: float = 0.35
    body_height_m: float = 0.5        # typical box height of your animal (dog ~0.5, cat ~0.25): sets distance
    hover_height_m: float = 4.0       # replaces ControlConfig.hover_height_above_target_m. Never low over animals
    acquire_threshold: float = 0.0    # 0 = the embedder's own default
    keep_threshold: float = 0.0       # 0 = the embedder's own default
    margin: float = 0.05              # best must beat the runner-up by this much
    confirm_margin: float = 0.08      # this far above acquire = identity confirmed (source "face")
    learn_every_s: float = 1.0        # learn a new look at most this often, only when confirmed
    gallery_size: int = 12            # learned looks kept (the enrolment photo is never forgotten)
    keep_window_s: float = 1.5        # after a hit, the lower keep_threshold applies this long
    max_candidates: int = 4


@dataclass
class SurveyConfig:
    """Mission 5: vegetation map of a field (missions/coverage.py, perception/veg_index.py)."""
    index: str = "vari"               # "vari": normal camera (what you have). "ndvi": NoIR camera + blue filter
    altitude_m: float = 10.0          # survey height (waypoint export)
    overlap: float = 0.3              # side overlap between passes
    cell_m: float = 2.0               # field map resolution
    every_s: float = 0.5              # analyse a frame at most this often
    min_pitch_deg: float = 60.0       # camera must look this steeply down to count
    min_alt_m: float = 3.0
    analysis_width: int = 160         # frames are shrunk to this width first (speed)
    center_frac: float = 0.3          # only the middle of the frame (least distortion) is scored
    green_above: float = 0.05         # index above this counts as "green" (healthy-ish vegetation)


@dataclass
class ThermalConfig:
    """Mission 3: thermal spotter, FLIR Lepton 3.5 on a PureThermal USB board (perception/thermal.py)."""
    camera_index: int = 1
    hfov_deg: float = 57.0            # Lepton 3.5
    aspect: float = 0.75              # 160 x 120
    pitch_deg: float = 60.0           # how the thermal camera is bolted on (0 = level, 90 = down)
    every_s: float = 0.25             # the Lepton makes ~9 frames/s; no point going faster
    min_c: float = 24.0               # a person seen from 10-30 m reads well below skin temperature
    max_c: float = 42.0               # hotter than this: engines, sun-baked roofs, fires -> ignored
    above_background_c: float = 5.0   # and at least this much warmer than the scene's median
    min_area_px: int = 3
    max_area_frac: float = 0.15       # bigger than this share of the image is not a person
    match_m: float = 4.0              # two sightings this close are the same thing
    confirm_hits: int = 3             # seen this many times before it is reported
    forget_s: float = 5.0             # candidate dropped if unseen this long
    cooldown_s: float = 60.0          # the same spot is reported again at most this often
    max_range_m: float = 60.0


@dataclass
class SectorConfig:
    """Mission 6: several drones share one area (fleet/sectors.py)."""
    min_horizontal_m: float = 10.0    # closer than this (and vertically close) = warning
    min_vertical_m: float = 3.0
