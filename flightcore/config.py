"""Every tunable number for flightcore lives here (same rule as the repo's config.py).

Units are SI unless the name says otherwise.  Frames: see mathutil.py.
`VehicleParams` is used twice: once as the *nominal* model the controller believes,
once as the *true* plant inside the simulator.  Make them differ to test robustness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace



@dataclass
class VehicleParams:
    mass: float = 1.2                     # kg
    inertia: tuple = (0.011, 0.011, 0.021)  # kg m^2 (Ixx, Iyy, Izz)
    arm: float = 0.225                    # centre -> motor distance, m (X frame)
    max_thrust: float = 8.0               # N per motor at command 1.0
    thrust_expo: float = 0.60             # thrust = (1-e)*u + e*u^2   (u = ESC command 0..1)
    yaw_coeff: float = 0.016              # reaction torque per thrust, m
    motor_tau: float = 0.03               # s, first-order motor+prop lag
    idle: float = 0.06                    # ESC command floor while armed
    drag_xy: float = 0.25                 # 1/s   (specific drag per m/s, body x,y)
    drag_z: float = 0.10                  # 1/s   (body z)
    ang_damp: float = 0.002               # N m s/rad

    @property
    def d(self) -> float:
        """Motor offset along each body axis (X frame)."""
        return self.arm / math.sqrt(2.0)

    @property
    def hover_frac(self) -> float:
        """Thrust fraction (0..1 of max_thrust) per motor that balances weight."""
        return self.mass * 9.80665 / (4.0 * self.max_thrust)


@dataclass
class ControlConfig:
    # --- rate loop (output: angular acceleration demand rad/s^2) ---
    rate_kp: tuple = (18.0, 18.0, 10.0)
    rate_ki: tuple = (10.0, 10.0, 6.0)
    rate_kd: tuple = (0.30, 0.30, 0.0)       # dimensionless, acts on measured angular accel (D on measurement)
    rate_i_max: tuple = (25.0, 25.0, 15.0)   # rad/s^2 integrator clamp
    rate_lpf_hz: float = 60.0                # gyro filter for P term
    rate_dterm_lpf_hz: float = 30.0          # extra filter on derivative
    max_rate: tuple = (4.0, 4.0, 2.0)        # rad/s  (roll, pitch, yaw rate setpoint clamp)
    # --- attitude loop (output: rate setpoint) ---
    att_kp: tuple = (7.0, 7.0, 3.0)          # 1/s
    max_tilt_deg: float = 35.0
    yaw_lag_max_deg: float = 60.0            # heading setpoint may lead heading by at most this
    # --- velocity / position loops (output: acceleration demand m/s^2) ---
    vel_kp_xy: float = 3.0
    vel_ki_xy: float = 0.6
    vel_kp_z: float = 4.0
    vel_ki_z: float = 1.5
    vel_i_max_xy: float = 2.0
    vel_i_max_z: float = 3.0
    pos_kp_xy: float = 0.9
    pos_kp_z: float = 1.2
    max_speed_xy: float = 6.0                # m/s
    max_speed_up: float = 2.5
    max_speed_down: float = 1.5
    max_accel_xy: float = 4.0                # m/s^2 slew of the velocity setpoint (xy)
    max_accel_z: float = 3.0                 # m/s^2 slew of the velocity setpoint (z)
    max_accel_up: float = 5.0                # m/s^2 cap on commanded vertical accel (up)
    max_accel_down: float = 6.0              # m/s^2 cap on commanded vertical accel (down)
    # --- vertical / thrust ---
    min_collective: float = 0.10             # min per-motor thrust fraction while flying
    hover_lpf_tau: float = 6.0               # s, hover-thrust learner
    hover_learn: bool = True
    # --- mixer priorities ---
    airmode: bool = True                     # shift collective to keep attitude authority
    # --- ground ---
    ground_idle: float = 0.0                 # command while armed on ground (0 -> vehicle idle)


@dataclass
class EstimatorConfig:
    # process noise densities
    accel_noise: float = 0.05                # m/s^2/sqrt(Hz)   (includes vibration margin)
    gyro_noise: float = 0.002                # rad/s/sqrt(Hz)
    accel_bias_walk: float = 0.01            # m/s^2 / sqrt(s)  (deliberately >> real sensor: lets the filter
    gyro_bias_walk: float = 0.002            # rad/s / sqrt(s)   absorb bias steps; see tests/test_flightcore_faults)
    baro_bias_walk: float = 0.02             # m / sqrt(s)
    # measurement noise (1-sigma)
    gps_pos_h: float = 1.0
    gps_pos_v: float = 1.5
    gps_vel_h: float = 0.25
    gps_vel_v: float = 0.35
    baro: float = 0.6
    range: float = 0.05
    mag_yaw: float = 0.10                    # rad
    drag_accel: float = 0.6                  # m/s^2
    zupt: float = 0.05                       # m/s
    # gating: innovation test ratio limit in sigmas
    gate_gps_pos: float = 5.0
    gate_gps_vel: float = 5.0
    gate_baro: float = 5.0
    gate_range: float = 5.0
    gate_mag: float = 4.0
    gate_drag: float = 6.0
    # sensor usage
    use_drag_fusion: bool = True
    drag_coeff_xy: float = 0.25              # 1/s, nominal; ~ VehicleParams.drag_xy
    use_range: bool = True
    range_min: float = 0.15
    range_max: float = 7.0
    range_max_tilt_deg: float = 30.0
    mag_declination: float = 0.0             # rad, field azimuth from true north (set per site)
    mag_norm_tol: float = 0.30               # reject mag if |m| off by this fraction
    # delayed fusion
    delay_window_s: float = 0.30             # measurements older than this are dropped
    # initialisation
    init_time_s: float = 1.0
    init_gyro_std_max: float = 0.05          # rad/s: vehicle must be still
    # initial 1-sigma
    init_sigma_pos: float = 1.0
    init_sigma_vel: float = 0.3
    init_sigma_tilt: float = 0.04
    init_sigma_yaw: float = 0.15
    init_sigma_ab: float = 0.15
    init_sigma_gb: float = 0.01
    init_sigma_bb: float = 1.0
    # health
    max_sigma_pos_h: float = 3.0             # m   position considered valid below this
    max_sigma_vel: float = 1.0
    max_sigma_tilt: float = 0.10
    max_sigma_yaw: float = 0.35
    max_sigma_alt: float = 1.5
    reset_after_reject_s: float = 4.0        # persistent GPS rejection -> reset to GPS
    inflate_after_reject_s: float = 1.0      # persistent GPS-velocity rejection -> inflate covariance (unstick)
    mag_reset_after_s: float = 5.0           # persistent (consistent-norm) mag rejection -> reset yaw to mag
    pos_valid_timeout_s: float = 3.0         # position invalid this long after the last accepted GPS position
    vel_valid_timeout_s: float = 3.0         # same for velocity (GPS vel or drag/ZUPT fusion)
    alt_valid_timeout_s: float = 1.0         # baro / range
    range_reject_holdoff_s: float = 2.0
    drag_period_s: float = 0.1               # airborne drag pseudo-measurement period
    static_period_s: float = 0.02            # on-ground ZUPT + gravity period
    gravity_noise: float = 0.20              # m/s^2, on-ground gravity-vector measurement
    static_gyro_noise: float = 0.01          # rad/s, on-ground zero-rotation measurement


@dataclass
class SupervisorConfig:
    # arming
    arm_min_battery: float = 0.30
    arm_max_tilt_deg: float = 10.0
    arm_max_rate: float = 0.15               # rad/s
    require_gps: bool = True
    require_mag: bool = True
    sensor_timeout_s: float = 0.5            # IMU / baro / mag staleness limits
    # takeoff / landing
    takeoff_speed: float = 1.0
    takeoff_tol: float = 0.25
    land_speed: float = 0.6
    land_speed_fast: float = 1.2
    land_fast_above: float = 4.0
    land_slow_below: float = 1.5             # slow descent below this height
    touchdown_alt: float = 0.25
    touchdown_time_s: float = 1.0
    # link loss (no setpoint update)
    setpoint_timeout_s: float = 0.5
    hold_before_land_s: float = 5.0
    # battery (fraction 0..1)
    battery_warn: float = 0.25               # -> RTL (latched)
    battery_crit: float = 0.12               # -> LAND (latched)
    # estimator failsafe
    est_pos_lost_hold_s: float = 1.0         # tolerate this long, then degrade
    # limits
    rtl_altitude: float = 5.0
    rtl_speed: float = 2.5
    rtl_arrive_tol: float = 1.0
    max_altitude: float = 30.0
    max_radius: float = 60.0
    fence_action: str = "rtl"                # 'rtl' | 'land' | 'none'
    arm_timeout_s: float = 20.0              # armed on ground with no takeoff -> disarm
    takeoff_timeout_s: float = 20.0          # climb not finished -> land
    takeoff_kp: float = 1.0                  # 1/s, slows the climb near the target
    brake_accel: float = 2.0                 # m/s^2, used to pick the stopping point when HOLD is entered
    emergency_thrust: float = 0.96           # fraction of learned hover thrust when altitude is unknown
    emergency_max_s: float = 60.0            # last resort: assume landed and disarm after this long
    ground_hint_range: float = 0.35          # raw rangefinder below this (m) while in emergency = on the ground
    ground_hint_time_s: float = 0.5
    impact_accel: float = 25.0               # m/s^2 specific force spike counted as ground impact
    # crash / flip
    kill_tilt_deg: float = 80.0
    kill_tilt_time_s: float = 0.4
    attitude_disagree_deg: float = 20.0      # ESKF vs backup filter (normal aggressive flight peaks ~16 deg)
    attitude_disagree_time_s: float = 1.0
    attitude_disagree_fast_deg: float = 30.0
    attitude_disagree_fast_time_s: float = 0.1
    att_validated_window_s: float = 0.6      # ESKF counts as validated this long after an accepted GPS-velocity update
    backup_sync_rate: float = 0.2            # 1/s, shadow filter follows the ESKF only while it is validated


@dataclass
class FlightConfig:
    loop_hz: float = 500.0
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    control: ControlConfig = field(default_factory=ControlConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)

    @property
    def dt(self) -> float:
        return 1.0 / self.loop_hz


def with_vehicle(cfg: FlightConfig, **kw) -> FlightConfig:
    """Copy of cfg with vehicle fields changed (handy in tests)."""
    return replace(cfg, vehicle=replace(cfg.vehicle, **kw))
