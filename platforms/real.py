"""
Real sensors, real (or dry-run) driver:

    camera -> face matcher (+ body re-ID when the face isn't visible) -> Detection
    range sensors          -> RangeScan
    driver                 -> pose, battery, "does a human have the controls?"

Also the only place that services phone enrolment requests, so the matcher is
only ever touched from this one thread.

Pose is relative to the launch point: main.py calls set_home(driver.pose())
right after arming. The flight controller's own local origin (EKF origin) can
be metres away from where this flight started (SITL keeps it across runs), and
home, the geofence and the altitude ceiling all mean "from where we took off".

Launch: a {"launch": true} command is checked at once with launch_check
(set by autonomy/launch.py) and answered with the reason if refused; an
accepted one is picked up by the launch gate with take_launch_request().
"""

import time
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from comms.phone_server import CommandBox, EnrollmentBox
from config import CameraConfig
from control.base_driver import BaseDriver
from datatypes import Detection, Observation, Pose, Task
from perception.calibration import FovCalibrator, save_calibration
from perception.face_matcher import FaceMatcher
from perception.target_finder import TargetFinder
from perception.range_sensors import RangeSensor
from perception.stream_input import FrameSource
from platforms.base import Platform


class RealPlatform(Platform):
    def __init__(self, camera: FrameSource, matcher: Union[FaceMatcher, TargetFinder], driver: BaseDriver,
                 ranger: Optional[RangeSensor] = None,
                 enrollment: Optional[EnrollmentBox] = None,
                 commands: Optional[CommandBox] = None,
                 startup_timeout_s: float = 5.0,
                 calibrator: Optional[FovCalibrator] = None,
                 calibration_path: Optional[str] = None):
        self.camera, self.matcher, self.driver = camera, matcher, driver
        self.ranger, self.enrollment, self.commands = ranger, enrollment, commands
        self.calibrator = calibrator or FovCalibrator(CameraConfig().face_height_m)
        self.calibration_path = calibration_path   # where a finished calibration is saved (None: don't)
        self.calibration_save_error: Optional[str] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_detection: Optional[Detection] = None
        self.last_match_ms = 0.0
        self._last_frame_t = time.monotonic()
        # launch (autonomy/launch.py): None = this platform takes no launch commands
        self.launch_check: Optional[Callable[[], List[str]]] = None
        self.launched = False
        self._launch_requested = False
        self._home: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # driver x, y, z of the launch point
        if ranger is not None:
            self._wait_for_ranger(startup_timeout_s)

    def _wait_for_ranger(self, timeout_s: float) -> None:
        """Do not fly until the range sensors are actually talking."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            scan, _ = self.ranger.read(time.monotonic())
            if scan is not None:
                return
            time.sleep(0.05)
        raise RuntimeError("range sensors are not responding: check wiring and ports "
                           "(tools/check_ranges.py shows raw readings)")

    def now(self) -> float:
        return time.monotonic()

    # ---- launch ----------------------------------------------------------
    def set_home(self, pose: Pose) -> None:
        """From now on, poses are measured from here (x, y, and height z)."""
        self._home = (pose.x, pose.y, pose.z)

    def take_launch_request(self) -> bool:
        """True once per accepted Launch command."""
        requested, self._launch_requested = self._launch_requested, False
        return requested

    def pose(self) -> Pose:
        p = self.driver.pose()
        hx, hy, hz = self._home
        return Pose(p.x - hx, p.y - hy, p.yaw, p.z - hz)

    def observe(self) -> Observation:
        self._service_enrollment()
        task, hover_height_m = self._service_commands()
        frame = self.camera.read()
        t = time.monotonic()
        detection = None
        if frame is None:
            time.sleep(0.02)  # do not spin flat out while the camera is stalled
        else:
            self._last_frame_t = t
            detection = self.matcher.find(frame)
            self.last_match_ms = (time.monotonic() - t) * 1000
        self.last_frame, self.last_detection = frame, detection
        camera_fov = self._feed_calibration(detection, frame, t)

        scan, scan_age = (None, 0.0)
        if self.ranger is not None:
            scan, scan_age = self.ranger.read(t)

        return Observation(
            now=t,
            pose=self.pose(),
            detection=detection,
            scan=scan,
            battery_pct=self.driver.battery_pct(),
            autonomy_permitted=self.driver.autonomy_permitted(),
            frame_age=t - self._last_frame_t,
            scan_age=scan_age,
            task=task,
            hover_height_m=hover_height_m,
            camera_fov=camera_fov,
        )

    def _feed_calibration(self, detection, frame, now) -> Optional[Tuple[float, float]]:
        result = self.calibrator.feed(detection, frame.shape[:2] if frame is not None else None, now)
        if result is not None and self.calibration_path:
            try:
                save_calibration(self.calibration_path, *result, distance_m=self.calibrator.distance_m,
                                 samples=self.calibrator.samples_needed)
                self.calibration_save_error = None
            except OSError as e:   # applied for this flight anyway; just won't survive a reboot
                self.calibration_save_error = str(e)
        return result

    def calibration_status(self) -> dict:
        status = self.calibrator.status()
        if self.calibration_save_error:
            status["save_error"] = self.calibration_save_error
        return status

    def _service_enrollment(self) -> None:
        req = self.enrollment.poll() if self.enrollment is not None else None
        if req is None:
            return
        image, reply = req
        try:
            if image is None:
                self.matcher.clear_target()
                reply.set_result({"ok": True, "target": False})
            else:
                faces = self.matcher.set_target(image)
                result = {"ok": True, "target": True, "faces_in_photo": faces}
                if isinstance(self.matcher, TargetFinder):
                    # 1 = the photo showed their body too, so they can be recognised from behind
                    result["body_looks"] = len(self.matcher.long_term)
                reply.set_result(result)
        except Exception as e:  # reported back to the phone by the server
            reply.set_exception(e)

    def _check_launch(self) -> None:
        """Raises ValueError (-> 422 on the phone, an error line on the console) with the reason."""
        if self.launched:
            raise ValueError("already launched")
        if self.launch_check is None:
            raise ValueError("this drone is not waiting for a launch command")
        problems = self.launch_check()
        if problems:
            raise ValueError("launch refused: " + "; ".join(problems))

    def _service_commands(self):
        """Drain one pending phone command (if any) into the fields the
        autopilot reads from the next Observation. Only ever touched from
        this thread, same pattern as enrolment."""
        req = self.commands.poll() if self.commands is not None else None
        if req is None:
            return None, None
        command, reply = req
        try:
            if command.get("launch"):
                self._check_launch()          # before anything else: a refused launch changes nothing
            task = Task[command["task"]] if "task" in command else None
            hover_height_m = command.get("hover_height_m")
            if "calibrate_distance_m" in command:
                if self.driver.airborne():   # the distance would be a guess; the result could fly us wrong
                    raise ValueError("calibration is a ground procedure: land first")
                # replies at once; progress and result show up in calibration_status()
                self.calibrator.start(float(command["calibrate_distance_m"]), time.monotonic())
            if command.get("launch"):
                self._launch_requested = True
            reply.set_result({"ok": True, **command})
            return task, hover_height_m
        except Exception as e:  # reported back to the phone by the server
            reply.set_exception(e)
            return None, None

    def close(self) -> None:
        self.camera.release()
        if self.ranger is not None:
            self.ranger.close()
