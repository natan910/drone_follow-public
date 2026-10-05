"""
Measure a camera's real field of view on the bench, by hand.

You usually DON'T need this: tap "Calibrate" on the phone page or fleet
console instead (perception/calibration.py) -- the drone measures your face
itself and saves the result. This tool is the manual alternative, for when
you'd rather calibrate from a still photo of something precise (an A4 sheet)
or with no target enrolled.

Every distance/height estimate (perception/geometry.py) depends on
CameraConfig.hfov_deg matching the lens; the 66 deg default is a Pi Camera
Module 3 number. Stand a known distance from the camera with something of a
known height, click its top and bottom in the frame:

    python tools/calibrate_camera.py --distance-m 2.0
        # live webcam; default reference is your own face (~0.22 m)

    python tools/calibrate_camera.py --distance-m 1.5 --height-m 0.297 --image board.jpg
        # a still photo of an A4 sheet (0.297 m) held 1.5 m away

The result is written to camera_calibration.json -- the same file the
one-tap calibration writes and main.py loads at every start.
"""

import argparse
import os
import sys
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CameraConfig  # noqa: E402
from perception.calibration import save_calibration  # noqa: E402
from perception.geometry import fov_from_measurement  # noqa: E402

Point = Tuple[int, int]


def size_frac_from_points(p1: Point, p2: Point, frame_height: int) -> float:
    """Two pixel points (the top and bottom of the reference object) -> its
    height as a fraction of the frame height, the same quantity geometry.py
    calls `size`."""
    if frame_height <= 0:
        raise ValueError("frame_height must be positive")
    frac = abs(p2[1] - p1[1]) / frame_height
    if frac <= 0:
        raise ValueError("the two points are at the same height -- click the top "
                         "and bottom of the reference object, not the same spot twice")
    return frac


# The math lives in perception/geometry.py so the drone's own one-tap
# calibration (perception/calibration.py) uses exactly the same formula.
compute_fov = fov_from_measurement


def aspect_from_frame_shape(shape: Tuple[int, int]) -> float:
    """frame.shape[:2] = (height, width) -> aspect the way CameraConfig means
    it: frame height / frame width."""
    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError("frame has zero height or width")
    return h / w


def pick_reference_points(frame, window_name: str = "calibrate_camera (click top, then bottom; q to cancel)"
                          ) -> Optional[Tuple[Point, Point]]:
    """Show `frame`, collect two left-clicks (top of the reference object,
    then bottom), and return them. None if the window was closed / 'q' was
    pressed before both were collected. Not unit tested -- it's a thin,
    side-effecting GUI wrapper around the pure functions above, which are."""
    import cv2

    points = []

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)
    try:
        while True:
            shown = frame.copy()
            for p in points:
                cv2.drawMarker(shown, p, (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
            if len(points) == 2:
                cv2.line(shown, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow(window_name, shown)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                return None
            if key == ord("r"):
                points.clear()
            if len(points) >= 2:
                return points[0], points[1]
    finally:
        cv2.destroyWindow(window_name)


def report(hfov_deg: float, vfov_deg: float, aspect: float, current: Optional[CameraConfig] = None) -> str:
    lines = [
        f"Measured horizontal field of view: {hfov_deg:.1f} deg  (vertical: {vfov_deg:.1f} deg)",
        f"Measured aspect (frame height / width): {aspect:.4f}",
        "",
        "Paste into config.py's CameraConfig:",
        f"    hfov_deg: float = {hfov_deg:.1f}",
        f"    aspect: float = {aspect:.3f}",
        "",
        "...or apply it for one run without editing anything:",
        f"    python main.py --hfov {hfov_deg:.1f} ...",
    ]
    if current is not None and abs(current.hfov_deg - hfov_deg) > 5:
        lines.append("")
        lines.append(f"Note: this differs from the current default ({current.hfov_deg:.1f} deg) "
                     f"by {abs(current.hfov_deg - hfov_deg):.1f} deg -- distance/height estimates "
                     "using the default will be off until you apply this.")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distance-m", type=float, required=True,
                   help="how far you stood from the camera, in metres, when you marked the reference object")
    p.add_argument("--height-m", type=float, default=CameraConfig().face_height_m,
                   help="real height of the reference object, in metres (default: an adult face, "
                   f"{CameraConfig().face_height_m} m -- stand at --distance-m and click the top and "
                   "bottom of your own face)")
    p.add_argument("--image", help="calibrate from a still photo instead of the live webcam")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--output", default="camera_calibration.json",
                   help="where to save the result (main.py --calibration-file reads the same path)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import cv2

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read image: {args.image}", file=sys.stderr)
            return 1
    else:
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            print(f"Could not open camera {args.camera_index}", file=sys.stderr)
            return 1
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            print("Camera opened but returned no frame", file=sys.stderr)
            return 1

    print(f"Stand {args.distance_m} m from the camera. In the window: click the TOP of the "
         f"reference object (height {args.height_m} m), then its BOTTOM. 'r' to redo, 'q' to cancel.")
    picked = pick_reference_points(frame)
    if picked is None:
        print("Cancelled -- no points picked.")
        return 1

    frame_h = frame.shape[0]
    aspect = aspect_from_frame_shape(frame.shape[:2])
    size_frac = size_frac_from_points(picked[0], picked[1], frame_h)
    hfov_deg, vfov_deg = compute_fov(args.distance_m, args.height_m, size_frac, aspect)
    print()
    print(report(hfov_deg, vfov_deg, aspect, current=CameraConfig()))
    save_calibration(args.output, hfov_deg, aspect, distance_m=args.distance_m,
                     reference_height_m=args.height_m, method="calibrate_camera.py")
    print(f"\nSaved to {args.output}: main.py loads it automatically on every start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
