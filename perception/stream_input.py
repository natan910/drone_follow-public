"""
Where frames come from. Anything with read() -> frame works, so nothing
downstream cares whether it is a laptop webcam or the drone's camera.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np


class FrameSource(ABC):
    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None if the stream ended or failed."""

    @abstractmethod
    def release(self) -> None:
        ...


class WebcamSource(FrameSource):
    """A laptop webcam or any USB (UVC) camera."""

    def __init__(self, camera_index: int = 0, size: Optional[Tuple[int, int]] = None):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        if size:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()


class PiCameraSource(FrameSource):
    """Raspberry Pi camera module via picamera2 (install with apt, not pip)."""

    def __init__(self, size: Tuple[int, int] = (640, 480)):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        # picamera2's "RGB888" is laid out blue-green-red in memory: exactly what OpenCV wants.
        self.cam.configure(self.cam.create_video_configuration(
            main={"size": size, "format": "RGB888"}))
        self.cam.start()

    def read(self) -> Optional[np.ndarray]:
        return self.cam.capture_array()

    def release(self) -> None:
        self.cam.stop()
