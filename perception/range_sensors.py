"""
Range sensing: turns distance sensors into a RangeScan.

The autopilot only ever sees a RangeScan (a list of beams: bearing + distance),
so a ring of single-point sensors, a multi-zone ToF chip, or columns of a
depth camera's image can all feed it. The one hardware backend included is the
Benewake TF-Luna on a USB-serial adapter (8 m, ~2 degree beam).

Failure policy: a beam with no fresh reading is left OUT of the scan (a blind
spot), and the avoider refuses to drive toward blind spots. "No return" (nothing
within range) is different: that is a beam reporting distance=None.
"""

import math
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from datatypes import RangeBeam, RangeScan


class RangeSensor(ABC):
    @abstractmethod
    def read(self, now: float) -> Tuple[Optional[RangeScan], float]:
        """(scan, age in seconds of its freshest beam). scan is None if no beam is
        fresh. Beams that have gone quiet are simply missing from the scan."""

    def close(self) -> None:
        pass


class TFLunaParser:
    """Decodes the TF-Luna's 9-byte UART frames:
        0x59 0x59 dist_lo dist_hi amp_lo amp_hi temp_lo temp_hi checksum
    dist is centimetres; checksum is the low byte of the sum of the first 8 bytes."""

    def __init__(self, min_strength: int = 100):
        self.min_strength = min_strength
        self._buf = bytearray()

    def feed(self, data: bytes) -> Optional[Tuple[Optional[float], bool]]:
        """Consume bytes; return (distance_m or None, valid) for the newest good
        frame, or None if no complete frame arrived. distance is None when the
        sensor reports no usable return (weak signal or zero distance)."""
        self._buf.extend(data)
        result = None
        while len(self._buf) >= 9:
            if self._buf[0] != 0x59 or self._buf[1] != 0x59:
                del self._buf[0]
                continue
            frame = bytes(self._buf[:9])
            if (sum(frame[:8]) & 0xFF) != frame[8]:
                del self._buf[0]  # corrupt: resync
                continue
            del self._buf[:9]
            dist_cm = frame[2] | (frame[3] << 8)
            amp = frame[4] | (frame[5] << 8)
            usable = dist_cm > 0 and self.min_strength <= amp < 65535
            result = (dist_cm / 100.0 if usable else None, True)
        return result


class TFLunaRing(RangeSensor):
    """Several TF-Lunas, each on its own serial port, at known bearings.

    ports: {bearing_degrees: "/dev/ttyUSB0", ...}  (0 = forward, + = right)
    """

    def __init__(self, ports: Dict[float, str], max_range_m: float = 5.0,
                 stale_after_s: float = 0.3, baud: int = 115200):
        import serial  # pyserial, imported lazily
        self.max_range_m, self.stale_after_s = max_range_m, stale_after_s
        self._ports = {b: serial.Serial(p, baud, timeout=0) for b, p in ports.items()}
        self._parsers = {b: TFLunaParser() for b in ports}
        self._latest: Dict[float, Tuple[Optional[float], float]] = {}

    def read(self, now: float) -> Tuple[Optional[RangeScan], float]:
        t = time.monotonic()
        for bearing, port in self._ports.items():
            got = self._parsers[bearing].feed(port.read(port.in_waiting or 0))
            if got is not None:
                self._latest[bearing] = (got[0], t)

        beams, ages = [], []
        for bearing, (dist, stamp) in self._latest.items():
            age = t - stamp
            if age > self.stale_after_s:
                continue  # stale beam: leave it out so it counts as a blind spot
            ages.append(age)
            far = dist is None or dist > self.max_range_m
            beams.append(RangeBeam(math.radians(bearing), None if far else dist))
        if not beams:
            oldest_fresh = min((t - stamp for _, stamp in self._latest.values()), default=1e9)
            return None, oldest_fresh
        return RangeScan(tuple(beams), self.max_range_m), min(ages)

    def close(self) -> None:
        for p in self._ports.values():
            p.close()
