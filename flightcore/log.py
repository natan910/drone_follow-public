"""Flight log: record what the core saw (CSV), replay it offline through the estimator.

Layout: one row per IMU tick.  A sensor sample that arrived during that tick sits in the same row; its own
timestamp `*_t` may be older than the row's `t` (latency).  If a sensor delivered more than one sample in a
tick, the earlier ones are written as measurement-only rows (IMU columns empty) just BEFORE the IMU row and are
read back into the same tick.  `airborne` records whether the core treated the vehicle as flying at that tick
(it switches the estimator between on-ground ZUPT/gravity updates and flight), so replay reproduces the
online estimate exactly.  Floats are written with repr(), which round-trips exactly.
"""
from __future__ import annotations

import csv
from typing import Iterable, Iterator, Optional, TextIO

import numpy as np

from .config import EstimatorConfig
from .estimation import Eskf
from .hal import BaroSample, BatterySample, GpsSample, ImuSample, MagSample, RangeSample, SensorFrame
from .mathutil import quat_to_euler

COLUMNS = [
    "t", "dt", "gx", "gy", "gz", "ax", "ay", "az", "airborne",
    "baro_t", "baro_alt",
    "gps_t", "gps_n", "gps_e", "gps_d", "gps_vn", "gps_ve", "gps_vd", "gps_fix", "gps_sh", "gps_sv",
    "mag_t", "mx", "my", "mz",
    "rng_t", "rng_m", "rng_valid",
    "batt_t", "batt_v", "batt_frac",
]
_IMU_COLS = ("dt", "gx", "gy", "gz", "ax", "ay", "az", "airborne")


def _r(x) -> str:
    """Exact text for a float (np.float64 too: its own repr() is 'np.float64(..)')."""
    return repr(float(x))


class CsvRecorder:
    """Call `record(frame)` with every SensorFrame handed to FlightCore.step()."""

    def __init__(self, out: TextIO):
        self._w = csv.writer(out)
        self._w.writerow(COLUMNS)
        self._i = {c: k for k, c in enumerate(COLUMNS)}

    def _blank(self) -> list:
        return [""] * len(COLUMNS)

    def record(self, frame: SensorFrame, airborne: Optional[bool] = None) -> None:
        i = self._i
        if airborne is None:
            airborne = frame.airborne
        imu = frame.imu

        # one blank row per sample, then peel the last sample of every sensor into the IMU row
        def rows(samples, fill):
            out = []
            for s in samples:
                r = self._blank()
                fill(r, s)
                out.append(r)
            return out

        def f_baro(r, b):
            r[i["baro_t"]], r[i["baro_alt"]] = _r(b.t), _r(b.alt)

        def f_gps(r, g):
            r[i["gps_t"]] = _r(g.t)
            for k, n in enumerate(("gps_n", "gps_e", "gps_d")):
                r[i[n]] = _r(g.pos_ned[k])
            for k, n in enumerate(("gps_vn", "gps_ve", "gps_vd")):
                r[i[n]] = _r(g.vel_ned[k])
            r[i["gps_fix"]] = int(g.fix)
            if g.sigma_h is not None:
                r[i["gps_sh"]] = _r(g.sigma_h)
            if g.sigma_v is not None:
                r[i["gps_sv"]] = _r(g.sigma_v)

        def f_mag(r, m):
            r[i["mag_t"]] = _r(m.t)
            for k, n in enumerate(("mx", "my", "mz")):
                r[i[n]] = _r(m.field[k])

        def f_rng(r, x):
            r[i["rng_t"]], r[i["rng_m"]], r[i["rng_valid"]] = _r(x.t), _r(x.range_m), int(x.valid)

        def f_batt(r, b):
            r[i["batt_t"]], r[i["batt_v"]], r[i["batt_frac"]] = _r(b.t), _r(b.volts), _r(b.frac)

        main = self._blank()
        main[i["t"]], main[i["dt"]] = _r(imu.t), _r(imu.dt)
        for k, n in enumerate(("gx", "gy", "gz")):
            main[i[n]] = _r(imu.gyro[k])
        for k, n in enumerate(("ax", "ay", "az")):
            main[i[n]] = _r(imu.accel[k])
        if airborne is not None:
            main[i["airborne"]] = int(bool(airborne))

        early = []                                     # measurement-only rows, in arrival order
        for samples, fill in ((frame.baro, f_baro), (frame.gps, f_gps), (frame.mag, f_mag),
                              (frame.range, f_rng), (frame.battery, f_batt)):
            rs = rows(samples, fill)
            if not rs:
                continue
            early.extend(rs[:-1])
            last = rs[-1]
            for k, v in enumerate(last):               # merge the last sample into the IMU row
                if v != "":
                    main[k] = v
        for r in early:
            r[i["t"]] = main[i["t"]]
            self._w.writerow(r)
        self._w.writerow(main)


def _f(row, key) -> Optional[float]:
    v = row.get(key, "")
    return float(v) if v not in ("", None) else None


def read_frames(src: TextIO) -> Iterator[SensorFrame]:
    """Rebuild SensorFrames from a CSV written by CsvRecorder (or by hand / another logger).
    Measurement-only rows are held and attached to the next IMU row."""
    frame = SensorFrame(imu=None)                      # type: ignore[arg-type]
    for row in csv.DictReader(src):
        if _f(row, "baro_t") is not None:
            frame.baro.append(BaroSample(t=_f(row, "baro_t"), alt=_f(row, "baro_alt")))
        if _f(row, "gps_t") is not None:
            fix = _f(row, "gps_fix")
            frame.gps.append(GpsSample(
                t=_f(row, "gps_t"),
                pos_ned=[_f(row, "gps_n"), _f(row, "gps_e"), _f(row, "gps_d")],
                vel_ned=[_f(row, "gps_vn"), _f(row, "gps_ve"), _f(row, "gps_vd")],
                fix=True if fix is None else bool(int(fix)),
                sigma_h=_f(row, "gps_sh"), sigma_v=_f(row, "gps_sv"),
            ))
        if _f(row, "mag_t") is not None:
            frame.mag.append(MagSample(t=_f(row, "mag_t"), field=[_f(row, "mx"), _f(row, "my"), _f(row, "mz")]))
        if _f(row, "rng_t") is not None:
            valid = _f(row, "rng_valid")
            frame.range.append(RangeSample(t=_f(row, "rng_t"), range_m=_f(row, "rng_m"),
                                           valid=True if valid is None else bool(int(valid))))
        if _f(row, "batt_t") is not None:
            frame.battery.append(BatterySample(t=_f(row, "batt_t"), volts=_f(row, "batt_v") or 0.0,
                                               frac=_f(row, "batt_frac")))
        if _f(row, "gx") is None:
            continue                                   # measurement-only row: keep collecting
        frame.imu = ImuSample(
            t=_f(row, "t"), dt=_f(row, "dt"),
            gyro=[_f(row, "gx"), _f(row, "gy"), _f(row, "gz")],
            accel=[_f(row, "ax"), _f(row, "ay"), _f(row, "az")],
        )
        a = _f(row, "airborne")
        frame.airborne = None if a is None else bool(int(a))
        yield frame
        frame = SensorFrame(imu=None)                  # type: ignore[arg-type]


def replay(frames: Iterable[SensorFrame], cfg: Optional[EstimatorConfig] = None, *,
           airborne_from: Optional[float] = None) -> list[dict]:
    """Run logged frames through a fresh ESKF and return one dict per IMU tick after alignment:
    t, position, velocity, roll/pitch/yaw, sigmas, validity flags.

    On-ground handling: a frame's own `airborne` flag (written by CsvRecorder) wins.  For logs without that
    column, `airborne_from` (log seconds) says when flight started; None means the vehicle is treated as
    on the ground for the whole log.
    """
    e = Eskf(cfg)
    out: list[dict] = []
    for fr in frames:
        if not e.initialized:
            e.feed_init(fr.imu, fr.baro, fr.mag, fr.gps)
            continue
        if fr.airborne is not None:
            airborne = fr.airborne
        else:
            airborne = airborne_from is not None and fr.imu.t >= airborne_from
        e.set_static(not airborne)
        e.step(fr.imu, list(fr.baro) + list(fr.gps) + list(fr.mag) + list(fr.range))
        n = e.estimate()
        r, p, y = quat_to_euler(n.q)
        out.append({
            "t": fr.imu.t, "pn": n.p[0], "pe": n.p[1], "pd": n.p[2], "vn": n.v[0], "ve": n.v[1], "vd": n.v[2],
            "roll": r, "pitch": p, "yaw": y, "sig_pos_h": n.sigma_pos_h, "sig_vel_h": n.sigma_vel_h,
            "sig_tilt": n.sigma_tilt, "sig_yaw": n.sigma_yaw,
            "att_ok": int(n.att_valid), "vel_ok": int(n.vel_valid), "pos_ok": int(n.pos_valid),
            "alt_ok": int(n.alt_valid),
        })
    return out


def write_estimates(rows: list[dict], out: TextIO) -> None:
    if not rows:
        return
    w = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: (_r(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
