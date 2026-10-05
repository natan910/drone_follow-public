"""FlightLoop (runtime glue), FlightCore threading/status, and the CSV log + replay.

No unittest.mock: the clock, motor driver and sensor source are small hand-written fakes.
"""
import io
import json
import threading
import time
import unittest

import numpy as np

from flightcore.autopilot import FlightCore
from flightcore.config import FlightConfig
from flightcore.hal import (BaroSample, BatterySample, GpsSample, ImuSample, MagSample, RangeSample,
                            SensorFrame)
from flightcore.log import COLUMNS, CsvRecorder, read_frames, replay, write_estimates
from flightcore.runtime import FlightLoop
from flightcore.sim import SimWorld
from flightcore.sim.harness import SimRunner, SimSource
from flightcore.runtime import ThreadRunner


class FakeClock:
    """Every call advances time by `step` seconds, so a tick that reads the clock twice 'takes' `step`."""

    def __init__(self, step):
        self.step, self.now = step, 0.0

    def __call__(self):
        self.now += self.step
        return self.now


class RecordingMotors:
    def __init__(self, inner=None):
        self.outs = []
        self.inner = inner

    def write(self, out):
        self.outs.append(out)
        if self.inner is not None:
            self.inner.write(out)


class ScriptedSource:
    """Plays a fixed list; a TimeoutError instance in the list is raised, the end returns None."""

    def __init__(self, items):
        self.items = list(items)

    def read(self):
        if not self.items:
            return None
        x = self.items.pop(0)
        if isinstance(x, Exception):
            raise x
        return x


def sim_loop(seed=1, clock=None, **kw):
    """FlightLoop wired to the simulator; motors record and forward to the sim."""
    cfg = FlightConfig()
    world = SimWorld(cfg.vehicle, dt=cfg.dt, seed=seed)
    src = SimSource(world)
    motors = RecordingMotors(src)
    core = FlightCore(cfg)
    loop = FlightLoop(core, src, motors, clock=clock or FakeClock(0.0005), **kw)
    return loop, core, src, motors


def run_until_ready(loop, core, max_ticks=6000):
    for _ in range(max_ticks):
        loop.run_once()
        if core.est.initialized and not core.prearm_check():
            return True
    return False


def imu_frame(t, dt=0.002, **kw):
    return SensorFrame(imu=ImuSample(t=t, dt=dt, gyro=[0.0, 0.0, 0.0], accel=[0.0, 0.0, -9.80665]), **kw)


class LoopBasics(unittest.TestCase):
    def test_ticks_counted_and_stream_end_stops_run(self):
        src = ScriptedSource([imu_frame(0.002 * k) for k in range(10)])
        loop = FlightLoop(FlightCore(), src, RecordingMotors(), clock=FakeClock(0.0001))
        stats = loop.run()
        self.assertEqual(stats.ticks, 10)
        self.assertEqual(stats.overruns, 0)
        self.assertFalse(loop.run_once())                      # ended source stays ended

    def test_max_ticks(self):
        src = ScriptedSource([imu_frame(0.002 * k) for k in range(10)])
        loop = FlightLoop(FlightCore(), src, RecordingMotors(), clock=FakeClock(0.0001))
        self.assertEqual(loop.run(max_ticks=4).ticks, 4)
        self.assertEqual(len(src.items), 6)

    def test_every_tick_writes_motors(self):
        src = ScriptedSource([imu_frame(0.002 * k) for k in range(5)])
        motors = RecordingMotors()
        FlightLoop(FlightCore(), src, motors, clock=FakeClock(0.0001)).run()
        self.assertEqual(len(motors.outs), 5)
        self.assertTrue(all(not o.armed and not o.cmd.any() for o in motors.outs))

    def test_overrun_counting_uses_the_injected_clock(self):
        frames = [imu_frame(0.002 * k) for k in range(20)]
        fast = FlightLoop(FlightCore(), ScriptedSource(frames), RecordingMotors(), clock=FakeClock(0.0005))
        slow = FlightLoop(FlightCore(), ScriptedSource(frames), RecordingMotors(), clock=FakeClock(0.003))
        self.assertEqual(fast.run().overruns, 0)
        s = slow.run()
        self.assertEqual(s.overruns, 20)
        self.assertAlmostEqual(s.max_step_s, 0.003, places=9)
        self.assertAlmostEqual(s.mean_step_s, 0.003, places=9)

    def test_stats_mean_is_zero_before_first_tick(self):
        loop = FlightLoop(FlightCore(), ScriptedSource([]), RecordingMotors())
        self.assertEqual(loop.stats.mean_step_s, 0.0)


class StopAndCrash(unittest.TestCase):
    def test_request_stop_ends_run_and_can_be_cleared(self):
        class StopsItself:
            def __init__(self):
                self.n, self.loop = 0, None

            def read(self):
                self.n += 1
                if self.n == 5:
                    self.loop.request_stop()
                return imu_frame(0.002 * self.n)

        src = StopsItself()
        loop = FlightLoop(FlightCore(), src, RecordingMotors(), clock=FakeClock(0.0001))
        src.loop = loop
        self.assertEqual(loop.run().ticks, 5)
        self.assertEqual(loop.run(max_ticks=3).ticks, 5)          # still stopped
        loop.clear_stop()
        self.assertEqual(loop.run(max_ticks=3).ticks, 8)

    def test_a_crashing_controller_zeroes_the_motors_and_the_error_propagates(self):
        class Exploding(FlightCore):
            def step(self, frame):
                raise ZeroDivisionError("bug")

        motors = RecordingMotors()
        loop = FlightLoop(Exploding(), ScriptedSource([imu_frame(0.002)]), motors, clock=FakeClock(0.0001))
        with self.assertRaises(ZeroDivisionError):
            loop.run_once()
        self.assertEqual(len(motors.outs), 1)
        self.assertFalse(motors.outs[0].armed)
        self.assertFalse(motors.outs[0].cmd.any())
        self.assertEqual(loop.stats.ticks, 0)


class SimRunnerTime(unittest.TestCase):
    def make(self, **kw):
        cfg = FlightConfig()

        class Plain:                                 # a clock that only moves when the test moves it
            t = 0.0

            def __call__(self):
                return self.t

        clock = Plain()
        world = SimWorld(cfg.vehicle, dt=cfg.dt, seed=1)
        core = FlightCore(cfg)
        r = SimRunner(core, world, clock=clock, **kw)
        return r, clock, cfg.dt

    def test_sync_simulates_the_elapsed_time_and_carries_the_remainder(self):
        r, clock, dt = self.make()
        r.start()
        clock.t += 0.1
        r.sync()
        self.assertEqual(r.loop.stats.ticks, 50)
        clock.t += 0.003                              # 1.5 ticks: one now, half a tick owed
        r.sync()
        self.assertEqual(r.loop.stats.ticks, 51)
        clock.t += 0.003                              # 0.001 owed + 0.003 = two ticks
        r.sync()
        self.assertEqual(r.loop.stats.ticks, 53)

    def test_sync_with_no_time_elapsed_does_nothing_and_first_sync_only_sets_the_baseline(self):
        r, clock, dt = self.make()
        clock.t = 123.0
        r.sync()                                      # never started: baseline only
        r.sync()
        self.assertEqual(r.loop.stats.ticks, 0)

    def test_long_stall_is_capped(self):
        r, clock, dt = self.make(max_catchup_s=1.0)
        r.start()
        clock.t += 3600.0
        r.sync()
        self.assertEqual(r.loop.stats.ticks, 500)     # 1 s at 500 Hz, not an hour

    def test_speed_scales_vehicle_time(self):
        r, clock, dt = self.make(speed=2.0)
        r.start()
        clock.t += 0.1
        r.sync()
        self.assertEqual(r.loop.stats.ticks, 100)

    def test_wait_advances_without_wall_time_and_does_not_double_count(self):
        r, clock, dt = self.make()
        r.start()
        r.wait(0.1)
        self.assertEqual(r.loop.stats.ticks, 50)
        r.sync()                                      # no wall time passed: nothing owed
        self.assertEqual(r.loop.stats.ticks, 50)
        clock.t += 0.02
        r.wait(0.02)                                  # 0.02 wall (synced) + 0.02 asked
        self.assertEqual(r.loop.stats.ticks, 50 + 10 + 10)

    def test_simulated_vehicle_actually_runs(self):
        r, clock, dt = self.make()
        r.start()
        r.wait(2.0)
        self.assertTrue(r.core.est.initialized)       # 1 s alignment
        self.assertAlmostEqual(r.core.t, 2.0, delta=0.01)


def wait_for(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


class ThreadRunnerBasics(unittest.TestCase):
    """Mechanics only: the source here is not paced to real time (real hardware paces itself by blocking)."""

    def test_runs_on_its_own_thread_and_stops(self):
        loop, core, src, motors = sim_loop(seed=8)
        r = ThreadRunner(loop)
        r.start()
        try:
            self.assertTrue(wait_for(lambda: core.est.initialized))
            r.sync()                                            # healthy: no exception
        finally:
            r.stop()
        self.assertFalse(r._thread.is_alive())
        r.sync()                                                # stopped on purpose: still fine
        ticks = loop.stats.ticks
        time.sleep(0.05)
        self.assertEqual(loop.stats.ticks, ticks)

    def test_a_crash_in_the_loop_surfaces_in_sync(self):
        class Boom:
            def read(self):
                raise ValueError("sensor bus fell over")

        r = ThreadRunner(FlightLoop(FlightCore(), Boom(), RecordingMotors()))
        r.start()
        self.assertTrue(wait_for(lambda: r.error is not None))
        with self.assertRaisesRegex(RuntimeError, "died"):
            r.sync()
        with self.assertRaises(RuntimeError):
            r.wait(0.0)
        r.stop()

    def test_an_ended_sensor_stream_is_reported(self):
        r = ThreadRunner(FlightLoop(FlightCore(), ScriptedSource([imu_frame(0.002)]), RecordingMotors()))
        r.start()
        self.assertTrue(wait_for(lambda: not r._thread.is_alive()))
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            r.sync()

    def test_restart_clears_a_previous_error(self):
        class Boom:
            def read(self):
                raise ValueError("x")

        loop = FlightLoop(FlightCore(), Boom(), RecordingMotors())
        r = ThreadRunner(loop)
        r.start()
        self.assertTrue(wait_for(lambda: r.error is not None))
        loop.source = ScriptedSource([])                        # now a source that just ends
        r.start()
        self.assertTrue(wait_for(lambda: not r._thread.is_alive()))
        self.assertIsNone(r.error)


class Watchdog(unittest.TestCase):
    def test_short_dropout_zeroes_motors_but_does_not_kill(self):
        loop, core, src, motors = sim_loop(seed=3)
        self.assertTrue(run_until_ready(loop, core))
        core.arm()
        core.takeoff(2.0)
        for _ in range(1000):
            loop.run_once()
        n0 = len(motors.outs)
        src.drop_next = 5
        for _ in range(5):
            self.assertTrue(loop.run_once())                   # a timeout is not the end of the stream
        self.assertEqual(loop.stats.source_timeouts, 5)
        self.assertEqual(loop.stats.consecutive_timeouts, 5)
        for o in motors.outs[n0:]:
            self.assertFalse(o.armed)
            self.assertFalse(o.cmd.any())
        self.assertFalse(loop.stats.killed_by_watchdog)
        loop.run_once()                                        # data is back
        self.assertEqual(loop.stats.consecutive_timeouts, 0)
        self.assertEqual(loop.stats.source_timeouts, 5)        # total is kept
        self.assertTrue(core.sup.armed)

    def test_long_dropout_kills_and_stays_dead(self):
        loop, core, src, motors = sim_loop(seed=4, max_consecutive_timeouts=10)
        self.assertTrue(run_until_ready(loop, core))
        core.arm()
        core.takeoff(2.0)
        for _ in range(1000):
            loop.run_once()
        src.drop_next = 10
        for _ in range(10):
            loop.run_once()
        self.assertTrue(loop.stats.killed_by_watchdog)
        self.assertEqual(core.sup.mode.value, "killed")
        for _ in range(200):                                   # IMU returns: motors must NOT restart
            loop.run_once()
        self.assertTrue(all(not o.armed and not o.cmd.any() for o in motors.outs[-200:]))
        self.assertEqual(core.sup.mode.value, "killed")

    def test_watchdog_kills_only_once(self):
        loop, core, src, motors = sim_loop(seed=5, max_consecutive_timeouts=3)
        self.assertTrue(run_until_ready(loop, core))
        core.arm()
        src.drop_next = 8
        for _ in range(8):
            loop.run_once()
        self.assertTrue(loop.stats.killed_by_watchdog)
        self.assertEqual(loop.stats.source_timeouts, 8)
        self.assertEqual(sum(1 for e in core.sup.events if "kill" in str(e)), 1)


class Threading(unittest.TestCase):
    def test_commands_and_status_from_another_thread(self):
        loop, core, src, motors = sim_loop(seed=6)
        self.assertTrue(run_until_ready(loop, core))
        core.arm()
        core.takeoff(2.0)
        for _ in range(2000):
            loop.run_once()
        errors, stop = [], threading.Event()

        def brain():
            k = 0
            try:
                while not stop.is_set():
                    core.set_velocity_body(0.5 * (k % 3), 0.0, 0.0, 0.0)
                    st = core.status()
                    json.dumps(st)
                    core.hold() if k % 7 == 0 else None
                    k += 1
                    time.sleep(0.0005)   # a real brain calls at 10-30 Hz; a spin loop starved the flight
                                         # loop for > 10 min on 2-core CI runners (2026-10-03)
            except Exception as e:                             # noqa: BLE001 - report any failure
                errors.append(repr(e))

        th = threading.Thread(target=brain)
        th.start()
        try:
            for _ in range(1500):
                loop.run_once()
        finally:
            stop.set()
            th.join(5.0)
        self.assertFalse(th.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(core.sup.airborne)
        self.assertGreater(-core.nav.p[2], 1.0)                # still up there


class StatusJson(unittest.TestCase):
    def test_status_serialises_in_every_phase(self):
        loop, core, src, motors = sim_loop(seed=7)
        json.dumps(core.status())                              # before any data
        self.assertTrue(run_until_ready(loop, core))
        json.dumps(core.status())                              # aligned, disarmed
        core.arm()
        core.takeoff(2.0)
        for _ in range(2000):
            loop.run_once()
        st = json.loads(json.dumps(core.status()))             # airborne
        self.assertTrue(st["airborne"])
        self.assertEqual(len(st["pos_ned"]), 3)
        core.kill()
        loop.run_once()
        self.assertEqual(json.loads(json.dumps(core.status()))["mode"], "killed")


# ---------------------------------------------------------------------------------------- log / replay
def sample_frames():
    """Hand-built frames that exercise every column, several samples per tick and an airborne flip."""
    f0 = imu_frame(0.002)
    f0.airborne = False
    f1 = imu_frame(0.004, baro=[BaroSample(t=0.001, alt=1.25), BaroSample(t=0.003, alt=1.5)],
                   gps=[GpsSample(t=0.0, pos_ned=[1.0, 2.0, -3.0], vel_ned=[0.1, 0.2, 0.3], fix=True,
                                  sigma_h=0.7, sigma_v=1.1),
                        GpsSample(t=0.002, pos_ned=[9.0, 8.0, -7.0], vel_ned=[0.0, 0.0, 0.0], fix=False)],
                   mag=[MagSample(t=0.004, field=[0.2, -0.1, 0.4])],
                   range=[RangeSample(t=0.004, range_m=1.75, valid=False)],
                   battery=[BatterySample(t=0.004, volts=15.6, frac=0.83)])
    f1.airborne = True
    f2 = imu_frame(0.006)
    f2.airborne = True
    return [f0, f1, f2]


def record_text(frames, **kw):
    buf = io.StringIO()
    rec = CsvRecorder(buf)
    for f in frames:
        rec.record(f, **kw)
    return buf.getvalue()


class LogRoundTrip(unittest.TestCase):
    def test_every_field_survives_and_frames_stay_separate(self):
        frames = sample_frames()
        back = list(read_frames(io.StringIO(record_text(frames))))
        self.assertEqual(len(back), 3)
        self.assertEqual([len(b.baro) for b in back], [0, 2, 0])
        self.assertEqual([len(b.gps) for b in back], [0, 2, 0])
        a, b = frames[1], back[1]
        self.assertEqual([x.t for x in b.baro], [0.001, 0.003])            # arrival order kept
        self.assertEqual([x.alt for x in b.baro], [1.25, 1.5])
        self.assertEqual([x.fix for x in b.gps], [True, False])            # fix=0 must not become True
        self.assertEqual([x.sigma_h for x in b.gps], [0.7, None])
        self.assertEqual([x.sigma_v for x in b.gps], [1.1, None])
        np.testing.assert_array_equal(b.gps[0].pos_ned, a.gps[0].pos_ned)
        np.testing.assert_array_equal(b.gps[1].vel_ned, a.gps[1].vel_ned)
        np.testing.assert_array_equal(b.mag[0].field, a.mag[0].field)
        self.assertEqual((b.range[0].range_m, b.range[0].valid), (1.75, False))
        self.assertEqual((b.battery[0].volts, b.battery[0].frac), (15.6, 0.83))
        self.assertEqual(b.imu.t, 0.004)
        self.assertEqual([x.airborne for x in back], [False, True, True])

    def test_extra_samples_are_read_back_into_the_same_tick(self):
        text = record_text(sample_frames())
        lines = text.strip().splitlines()
        self.assertEqual(len(lines), 1 + 3 + 2)             # header + 3 IMU rows + 2 measurement-only rows
        ti, gi = COLUMNS.index("gx"), COLUMNS.index("t")
        early = [ln.split(",") for ln in lines[1:] if ln.split(",")[ti] == ""]
        self.assertEqual(len(early), 2)
        self.assertTrue(all(r[gi] == "0.004" for r in early))
        self.assertEqual(lines[1].split(",")[ti], "0.0")     # first data row: tick 1 IMU (nothing before it)

    def test_no_numpy_repr_in_the_file(self):
        self.assertNotIn("np.float64", record_text(sample_frames()))
        self.assertNotIn("np.", record_text(sample_frames()))

    def test_airborne_default_none_stays_none(self):
        f = imu_frame(0.002)
        back = list(read_frames(io.StringIO(record_text([f]))))
        self.assertIsNone(back[0].airborne)

    def test_explicit_airborne_argument_overrides_frame_flag(self):
        f = imu_frame(0.002)
        f.airborne = False
        back = list(read_frames(io.StringIO(record_text([f], airborne=True))))
        self.assertTrue(back[0].airborne)

    def test_trailing_measurement_only_row_is_dropped(self):
        rows = record_text([imu_frame(0.002)]).strip().splitlines()        # header + one IMU row
        i = COLUMNS.index("baro_t")
        r = [""] * len(COLUMNS)                                              # a baro-only row, no IMU after it
        r[0], r[i], r[i + 1] = "0.004", "0.003", "2.0"
        text = "\n".join(rows) + "\n" + ",".join(r) + "\n"
        back = list(read_frames(io.StringIO(text)))
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0].baro, [])

    def test_hand_written_csv_with_missing_columns(self):
        text = "t,dt,gx,gy,gz,ax,ay,az\n0.002,0.002,0,0,0,0,0,-9.8\n"
        back = list(read_frames(io.StringIO(text)))
        self.assertEqual(len(back), 1)
        self.assertEqual((back[0].baro, back[0].gps, back[0].mag, back[0].range), ([], [], [], []))
        self.assertIsNone(back[0].airborne)


class LogReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Fly a short mission online with a CsvRecorder attached; keep the online estimate of every tick."""
        buf = io.StringIO()
        rec = CsvRecorder(buf)
        loop, core, src, motors = sim_loop(seed=11, recorder=rec.record)
        assert run_until_ready(loop, core)
        core.arm()
        core.takeoff(2.0)
        online = []
        for k in range(4500):                                  # 9 s: takeoff, hover, forward, stop
            if k == 2500:
                core.set_velocity_body(2.0, 0.0, 0.0, 0.0)
            if k == 3500:
                core.set_velocity_body(0.0, 0.0, 0.0, 0.0)
            loop.run_once()
            n = core.nav
            online.append((core.t, n.p.copy(), n.v.copy()))
        cls.text = buf.getvalue()
        cls.online = {round(t, 9): (p, v) for t, p, v in online}
        cls.was_airborne = core.sup.airborne
        cls.ref_rows = replay(read_frames(io.StringIO(cls.text)))

    def test_replay_reproduces_the_online_estimate_exactly(self):
        rows = self.ref_rows
        self.assertGreater(len(rows), 4000)
        checked = 0
        for r in rows:
            ref = self.online.get(round(r["t"], 9))
            if ref is None:
                continue
            p, v = ref
            self.assertEqual((r["pn"], r["pe"], r["pd"]), tuple(p.tolist()))
            self.assertEqual((r["vn"], r["ve"], r["vd"]), tuple(v.tolist()))
            checked += 1
        self.assertGreater(checked, 4000)

    def test_replay_without_the_airborne_column_needs_airborne_from(self):
        """A log from another logger has no flag: 'always on the ground' is wrong once flying, and
        airborne_from fixes it."""
        text = self.text.split("\n")
        ai = COLUMNS.index("airborne")
        stripped = "\n".join(",".join(c for k, c in enumerate(ln.split(",")) if k != ai) for ln in text)
        names = stripped.split("\n", 1)[0].split(",")
        self.assertNotIn("airborne", names)
        frames = list(read_frames(io.StringIO(stripped)))
        self.assertTrue(all(f.airborne is None for f in frames))
        # find when flight started in the flagged log
        t_fly = next(f.imu.t for f in read_frames(io.StringIO(self.text)) if f.airborne)
        good = replay(frames, airborne_from=t_fly)
        ref = self.ref_rows
        self.assertEqual([r["pn"] for r in good], [r["pn"] for r in ref])
        bad = replay(frames)                                    # treats the whole flight as static
        self.assertNotEqual([r["pn"] for r in bad], [r["pn"] for r in ref])

    def test_replay_output_and_writer(self):
        rows = self.ref_rows
        self.assertEqual(rows[0].keys(), rows[-1].keys())
        self.assertTrue(all(r["att_ok"] in (0, 1) for r in rows))
        self.assertEqual(rows[-1]["alt_ok"], 1)
        out = io.StringIO()
        write_estimates(rows, out)
        lines = out.getvalue().strip().splitlines()
        self.assertEqual(len(lines), len(rows) + 1)
        self.assertEqual(lines[0].split(",")[0], "t")
        self.assertNotIn("np.", out.getvalue())
        write_estimates([], io.StringIO())                      # empty is fine

    def test_online_flight_really_flew(self):
        """Guard against a vacuous replay test: the mission moved north and climbed."""
        last = max(self.online)
        p = self.online[last][0]
        self.assertGreater(p[0], 1.0)
        self.assertLess(p[2], -1.5)
        self.assertTrue(self.was_airborne)


if __name__ == "__main__":
    unittest.main()
