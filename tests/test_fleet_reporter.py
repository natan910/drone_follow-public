import threading
import time
import unittest

from fleet.reporter import FleetReporter


class FakePoster:
    """Records every call instead of touching the network."""

    def __init__(self, fail_times=0):
        self.calls = []
        self.fail_times = fail_times
        self.event = threading.Event()

    def __call__(self, url, token, body, timeout_s):
        self.calls.append((url, token, body, timeout_s))
        self.event.set()
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("fleet server unreachable")


class PostOnceTests(unittest.TestCase):
    def test_nothing_to_send_returns_false(self):
        poster = FakePoster()
        r = FleetReporter("http://x", "drone-1", "tok", poster=poster)
        self.assertFalse(r.post_once())
        self.assertEqual(poster.calls, [])

    def test_sends_the_latest_update_with_drone_id_merged_in(self):
        poster = FakePoster()
        r = FleetReporter("http://x", "drone-1", "tok", poster=poster)
        r.update({"mode": "FOLLOW", "battery": 77})
        sent = r.post_once()
        self.assertTrue(sent)
        url, token, body, timeout_s = poster.calls[0]
        self.assertEqual(url, "http://x")
        self.assertEqual(token, "tok")
        self.assertEqual(body, {"drone_id": "drone-1", "mode": "FOLLOW", "battery": 77, "acks": []})

    def test_only_the_latest_of_several_updates_is_sent(self):
        poster = FakePoster()
        r = FleetReporter("http://x", "drone-1", "tok", poster=poster)
        r.update({"mode": "PATROL"})
        r.update({"mode": "FOLLOW"})
        r.update({"mode": "HOVER"})
        r.post_once()
        self.assertEqual(len(poster.calls), 1)
        self.assertEqual(poster.calls[0][2]["mode"], "HOVER")

    def test_a_sent_update_is_not_sent_again(self):
        poster = FakePoster()
        r = FleetReporter("http://x", "drone-1", "tok", poster=poster)
        r.update({"mode": "FOLLOW"})
        self.assertTrue(r.post_once())
        self.assertFalse(r.post_once())
        self.assertEqual(len(poster.calls), 1)

    def test_poster_exception_is_swallowed_not_raised(self):
        poster = FakePoster(fail_times=1)
        r = FleetReporter("http://x", "drone-1", "tok", poster=poster)
        r.update({"mode": "FOLLOW"})
        try:
            sent = r.post_once()
        except Exception:
            self.fail("post_once() must never raise -- a flaky fleet server must not affect flight")
        self.assertTrue(sent)

    def test_on_error_callback_receives_the_exception(self):
        poster = FakePoster(fail_times=1)
        errors = []
        r = FleetReporter("http://x", "drone-1", "tok", poster=poster, on_error=errors.append)
        r.update({"mode": "FOLLOW"})
        r.post_once()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ConnectionError)

    def test_a_failed_send_still_drops_the_update_rather_than_retrying_it(self):
        poster = FakePoster(fail_times=1)
        r = FleetReporter("http://x", "drone-1", "tok", poster=poster)
        r.update({"mode": "FOLLOW"})
        r.post_once()  # fails
        self.assertFalse(r.post_once())  # nothing left to retry
        self.assertEqual(len(poster.calls), 1)


class BackgroundThreadTests(unittest.TestCase):
    def test_start_posts_automatically_without_calling_post_once(self):
        poster = FakePoster()
        r = FleetReporter("http://x", "drone-1", "tok", interval_s=0.02, poster=poster)
        r.update({"mode": "FOLLOW"})
        r.start()
        try:
            self.assertTrue(poster.event.wait(timeout=2.0), "background thread never posted")
        finally:
            r.stop()
        self.assertGreaterEqual(len(poster.calls), 1)

    def test_stop_joins_the_thread_and_stops_further_posts(self):
        poster = FakePoster()
        r = FleetReporter("http://x", "drone-1", "tok", interval_s=0.02, poster=poster)
        r.update({"mode": "FOLLOW"})
        r.start()
        self.assertTrue(poster.event.wait(timeout=2.0))
        r.stop()
        count_at_stop = len(poster.calls)
        time.sleep(0.1)
        self.assertEqual(len(poster.calls), count_at_stop)  # no more posts after stop()

    def test_stop_without_start_does_not_raise(self):
        r = FleetReporter("http://x", "drone-1", "tok", poster=FakePoster())
        r.stop()  # must be a no-op, not an error


if __name__ == "__main__":
    unittest.main()
