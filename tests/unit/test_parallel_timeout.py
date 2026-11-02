import time
import unittest

from scenesmith.utils.parallel import run_isolated_with_timeout


def _sleep_worker(*, duration_s: float) -> None:
    time.sleep(duration_s)


class TestRunIsolatedWithTimeout(unittest.TestCase):
    def test_times_out(self):
        ok, err = run_isolated_with_timeout(
            task_id="sleep_timeout",
            target=_sleep_worker,
            kwargs={"duration_s": 2.0},
            timeout_s=0.1,
            return_value=False,
        )
        self.assertFalse(ok)
        self.assertIn("Timed out", str(err))

    def test_succeeds(self):
        ok, err = run_isolated_with_timeout(
            task_id="sleep_ok",
            target=_sleep_worker,
            kwargs={"duration_s": 0.01},
            timeout_s=2.0,
            return_value=False,
        )
        self.assertTrue(ok)
        self.assertIsNone(err)
