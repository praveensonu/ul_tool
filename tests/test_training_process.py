import threading
import time
import unittest

from training_process import (
    TrainingAlreadyRunningError,
    TrainingProcessManager,
)


def return_training_result(api_config):
    return {"received": api_config["value"]}


def slow_training(api_config):
    time.sleep(api_config["seconds"])
    return {"status": "completed"}


class TrainingProcessManagerTests(unittest.TestCase):
    def wait_until_running(self, manager):
        deadline = time.monotonic() + 3
        while not manager.is_running and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(manager.is_running)

    def test_returns_child_process_result(self):
        manager = TrainingProcessManager(runner=return_training_result)

        outcome = manager.run({"value": 42})

        self.assertEqual(outcome.status, "success")
        self.assertEqual(outcome.result, {"received": 42})
        self.assertFalse(manager.is_running)

    def test_stop_terminates_active_training_process(self):
        manager = TrainingProcessManager(runner=slow_training, stop_timeout=2)
        outcomes = []
        run_thread = threading.Thread(
            target=lambda: outcomes.append(manager.run({"seconds": 30}))
        )
        run_thread.start()
        self.wait_until_running(manager)

        self.assertTrue(manager.stop())
        run_thread.join(timeout=3)

        self.assertFalse(run_thread.is_alive())
        self.assertEqual(outcomes[0].status, "stopped")
        self.assertFalse(manager.is_running)

    def test_rejects_a_second_active_training_run(self):
        manager = TrainingProcessManager(runner=slow_training, stop_timeout=2)
        run_thread = threading.Thread(target=lambda: manager.run({"seconds": 30}))
        run_thread.start()
        self.wait_until_running(manager)

        with self.assertRaises(TrainingAlreadyRunningError):
            manager.run({"seconds": 1})

        manager.stop()
        run_thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
