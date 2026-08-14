from __future__ import annotations

import multiprocessing
import os
import queue
import signal
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

from orchestrator import run_orchestrator


class TrainingAlreadyRunningError(RuntimeError):
    pass


class TrainingProcessError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingOutcome:
    status: Literal["success", "stopped"]
    result: Optional[Dict[str, Any]] = None


def _training_worker(
    api_config: Dict[str, Any],
    result_queue: Any,
    runner: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> None:
    try:
        result_queue.put({"status": "success", "result": runner(api_config)})
    except Exception as exc:
        traceback.print_exc()
        result_queue.put(
            {
                "status": "error",
                "error": str(exc),
                "exception_type": type(exc).__name__,
            }
        )


class TrainingProcessManager:
    """Own the single training child process used by the API server."""

    def __init__(
        self,
        runner: Callable[[Dict[str, Any]], Dict[str, Any]] = run_orchestrator,
        stop_timeout: float = 5.0,
    ) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._runner = runner
        self._stop_timeout = stop_timeout
        self._condition = threading.Condition()
        self._process: Optional[multiprocessing.Process] = None
        self._stop_requested = False

    @property
    def is_running(self) -> bool:
        with self._condition:
            return self._process is not None

    def run(self, api_config: Dict[str, Any]) -> TrainingOutcome:
        result_queue = self._context.Queue()

        with self._condition:
            if self._process is not None:
                result_queue.close()
                raise TrainingAlreadyRunningError("A training run is already active.")

            process = self._context.Process(
                target=_training_worker,
                args=(api_config, result_queue, self._runner),
                name="ascent-training",
            )
            self._process = process
            self._stop_requested = False

            try:
                process.start()
            except Exception:
                self._process = None
                result_queue.close()
                self._condition.notify_all()
                raise

        process.join()

        with self._condition:
            was_stopped = self._stop_requested
            if self._process is process:
                self._process = None
                self._stop_requested = False
                self._condition.notify_all()

        if was_stopped:
            result_queue.close()
            process.close()
            return TrainingOutcome(status="stopped")

        try:
            message = result_queue.get(timeout=1.0)
        except queue.Empty as exc:
            raise TrainingProcessError(
                f"Training process exited unexpectedly with code {process.exitcode}."
            ) from exc
        finally:
            result_queue.close()
            process.close()

        if message.get("status") == "error":
            exception_type = message.get("exception_type", "TrainingError")
            error = message.get("error", "Training failed without an error message.")
            raise TrainingProcessError(f"{exception_type}: {error}")

        return TrainingOutcome(status="success", result=message["result"])

    def stop(self) -> bool:
        """Terminate the active training child and wait for the run call to unwind."""

        with self._condition:
            process = self._process
            if process is None or process.pid is None:
                return False

            self._stop_requested = True
            process_pid = process.pid

        try:
            os.kill(process_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        with self._condition:
            stopped = self._condition.wait_for(
                lambda: self._process is not process,
                timeout=self._stop_timeout,
            )

        if stopped:
            return True

        try:
            os.kill(process_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        with self._condition:
            stopped = self._condition.wait_for(
                lambda: self._process is not process,
                timeout=self._stop_timeout,
            )

        if not stopped:
            raise TrainingProcessError("Training process did not terminate.")

        return True


training_process_manager = TrainingProcessManager()
