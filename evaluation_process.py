from __future__ import annotations

import copy
import inspect
import multiprocessing
import queue
import threading
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from eval_orchestrator import run_eval_orchestrator


class EvaluationAlreadyRunningError(RuntimeError):
    pass


class EvaluationProcessError(RuntimeError):
    pass


class EvaluationCancelledError(RuntimeError):
    pass


def _supports_progress_callback(runner: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(runner).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "progress_callback"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _evaluation_worker(
    api_config: Dict[str, Any],
    result_queue: Any,
    progress_queue: Any,
    cancel_event: Any,
    runner: Callable[..., Dict[str, Any]],
) -> None:
    def report(stage: str, message: str) -> None:
        if cancel_event.is_set():
            raise EvaluationCancelledError("Evaluation cancelled by the user.")
        progress_queue.put({"stage": stage, "message": message})

    try:
        report("starting", "Evaluation process started.")
        if _supports_progress_callback(runner):
            result = runner(api_config, progress_callback=report)
        else:
            result = runner(api_config)
        if cancel_event.is_set():
            raise EvaluationCancelledError("Evaluation cancelled by the user.")
        result_queue.put({"status": "success", "result": result})
    except EvaluationCancelledError as exc:
        result_queue.put({"status": "cancelled", "message": str(exc)})
    except Exception as exc:
        if cancel_event.is_set():
            result_queue.put(
                {
                    "status": "cancelled",
                    "message": "Evaluation cancelled by the user.",
                }
            )
            return
        traceback.print_exc()
        progress_queue.put({"stage": "failed", "message": str(exc)})
        result_queue.put(
            {
                "status": "error",
                "error": str(exc),
                "exception_type": type(exc).__name__,
            }
        )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvaluationProcessManager:
    """Own one background evaluation process and its progress history."""

    def __init__(
        self,
        runner: Callable[..., Dict[str, Any]] = run_eval_orchestrator,
    ) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._runner = runner
        self._condition = threading.Condition()
        self._process: Optional[multiprocessing.Process] = None
        self._cancel_event: Any = None
        self._active_job_id: Optional[str] = None
        self._jobs: Dict[str, Dict[str, Any]] = {}

    @property
    def is_running(self) -> bool:
        with self._condition:
            return self._active_job_id is not None

    def _append_progress(self, job_id: str, event: Dict[str, str]) -> None:
        with self._condition:
            job = self._jobs[job_id]
            update = {
                "stage": event["stage"],
                "message": event["message"],
                "timestamp": _timestamp(),
            }
            job["current_stage"] = update["stage"]
            job["message"] = update["message"]
            job["progress"].append(update)
            self._condition.notify_all()

    def start(self, api_config: Dict[str, Any]) -> Dict[str, Any]:
        result_queue = self._context.Queue()
        progress_queue = self._context.Queue()
        cancel_event = self._context.Event()
        job_id = uuid.uuid4().hex

        with self._condition:
            if self._active_job_id is not None:
                result_queue.close()
                progress_queue.close()
                raise EvaluationAlreadyRunningError(
                    "An evaluation run is already active."
                )

            process = self._context.Process(
                target=_evaluation_worker,
                args=(
                    api_config,
                    result_queue,
                    progress_queue,
                    cancel_event,
                    self._runner,
                ),
                name=f"ascent-evaluation-{job_id[:8]}",
            )
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "current_stage": "queued",
                "message": "Evaluation is queued.",
                "progress": [],
                "result": None,
                "error": None,
            }
            self._active_job_id = job_id
            self._process = process
            self._cancel_event = cancel_event
            try:
                process.start()
            except Exception:
                self._active_job_id = None
                self._process = None
                self._cancel_event = None
                self._jobs.pop(job_id, None)
                result_queue.close()
                progress_queue.close()
                raise

            self._jobs[job_id]["status"] = "running"
            monitor = threading.Thread(
                target=self._monitor,
                args=(job_id, process, result_queue, progress_queue),
                name=f"evaluation-monitor-{job_id[:8]}",
                daemon=True,
            )
            monitor.start()

            return {
                "job_id": job_id,
                "status": "running",
                "message": "Evaluation started.",
            }

    def _monitor(
        self,
        job_id: str,
        process: multiprocessing.Process,
        result_queue: Any,
        progress_queue: Any,
    ) -> None:
        while process.is_alive():
            try:
                event = progress_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._append_progress(job_id, event)

        process.join()
        while True:
            try:
                event = progress_queue.get_nowait()
            except queue.Empty:
                break
            self._append_progress(job_id, event)

        try:
            message = result_queue.get(timeout=1.0)
        except queue.Empty:
            message = {
                "status": "error",
                "exception_type": "EvaluationProcessError",
                "error": (
                    "Evaluation process exited unexpectedly with "
                    f"code {process.exitcode}."
                ),
            }

        with self._condition:
            job = self._jobs[job_id]
            cancel_requested = job["status"] == "cancelling"
            if message.get("status") == "success" and not cancel_requested:
                job["status"] = "completed"
                job["current_stage"] = "completed"
                job["message"] = message["result"].get(
                    "message", "Evaluation completed."
                )
                job["result"] = message["result"]
            elif message.get("status") == "cancelled" or cancel_requested:
                job["status"] = "cancelled"
                job["current_stage"] = "cancelled"
                job["message"] = message.get(
                    "message", "Evaluation cancelled by the user."
                )
            else:
                exception_type = message.get(
                    "exception_type", "EvaluationError"
                )
                error = message.get(
                    "error", "Evaluation failed without an error message."
                )
                job["status"] = "failed"
                job["current_stage"] = "failed"
                job["message"] = error
                job["error"] = f"{exception_type}: {error}"

            if self._active_job_id == job_id:
                self._active_job_id = None
                self._process = None
                self._cancel_event = None
            self._condition.notify_all()

        result_queue.close()
        progress_queue.close()
        process.close()

    def get_status(self, job_id: str) -> Dict[str, Any]:
        with self._condition:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return copy.deepcopy(self._jobs[job_id])

    def cancel(self, job_id: str) -> bool:
        """Request cooperative cancellation at the next progress boundary."""

        with self._condition:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
            if job["status"] in {"cancelled", "completed", "failed"}:
                return False
            if self._active_job_id != job_id or self._cancel_event is None:
                return False
            self._cancel_event.set()
            job["status"] = "cancelling"
            job["current_stage"] = "cancelling"
            job["message"] = (
                "Cancellation requested. Finishing the current operation and "
                "releasing loaded models."
            )
            job["progress"].append(
                {
                    "stage": "cancelling",
                    "message": job["message"],
                    "timestamp": _timestamp(),
                }
            )
            self._condition.notify_all()
            return True

    def run(self, api_config: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility helper for clients that still use blocking `/run`."""

        job_id = self.start(api_config)["job_id"]
        with self._condition:
            self._condition.wait_for(
                lambda: self._jobs[job_id]["status"]
                in {"cancelled", "completed", "failed"}
            )
            job = self._jobs[job_id]
            if job["status"] in {"cancelled", "failed"}:
                raise EvaluationProcessError(job["error"] or job["message"])
            return copy.deepcopy(job["result"])


evaluation_process_manager = EvaluationProcessManager()
