from __future__ import annotations

import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import UploadFile

from dataset.dataset_loader import (
    dataframe_preview,
    prepare_raslik_dataframe,
    read_file,
    save_jsonl_dataframe,
    save_upload_file,
)
from data_selection.selection import (
    select_and_export_datasets,
    validate_selection_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MP_MAIN_PATH = PROJECT_ROOT / "data_selection" / "MP_main.py"
GRADIENTS_ROOT = PROJECT_ROOT / "outputs" / "gradients"
RASLIK_UPLOAD_ROOT = PROJECT_ROOT / "uploaded_datasets" / "raslik"


class GradientCacheAlreadyRunningError(RuntimeError):
    pass


class GradientCacheError(RuntimeError):
    pass


class ExtractionCancelledError(RuntimeError):
    pass


CommandRunner = Callable[[Path, Path], None]
DatasetValidator = Callable[[int], None]
ProgressCallback = Callable[[str, str], None]


def _path_for_config(path: Path) -> str:
    """Use stable project-relative paths in generated, user-visible configs."""

    return path.relative_to(PROJECT_ROOT).as_posix()


def _experiment_name(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("._-")[:80]
    if not base:
        base = "raslik"
    return f"{base}-{uuid.uuid4().hex[:8]}"


def build_caching_config(
    *,
    train_data_path: Path,
    grads_path: Path,
    outdir: Path,
    model_path: str,
    lora_path: str | None,
    max_length: int,
) -> dict:
    """Build the caching-only RASLIK config from the checked-in example."""

    return {
        "data": {
            "train_data_path": _path_for_config(train_data_path),
        },
        "influence": {
            "outdir": _path_for_config(outdir),
            "seed": 42,
            "cal_words_infl": False,
            "save_to_grads_path": True,
            "n_threads": 1,
            "RapidGrad": {
                "enable": True,
                "RapidGrad_K": 65536,
                "shuffle_lambda": 20,
            },
            "offload_train_grad": False,
            "skip_test": True,
            "skip_influence": True,
            "grads_path": _path_for_config(grads_path),
        },
        "model": {
            "model_path": model_path,
            "max_length": max_length,
            "load_in_4bit": False,
            "lora_path": lora_path,
        },
        "postprocess": {
            "enable": False,
        },
    }


def _write_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")


def run_mp_main(
    config_path: Path,
    log_path: Path,
    cancel_event: threading.Event | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, str(MP_MAIN_PATH), "--config_path", str(config_path)],
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        while process.poll() is None:
            if cancel_event is not None and cancel_event.wait(0.2):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                raise ExtractionCancelledError(
                    "Dataset extraction cancelled by the user."
                )
            time.sleep(0.05)

    if process.returncode != 0:
        log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        detail = "\n".join(log_lines[-20:])
        raise GradientCacheError(
            f"RASLIK exited with code {process.returncode}. See {log_path}.\n{detail}"
        )


def _report(
    callback: ProgressCallback | None, stage: str, message: str
) -> None:
    if callback is not None:
        callback(stage, message)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ExtractionCancelledError("Dataset extraction cancelled by the user.")


def prepare_uploaded_datasets(
    *,
    full_dataset: UploadFile,
    poison_set: UploadFile,
    prompt_template: str,
    experiment_name: str,
    model_name: str,
    max_length: int,
    adaptor_path: str | None,
    dataset_validator: DatasetValidator | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    if not model_name.strip():
        raise ValueError("model_name must not be empty.")
    if max_length < 1:
        raise ValueError("max_length must be at least 1.")

    run_name = _experiment_name(experiment_name)
    gradient_root = GRADIENTS_ROOT / run_name
    input_root = RASLIK_UPLOAD_ROOT / run_name
    config_root = gradient_root / "configs"
    metadata_root = gradient_root / "run_metadata"

    _report(
        progress_callback,
        "preparing_datasets",
        "Validating and normalizing the full dataset and poison set.",
    )
    prepared_datasets: dict[str, tuple[Path, int]] = {}
    for label, upload in (("training", full_dataset), ("poison", poison_set)):
        raw_path = save_upload_file(upload)
        dataframe = read_file(raw_path)
        prepared = prepare_raslik_dataframe(
            dataframe,
            prompt_template=prompt_template,
            dataset_name="full dataset" if label == "training" else "poison set",
        )
        dataset_path = input_root / f"{label}.jsonl"
        save_jsonl_dataframe(prepared, dataset_path)
        prepared_datasets[label] = (dataset_path, len(prepared))

    if dataset_validator is not None:
        dataset_validator(prepared_datasets["training"][1])

    _report(
        progress_callback,
        "datasets_prepared",
        "Datasets normalized to JSONL and selection parameters validated.",
    )

    config_paths: dict[str, Path] = {}
    log_paths: dict[str, Path] = {}
    for label in ("training", "poison"):
        grads_path = gradient_root / label
        grads_path.mkdir(parents=True, exist_ok=False)
        config = build_caching_config(
            train_data_path=prepared_datasets[label][0],
            grads_path=grads_path,
            outdir=metadata_root / label,
            model_path=model_name.strip(),
            lora_path=adaptor_path.strip() if adaptor_path and adaptor_path.strip() else None,
            max_length=max_length,
        )
        config_path = config_root / f"{label}.json"
        log_path = config_root / f"{label}.log"
        _write_config(config, config_path)
        config_paths[label] = config_path
        log_paths[label] = log_path

    return {
        "run_name": run_name,
        "gradient_root": gradient_root,
        "prepared_datasets": prepared_datasets,
        "config_paths": config_paths,
        "log_paths": log_paths,
        "prompt_template": prompt_template,
    }


def cache_prepared_gradients(
    prepared_run: dict[str, Any],
    *,
    runner: CommandRunner = run_mp_main,
    cancel_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    run_name = prepared_run["run_name"]
    gradient_root = prepared_run["gradient_root"]
    prepared_datasets = prepared_run["prepared_datasets"]
    config_paths = prepared_run["config_paths"]
    log_paths = prepared_run["log_paths"]

    # The original RASLIK workflow caches the full training set first and then
    # repeats the same operation for the poison set.
    for label in ("training", "poison"):
        _raise_if_cancelled(cancel_event)
        display_label = "full training dataset" if label == "training" else "poison set"
        _report(
            progress_callback,
            f"caching_{label}_gradients",
            f"Calculating and compressing gradients for the {display_label}.",
        )
        if runner is run_mp_main:
            run_mp_main(
                config_paths[label],
                log_paths[label],
                cancel_event=cancel_event,
            )
        else:
            runner(config_paths[label], log_paths[label])
        _report(
            progress_callback,
            f"cached_{label}_gradients",
            f"Stored compressed gradients for the {display_label}.",
        )

    return {
        "status": "success",
        "experiment_name": run_name,
        "training_grads_path": _path_for_config(gradient_root / "training"),
        "poison_grads_path": _path_for_config(gradient_root / "poison"),
        "training_data_path": _path_for_config(prepared_datasets["training"][0]),
        "poison_data_path": _path_for_config(prepared_datasets["poison"][0]),
        "training_config_path": _path_for_config(config_paths["training"]),
        "poison_config_path": _path_for_config(config_paths["poison"]),
        "training_rows": prepared_datasets["training"][1],
        "poison_rows": prepared_datasets["poison"][1],
        "message": "Cached compressed gradients for the full dataset and poison set.",
    }


def cache_uploaded_gradients(
    *,
    runner: CommandRunner = run_mp_main,
    dataset_validator: DatasetValidator | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
    **prepare_kwargs,
) -> dict:
    prepared_run = prepare_uploaded_datasets(
        dataset_validator=dataset_validator,
        progress_callback=progress_callback,
        **prepare_kwargs,
    )
    return cache_prepared_gradients(
        prepared_run,
        runner=runner,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )


def _remove_gradient_tensors(cache_result: dict) -> None:
    for key in ("training_grads_path", "poison_grads_path"):
        path = PROJECT_ROOT / cache_result[key]
        if path.is_dir():
            shutil.rmtree(path)


def extract_uploaded_datasets(
    *,
    selection_method: str,
    forget_size: int,
    retain_size: int,
    grace_top_n: int | None,
    grace_num_clusters: int | None,
    keep_gradients: bool = False,
    runner: CommandRunner = run_mp_main,
    cancel_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
    **cache_kwargs,
) -> dict:
    def validate_dataset_size(dataset_size: int) -> None:
        validate_selection_parameters(
            method=selection_method,
            dataset_size=dataset_size,
            forget_size=forget_size,
            retain_size=retain_size,
            top_n=grace_top_n,
            num_clusters=grace_num_clusters,
        )

    prepared_run = prepare_uploaded_datasets(
        dataset_validator=validate_dataset_size,
        progress_callback=progress_callback,
        **cache_kwargs,
    )
    return extract_prepared_datasets(
        prepared_run,
        selection_method=selection_method,
        forget_size=forget_size,
        retain_size=retain_size,
        grace_top_n=grace_top_n,
        grace_num_clusters=grace_num_clusters,
        keep_gradients=keep_gradients,
        runner=runner,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )


def extract_prepared_datasets(
    prepared_run: dict[str, Any],
    *,
    selection_method: str,
    forget_size: int,
    retain_size: int,
    grace_top_n: int | None,
    grace_num_clusters: int | None,
    keep_gradients: bool,
    runner: CommandRunner = run_mp_main,
    cancel_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    cache_result: dict | None = None
    try:
        cache_result = cache_prepared_gradients(
            prepared_run,
            runner=runner,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        _raise_if_cancelled(cancel_event)
        _report(
            progress_callback,
            "selecting_datasets",
            f"Selecting forget and retain samples with {selection_method.upper()}.",
        )
        run_name = cache_result["experiment_name"]
        selection = select_and_export_datasets(
            method=selection_method,
            training_grads_path=PROJECT_ROOT / cache_result["training_grads_path"],
            poison_grads_path=PROJECT_ROOT / cache_result["poison_grads_path"],
            training_data_path=PROJECT_ROOT / cache_result["training_data_path"],
            output_dir=GRADIENTS_ROOT / run_name / "selected" / selection_method,
            forget_size=forget_size,
            retain_size=retain_size,
            top_n=grace_top_n,
            num_clusters=grace_num_clusters,
        )
        _raise_if_cancelled(cancel_event)

        forget_dataframe = selection.pop("forget_dataframe")
        retain_dataframe = selection.pop("retain_dataframe")
        if keep_gradients:
            _report(
                progress_callback,
                "retaining_gradients",
                "Keeping cached training and poison gradients.",
            )
        else:
            _report(
                progress_callback,
                "removing_gradients",
                "Removing cached training and poison gradients.",
            )
            _remove_gradient_tensors(cache_result)
            average_gradient_path = (
                Path(selection["metadata_path"]).parent
                / "average_poison_gradient.pt"
            )
            average_gradient_path.unlink(missing_ok=True)
        _raise_if_cancelled(cancel_event)

        result = {
            **cache_result,
            "selection_method": selection_method,
            "forget_set_path": _path_for_config(selection["forget_path"]),
            "retain_set_path": _path_for_config(selection["retain_path"]),
            "selection_metadata_path": _path_for_config(selection["metadata_path"]),
            "forget_rows": len(forget_dataframe),
            "retain_rows": len(retain_dataframe),
            "has_retain_set": True,
            "prompt_template": prepared_run["prompt_template"],
            "forget_preview": dataframe_preview(forget_dataframe),
            "retain_preview": dataframe_preview(retain_dataframe),
            "gradients_retained": keep_gradients,
            "message": (
                f"Extracted {len(forget_dataframe)} forget and "
                f"{len(retain_dataframe)} retain samples with "
                f"{selection_method.upper()}. Cached gradients were "
                f"{'kept' if keep_gradients else 'removed'}."
            ),
        }
        _report(progress_callback, "completed", result["message"])
        return result
    except ExtractionCancelledError:
        if cache_result is not None:
            _remove_gradient_tensors(cache_result)
        else:
            for label in ("training", "poison"):
                path = prepared_run["gradient_root"] / label
                if path.is_dir():
                    shutil.rmtree(path)
        selected_path = (
            prepared_run["gradient_root"] / "selected" / selection_method
        )
        if selected_path.is_dir():
            shutil.rmtree(selected_path)
        raise


class GradientCacheManager:
    """Serialize cache work and expose background extraction job state."""

    def __init__(self, runner: CommandRunner = run_mp_main) -> None:
        self._runner = runner
        self._operation_lock = threading.Lock()
        self._condition = threading.Condition()
        self._active_job_id: str | None = None
        self._cancel_event: threading.Event | None = None
        self._jobs: dict[str, dict[str, Any]] = {}

    @property
    def is_running(self) -> bool:
        return self._operation_lock.locked()

    def _append_progress(self, job_id: str, stage: str, message: str) -> None:
        with self._condition:
            job = self._jobs[job_id]
            event = {
                "stage": stage,
                "message": message,
                "timestamp": _timestamp(),
            }
            job["current_stage"] = stage
            job["message"] = message
            job["progress"].append(event)
            self._condition.notify_all()

    def run(self, **kwargs) -> dict:
        if not self._operation_lock.acquire(blocking=False):
            raise GradientCacheAlreadyRunningError(
                "A RASLIK gradient-caching run is already active."
            )
        try:
            return cache_uploaded_gradients(runner=self._runner, **kwargs)
        finally:
            self._operation_lock.release()

    def extract(self, **kwargs) -> dict:
        if not self._operation_lock.acquire(blocking=False):
            raise GradientCacheAlreadyRunningError(
                "A dataset-selection run is already active."
            )
        try:
            return extract_uploaded_datasets(runner=self._runner, **kwargs)
        finally:
            self._operation_lock.release()

    def start_extract(self, **kwargs) -> dict:
        if not self._operation_lock.acquire(blocking=False):
            raise GradientCacheAlreadyRunningError(
                "A dataset-selection run is already active."
            )

        job_id = uuid.uuid4().hex
        cancel_event = threading.Event()
        with self._condition:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "current_stage": "queued",
                "message": "Dataset extraction is queued.",
                "progress": [],
                "result": None,
                "error": None,
            }
            self._active_job_id = job_id
            self._cancel_event = cancel_event

        selection_args = {
            key: kwargs.pop(key)
            for key in (
                "selection_method",
                "forget_size",
                "retain_size",
                "grace_top_n",
                "grace_num_clusters",
                "keep_gradients",
            )
        }

        def validate_dataset_size(dataset_size: int) -> None:
            validate_selection_parameters(
                method=selection_args["selection_method"],
                dataset_size=dataset_size,
                forget_size=selection_args["forget_size"],
                retain_size=selection_args["retain_size"],
                top_n=selection_args["grace_top_n"],
                num_clusters=selection_args["grace_num_clusters"],
            )

        try:
            prepared_run = prepare_uploaded_datasets(
                dataset_validator=validate_dataset_size,
                progress_callback=lambda stage, message: self._append_progress(
                    job_id, stage, message
                ),
                **kwargs,
            )
            with self._condition:
                self._jobs[job_id]["status"] = "running"
            worker = threading.Thread(
                target=self._run_extraction_job,
                args=(job_id, prepared_run, selection_args, cancel_event),
                name=f"extraction-worker-{job_id[:8]}",
                daemon=True,
            )
            worker.start()
        except Exception:
            with self._condition:
                self._active_job_id = None
                self._cancel_event = None
                self._jobs.pop(job_id, None)
            self._operation_lock.release()
            raise

        return {
            "job_id": job_id,
            "status": "running",
            "message": "Dataset extraction started.",
        }

    def _run_extraction_job(
        self,
        job_id: str,
        prepared_run: dict[str, Any],
        selection_args: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        try:
            result = extract_prepared_datasets(
                prepared_run,
                runner=self._runner,
                cancel_event=cancel_event,
                progress_callback=lambda stage, message: self._append_progress(
                    job_id, stage, message
                ),
                **selection_args,
            )
            with self._condition:
                job = self._jobs[job_id]
                job["status"] = "completed"
                job["current_stage"] = "completed"
                job["message"] = result["message"]
                job["result"] = result
        except ExtractionCancelledError as exc:
            with self._condition:
                job = self._jobs[job_id]
                job["status"] = "cancelled"
                job["current_stage"] = "cancelled"
                job["message"] = str(exc)
        except Exception as exc:
            with self._condition:
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["current_stage"] = "failed"
                job["message"] = str(exc)
                job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self._condition:
                if self._active_job_id == job_id:
                    self._active_job_id = None
                    self._cancel_event = None
                self._condition.notify_all()
            self._operation_lock.release()

    def get_status(self, job_id: str) -> dict:
        with self._condition:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return copy.deepcopy(self._jobs[job_id])

    def cancel(self, job_id: str) -> bool:
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
            message = "Cancellation requested. Stopping gradient caching safely."
            self._append_progress(job_id, "cancelling", message)
            return True


gradient_cache_manager = GradientCacheManager()
