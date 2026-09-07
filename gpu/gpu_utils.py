from __future__ import annotations

import os
import subprocess
from typing import Iterable


GPU_QUERY_FIELDS = (
    "index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu"
)


def _run_nvidia_smi(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi is not installed or is not on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "nvidia-smi failed."
        raise RuntimeError(detail) from exc
    return result.stdout


def get_gpu_info() -> list[dict]:
    """Return physical GPU capacity and whether a compute process is using it."""

    output = _run_nvidia_smi(
        f"--query-gpu={GPU_QUERY_FIELDS}",
        "--format=csv,noheader,nounits",
    )
    active_uuids: set[str] = set()
    try:
        process_output = _run_nvidia_smi(
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        )
        active_uuids = {
            value.strip()
            for value in process_output.splitlines()
            if value.strip() and value.strip().lower() != "[not supported]"
        }
    except RuntimeError:
        # Some drivers do not expose compute-app queries. Capacity data remains
        # useful, and utilization/memory provide a conservative busy fallback.
        active_uuids = set()

    devices: list[dict] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 6)]
        if len(parts) != 7:
            raise RuntimeError(f"Unexpected nvidia-smi GPU row: {line}")
        index, uuid, name, total, used, free, utilization = parts
        memory_used = int(used)
        utilization_percent = int(utilization)
        has_compute_process = uuid in active_uuids
        is_available = (
            not has_compute_process
            and utilization_percent <= 5
            and memory_used <= 512
        )
        devices.append(
            {
                "id": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mb": int(total),
                "memory_used_mb": memory_used,
                "memory_free_mb": int(free),
                "utilization_percent": utilization_percent,
                "is_available": is_available,
                "status": "available" if is_available else "busy",
            }
        )
    return devices


def get_available_gpu_ids() -> list[int]:
    return [device["id"] for device in get_gpu_info() if device["is_available"]]


def validate_gpu_ids(gpu_ids: Iterable[int], *, require_available: bool = False) -> list[int]:
    selected = list(gpu_ids)
    if not selected:
        raise ValueError("Select at least one GPU.")
    if len(selected) != len(set(selected)):
        raise ValueError("Selected GPU ids must be unique.")

    devices = get_gpu_info()
    known = {device["id"] for device in devices}
    invalid = [gpu_id for gpu_id in selected if gpu_id not in known]
    if invalid:
        raise ValueError(
            f"Invalid GPU ids {invalid}. Installed GPU ids are: {sorted(known)}"
        )
    if require_available:
        busy = [d["id"] for d in devices if d["id"] in selected and not d["is_available"]]
        if busy:
            raise ValueError(f"Selected GPUs are busy: {busy}. Refresh and select free GPUs.")
    return selected


def validate_gpu_id(gpu_id: int) -> None:
    validate_gpu_ids([gpu_id])


def set_cuda_visible_devices(gpu_ids: Iterable[int]) -> str:
    selected = list(gpu_ids)
    if not selected:
        raise ValueError("Select at least one GPU.")
    value = ",".join(str(gpu_id) for gpu_id in selected)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    return value


def selected_gpu_ids(config: dict) -> list[int]:
    """Normalize ordered physical IDs, including legacy single-GPU configs."""
    selected = config.get("gpu_ids")
    if selected is None:
        selected = [config["gpu_id"]] if "gpu_id" in config else []
    if not isinstance(selected, list) or not selected:
        raise ValueError("Select at least one GPU.")
    if any(type(value) is not int or value < 0 for value in selected):
        raise ValueError("GPU ids must be non-negative integers.")
    if len(selected) != len(set(selected)):
        raise ValueError("Selected GPU ids must be unique.")
    return selected
