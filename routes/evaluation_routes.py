from fastapi import APIRouter, HTTPException

from evaluation_process import (
    EvaluationAlreadyRunningError,
    EvaluationProcessError,
    evaluation_process_manager,
)
from schemas import (
    EvaluationRequest,
    EvaluationResponse,
    EvaluationStartResponse,
    EvaluationStatusResponse,
    JobCancelResponse,
)
from training_process import training_process_manager
from data_selection.caching import gradient_cache_manager


router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


def _validate_gpus(request: EvaluationRequest) -> None:
    from gpu.gpu_utils import validate_gpu_ids
    try:
        validate_gpu_ids(request.orchestrator_config["gpu"]["gpu_ids"])
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _ensure_gpu_workloads_are_idle() -> None:
    if gradient_cache_manager.is_running:
        raise HTTPException(
            status_code=409,
            detail="Evaluation cannot start while RASLIK gradient caching is active.",
        )
    if training_process_manager.is_running:
        raise HTTPException(
            status_code=409,
            detail="Evaluation cannot start while training is active.",
        )


@router.post("/start", response_model=EvaluationStartResponse)
def start_evaluation(request: EvaluationRequest):
    _ensure_gpu_workloads_are_idle()
    try:
        _validate_gpus(request)
        return evaluation_process_manager.start(request.model_dump())
    except EvaluationAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/status/{job_id}", response_model=EvaluationStatusResponse)
def evaluation_status(job_id: str):
    try:
        return evaluation_process_manager.get_status(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Evaluation job not found: {job_id}"
        ) from exc


@router.post("/cancel/{job_id}", response_model=JobCancelResponse)
def cancel_evaluation(job_id: str):
    try:
        requested = evaluation_process_manager.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Evaluation job not found: {job_id}"
        ) from exc

    if not requested:
        return {
            "status": "idle",
            "message": "Evaluation is no longer running.",
        }
    return {
        "status": "cancelling",
        "message": (
            "Evaluation cancellation requested. The current operation will "
            "finish before loaded models are released."
        ),
    }


@router.post("/run", response_model=EvaluationResponse)
def run_evaluation(request: EvaluationRequest):
    _ensure_gpu_workloads_are_idle()

    try:
        _validate_gpus(request)
        return evaluation_process_manager.run(request.model_dump())
    except EvaluationAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EvaluationProcessError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
