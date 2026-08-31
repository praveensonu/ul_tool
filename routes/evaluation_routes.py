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
)
from training_process import training_process_manager


router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


def _ensure_training_is_idle() -> None:
    if training_process_manager.is_running:
        raise HTTPException(
            status_code=409,
            detail="Evaluation cannot start while training is active.",
        )


@router.post("/start", response_model=EvaluationStartResponse)
def start_evaluation(request: EvaluationRequest):
    _ensure_training_is_idle()
    try:
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


@router.post("/run", response_model=EvaluationResponse)
def run_evaluation(request: EvaluationRequest):
    _ensure_training_is_idle()

    try:
        return evaluation_process_manager.run(request.model_dump())
    except EvaluationAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EvaluationProcessError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
