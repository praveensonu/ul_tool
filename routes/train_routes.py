from fastapi import APIRouter, HTTPException

from schemas import (
    FinalTrainingConfigRequest,
    TrainRunResponse,
    TrainStopResponse,
)
from config.training_config import build_orchestrator_config
from training_process import (
    TrainingAlreadyRunningError,
    TrainingProcessError,
    training_process_manager,
)
from evaluation_process import evaluation_process_manager
from data_selection.caching import gradient_cache_manager

router = APIRouter(prefix="/train", tags=["Training"])


@router.post("/run", response_model=TrainRunResponse)
def run_training(request: FinalTrainingConfigRequest):
    if gradient_cache_manager.is_running:
        raise HTTPException(
            status_code=409,
            detail="Training cannot start while RASLIK gradient caching is active.",
        )
    if evaluation_process_manager.is_running:
        raise HTTPException(
            status_code=409,
            detail="Training cannot start while evaluation is active.",
        )

    try:
        from gpu.gpu_utils import validate_gpu_ids
        validate_gpu_ids(request.gpu_ids)
        orchestrator_config = build_orchestrator_config(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        outcome = training_process_manager.run(orchestrator_config)
    except TrainingAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except TrainingProcessError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if outcome.status == "stopped":
        return {
            "status": "stopped",
            "orchestrator_config": orchestrator_config,
            "result": None,
            "message": "Training was stopped before completion.",
        }

    return {
        "status": "success",
        "orchestrator_config": orchestrator_config,
        "result": outcome.result,
    }


@router.post("/stop", response_model=TrainStopResponse)
def stop_training():
    try:
        stopped = training_process_manager.stop()
    except TrainingProcessError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not stopped:
        return {
            "status": "idle",
            "message": "No training process is currently running.",
        }

    return {
        "status": "stopped",
        "message": "Training process terminated.",
    }
