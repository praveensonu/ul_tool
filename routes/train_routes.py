from fastapi import APIRouter, HTTPException

from schemas import FinalTrainingConfigRequest
from config.training_config import build_orchestrator_config
from orchestrator import run_orchestrator

router = APIRouter(prefix="/train", tags=["Training"])


@router.post("/run")
def run_training(request: FinalTrainingConfigRequest):
    try:
        orchestrator_config = build_orchestrator_config(request)

        result = run_orchestrator(orchestrator_config)

        return {
            "status": "success",
            "orchestrator_config": orchestrator_config,
            "result": result,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))