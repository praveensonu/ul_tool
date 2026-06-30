from fastapi import APIRouter, HTTPException

from schemas import FinalTrainingConfigRequest, FinalTrainingConfigResponse
from config.training_config import build_orchestrator_config


router = APIRouter(prefix="/config", tags=["Training Config"])


@router.post("/build", response_model=FinalTrainingConfigResponse)
def build_config(request: FinalTrainingConfigRequest):
    try:
        orchestrator_config = build_orchestrator_config(request)

        return FinalTrainingConfigResponse(
            status="success",
            orchestrator_config=orchestrator_config,
            message="Orchestrator config created successfully.",
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))