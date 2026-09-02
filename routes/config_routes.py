from fastapi import APIRouter, HTTPException

from schemas import (
    FinalTrainingConfigRequest,
    FinalTrainingConfigResponse,
    UnlearningMethodsResponse,
)
from config.training_config import build_orchestrator_config
from unlearning import methods as registry


router = APIRouter(prefix="/config", tags=["Training Config"])


@router.get("/unlearning-methods", response_model=UnlearningMethodsResponse)
def list_unlearning_methods():
    return UnlearningMethodsResponse(
        methods=[
            {
                "value": name,
                "label": spec.label,
                "requires_retain": spec.requires_retain,
            }
            for name, spec in registry.UNLEARNING_METHODS.items()
        ],
        default=registry.DEFAULT_UNLEARNING_METHOD,
    )


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