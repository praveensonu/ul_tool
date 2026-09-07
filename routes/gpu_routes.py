from fastapi import APIRouter, HTTPException

from gpu.gpu_utils import get_gpu_info
from schemas import GpuListResponse


router = APIRouter(prefix="/gpus", tags=["GPUs"])


@router.get("", response_model=GpuListResponse)
def list_gpus():
    try:
        devices = get_gpu_info()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "gpus": devices,
        "available_gpu_ids": [
            device["id"] for device in devices if device["is_available"]
        ],
    }
