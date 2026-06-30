from fastapi import APIRouter, HTTPException

from schemas import LoadModelRequest, LoadModelResponse
from model.model_loader import load_model_by_method, store_loaded_model

router = APIRouter(prefix="/models", tags=["Models"])


@router.post("/load", response_model=LoadModelResponse)
def load_model(request: LoadModelRequest):
    try:
        model, tokenizer, device, merged = load_model_by_method(
            method=request.method.value,
            base_model_path=request.base_model_path,
            adaptor_path=request.adaptor_path,
            gpu_id=request.gpu_id,
            hf_key=request.hf_key,
        )

        model_key = f"{request.base_model_path}:{request.method.value}:gpu{request.gpu_id}"

        store_loaded_model(
            model_key=model_key,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        return LoadModelResponse(
            status="success",
            method=request.method.value,
            base_model_path=request.base_model_path,
            adaptor_path=request.adaptor_path,
            gpu_id=request.gpu_id,
            device=device,
            merged=merged,
            message=f"Model loaded successfully with key: {model_key}",
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))