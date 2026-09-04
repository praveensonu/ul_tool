from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from schemas import (
    DataSelectionMethod,
    DatasetExtractionResponse,
    DatasetUploadResponse,
    ExtractionStartResponse,
    ExtractionStatusResponse,
    GradientCacheResponse,
    JobCancelResponse,
)
from dataset.dataset_loader import (
    process_dataset,
    dataframe_preview,
)
from data_selection.caching import (
    GradientCacheAlreadyRunningError,
    GradientCacheError,
    gradient_cache_manager,
)
from evaluation_process import evaluation_process_manager
from training_process import training_process_manager


router = APIRouter(prefix="/dataset", tags=["Dataset"])


@router.post("/upload", response_model=DatasetUploadResponse)
def upload_dataset_config(
    forget_set: UploadFile = File(...),
    retain_set: Optional[UploadFile] = File(None),
    prompt_template: str = Form(...),
):
    try:
        if retain_set is not None and forget_set is None:
            raise ValueError("retain_set cannot be uploaded without forget_set.")

        forget_df, forget_path = process_dataset(
            file=forget_set,
            prompt_template=prompt_template,
            dataset_name="forget_set",
        )

        retain_df = None
        retain_path = None

        if retain_set is not None:
            retain_df, retain_path = process_dataset(
                file=retain_set,
                prompt_template=prompt_template,
                dataset_name="retain_set",
            )

        training_config = {
            "forget_set_path": forget_path,
            "retain_set_path": retain_path,
            "prompt_template": prompt_template,
            "forget_num_rows": len(forget_df),
            "retain_num_rows": len(retain_df) if retain_df is not None else None,
        }

        return DatasetUploadResponse(
            status="success",
            forget_set_path=forget_path,
            retain_set_path=retain_path,
            forget_rows=len(forget_df),
            retain_rows=len(retain_df) if retain_df is not None else None,
            has_retain_set=retain_df is not None,
            prompt_template=prompt_template,
            forget_preview=dataframe_preview(forget_df),
            retain_preview=dataframe_preview(retain_df) if retain_df is not None else None,
            message=f"Dataset config created successfully: {training_config}",
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/cache-gradients", response_model=GradientCacheResponse)
def cache_gradients(
    full_dataset: UploadFile = File(...),
    poison_set: UploadFile = File(...),
    prompt_template: str = Form(...),
    experiment_name: str = Form("raslik"),
    model_name: str = Form(...),
    max_length: int = Form(..., ge=1),
    adaptor_path: Optional[str] = Form(None),
):
    if training_process_manager.is_running or evaluation_process_manager.is_running:
        raise HTTPException(
            status_code=409,
            detail=(
                "Gradient caching cannot start while training or evaluation is active."
            ),
        )

    try:
        return gradient_cache_manager.run(
            full_dataset=full_dataset,
            poison_set=poison_set,
            prompt_template=prompt_template,
            experiment_name=experiment_name,
            model_name=model_name,
            max_length=max_length,
            adaptor_path=adaptor_path,
        )
    except GradientCacheAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (GradientCacheError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not prepare datasets for RASLIK: {exc}",
        ) from exc


@router.post("/extract", response_model=DatasetExtractionResponse)
def extract_forget_retain(
    full_dataset: UploadFile = File(...),
    poison_set: UploadFile = File(...),
    prompt_template: str = Form(...),
    experiment_name: str = Form("selection"),
    model_name: str = Form(...),
    max_length: int = Form(..., ge=1),
    adaptor_path: Optional[str] = Form(None),
    selection_method: DataSelectionMethod = Form(...),
    forget_size: int = Form(..., ge=1),
    retain_size: int = Form(..., ge=1),
    grace_top_n: Optional[int] = Form(None, ge=1),
    grace_num_clusters: Optional[int] = Form(None, ge=1),
    keep_gradients: bool = Form(False),
):
    if training_process_manager.is_running or evaluation_process_manager.is_running:
        raise HTTPException(
            status_code=409,
            detail=(
                "Dataset selection cannot start while training or evaluation is active."
            ),
        )

    try:
        return gradient_cache_manager.extract(
            full_dataset=full_dataset,
            poison_set=poison_set,
            prompt_template=prompt_template,
            experiment_name=experiment_name,
            model_name=model_name,
            max_length=max_length,
            adaptor_path=adaptor_path,
            selection_method=selection_method.value,
            forget_size=forget_size,
            retain_size=retain_size,
            grace_top_n=grace_top_n,
            grace_num_clusters=grace_num_clusters,
            keep_gradients=keep_gradients,
        )
    except GradientCacheAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (GradientCacheError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not extract forget/retain datasets: {exc}",
        ) from exc


@router.post("/extract/start", response_model=ExtractionStartResponse)
def start_forget_retain_extraction(
    full_dataset: UploadFile = File(...),
    poison_set: UploadFile = File(...),
    prompt_template: str = Form(...),
    experiment_name: str = Form("selection"),
    model_name: str = Form(...),
    max_length: int = Form(..., ge=1),
    adaptor_path: Optional[str] = Form(None),
    selection_method: DataSelectionMethod = Form(...),
    forget_size: int = Form(..., ge=1),
    retain_size: int = Form(..., ge=1),
    grace_top_n: Optional[int] = Form(None, ge=1),
    grace_num_clusters: Optional[int] = Form(None, ge=1),
    keep_gradients: bool = Form(False),
):
    if training_process_manager.is_running or evaluation_process_manager.is_running:
        raise HTTPException(
            status_code=409,
            detail="Dataset selection cannot start while training or evaluation is active.",
        )

    try:
        return gradient_cache_manager.start_extract(
            full_dataset=full_dataset,
            poison_set=poison_set,
            prompt_template=prompt_template,
            experiment_name=experiment_name,
            model_name=model_name,
            max_length=max_length,
            adaptor_path=adaptor_path,
            selection_method=selection_method.value,
            forget_size=forget_size,
            retain_size=retain_size,
            grace_top_n=grace_top_n,
            grace_num_clusters=grace_num_clusters,
            keep_gradients=keep_gradients,
        )
    except GradientCacheAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (GradientCacheError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/extract/status/{job_id}", response_model=ExtractionStatusResponse)
def extraction_status(job_id: str):
    try:
        return gradient_cache_manager.get_status(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Dataset extraction job not found: {job_id}"
        ) from exc


@router.post("/extract/cancel/{job_id}", response_model=JobCancelResponse)
def cancel_extraction(job_id: str):
    try:
        requested = gradient_cache_manager.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Dataset extraction job not found: {job_id}"
        ) from exc

    if not requested:
        return {"status": "idle", "message": "Dataset extraction is no longer running."}
    return {
        "status": "cancelling",
        "message": "Dataset extraction cancellation requested.",
    }
