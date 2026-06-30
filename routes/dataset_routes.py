from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from schemas import DatasetUploadResponse
from dataset.dataset_loader import (
    process_dataset,
    dataframe_preview,
)


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
