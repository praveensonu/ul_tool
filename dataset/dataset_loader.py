import os
import uuid
from pathlib import Path

import pandas as pd
from fastapi import UploadFile

UPLOAD_DIR = "uploaded_datasets"


def save_upload_file(file: UploadFile) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    ext = os.path.splitext(file.filename)[-1]
    raw_path = os.path.join(UPLOAD_DIR, f"raw_{uuid.uuid4()}{ext}")

    with open(raw_path, "wb") as f:
        f.write(file.file.read())

    return raw_path


def read_file(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError("Unsupported file type. Use csv, json, jsonl, or parquet.")


def validate_qa_columns(df: pd.DataFrame, dataset_name: str):
    required = {"question", "answer"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} missing required columns: {list(missing)}"
        )


def apply_prompt_template(df: pd.DataFrame, prompt_template: str) -> pd.DataFrame:
    """Substitute questions verbatim, keeping answers separate for continuation."""
    if "{question}" not in prompt_template:
        raise ValueError("prompt_template must contain {question}")

    df = df.copy()
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)

    df["question"] = df["question"].apply(
        lambda q: prompt_template.replace("{question}", q)
    )

    return df


def prepare_raslik_dataframe(
    df: pd.DataFrame,
    prompt_template: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Normalize an uploaded dataset to the JSONL schema consumed by RASLIK."""

    if df.empty:
        raise ValueError(f"{dataset_name} must contain at least one row.")

    prepared = df.copy()
    if "id" not in prepared.columns:
        prepared.insert(0, "id", range(len(prepared)))

    if prepared["id"].isna().any():
        raise ValueError(f"{dataset_name} contains an empty id.")

    prepared["id"] = prepared["id"].map(_stringify_raslik_id)
    if prepared["id"].duplicated().any():
        duplicates = prepared.loc[prepared["id"].duplicated(), "id"].tolist()
        raise ValueError(
            f"{dataset_name} contains duplicate ids: {duplicates[:5]}"
        )

    invalid_ids = prepared["id"].map(
        lambda value: value in {"", ".", ".."} or "/" in value or "\\" in value
    )
    if invalid_ids.any():
        raise ValueError(
            f"{dataset_name} ids must be non-empty file-safe values without slashes."
        )

    if {"question", "answer"}.issubset(prepared.columns):
        prepared = apply_prompt_template(prepared, prompt_template)
        prepared["prompt"] = prepared["question"]
        prepared["generation"] = prepared["answer"]
    elif {"prompt", "generation"}.issubset(prepared.columns):
        # Already-normalized JSONL exported by RASLIK can be cached again.
        prepared["prompt"] = prepared["prompt"].astype(str)
        prepared["generation"] = prepared["generation"].astype(str)
        if "question" not in prepared.columns:
            prepared["question"] = prepared["prompt"]
        if "answer" not in prepared.columns:
            prepared["answer"] = prepared["generation"]
    else:
        raise ValueError(
            f"{dataset_name} must contain question/answer or prompt/generation columns."
        )

    return prepared


def _stringify_raslik_id(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def save_jsonl_dataframe(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    return str(path)


def save_processed_dataframe(df: pd.DataFrame, dataset_name: str) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    path = os.path.join(
        UPLOAD_DIR,
        f"{dataset_name}_{uuid.uuid4()}.parquet",
    )

    df.to_parquet(path, index=False)
    return path


def process_dataset(
    file: UploadFile,
    prompt_template: str,
    dataset_name: str,
):
    raw_path = save_upload_file(file)
    df = read_file(raw_path)

    validate_qa_columns(df, dataset_name)

    df = apply_prompt_template(df, prompt_template)

    processed_path = save_processed_dataframe(df, dataset_name)

    return df, processed_path


def dataframe_preview(df: pd.DataFrame, n: int = 3):
    return df.head(n).to_dict(orient="records")
