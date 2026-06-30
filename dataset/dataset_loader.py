import os
import uuid
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
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".json"):
        return pd.read_json(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith(".parquet"):
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
    if "{question}" not in prompt_template:
        raise ValueError("prompt_template must contain {question}")

    df = df.copy()
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)

    df["question"] = df["question"].apply(
        lambda q: prompt_template.format(question=q)
    )

    return df


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