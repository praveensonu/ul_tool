# LLM Unlearning FastAPI Backend Architecture

## 1. Purpose

This project is a FastAPI backend for launching LLM unlearning runs from structured API inputs.

The backend currently supports:

- GPU inspection and GPU selection.
- Hugging Face model loading.
- Optional PEFT/LoRA adapter loading.
- Dataset upload for forget and optional retain sets.
- Prompt-template formatting for question/answer datasets.
- Hyperparameter validation.
- Construction of a single orchestrator JSON.
- Running the orchestrator from a FastAPI endpoint.

The next planned modules are:

- Evaluation pipeline.
- Frontend UI.
- Training progress streaming from trainer callbacks back to FastAPI.

---

## 2. Current Project Tree

```text
├── config
│   ├── __init__.py
│   └── training_config.py
├── dataset
│   ├── __init__.py
│   └── dataset_loader.py
├── gpu
│   ├── __init__.py
│   └── gpu_utils.py
├── main.py
├── model
│   ├── __init__.py
│   └── model_loader.py
├── orchestrator.py
├── training_process.py
├── routes
│   ├── __init__.py
│   ├── config_routes.py
│   ├── dataset_routes.py
│   ├── model_routes.py
│   └── train_routes.py
├── schemas.py
├── unlearning
│   ├── data_helpers
│   │   ├── collators.py
│   │   └── data_module.py
│   └── snpo
│       ├── losses.py
│       └── trainer.py
└── uploaded_datasets
```

`unlearning/` is treated as core research code and should not be modified unless the training method itself changes.

---

## 3. Main Components

### 3.1 `main.py`

Entry point for the FastAPI app.

Responsibilities:

- Create the FastAPI app.
- Register route modules.
- Provide a health/home endpoint.

Expected routers:

```python
app.include_router(model_router)
app.include_router(dataset_router)
app.include_router(config_router)
app.include_router(train_router)
```

---

### 3.2 `schemas.py`

Central Pydantic schema file.

Responsibilities:

- Validate model-loading requests.
- Validate dataset/config/training requests.
- Validate hyperparameter constraints.
- Enforce method-specific rules.

Important schema groups:

#### Model method

```text
full
lora
adaptor
```

#### General hyperparameters

Rules:

- Exactly one of `max_steps` or `epochs` must be provided.
- Both are integers with minimum value `1`.
- `learning_rate` can be a float or float-like string such as `"1e-4"`.
- `context_length` must be a power of 2.

#### LoRA settings

Allowed target modules:

```text
q_proj
v_proj
k_proj
o_proj
```

Rules:

- `lora_settings` is required only when `method == "lora"`.
- `lora_settings` is forbidden for `full` and `adaptor`.

#### Dataset rules

- `forget_set_path` is required.
- `retain_set_path` is optional.
- Retain-only training is not allowed.

#### Memory rule

```python
assistant_completions_only = True
```

This is always true.

---

### 3.3 `gpu/gpu_utils.py`

GPU utility module.

Responsibilities:

- Query available GPUs.
- Validate selected GPU id.
- Support future GPU information endpoints.

Expected behavior:

- The API receives a `gpu_id`.
- The orchestrator sets:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
```

- Inside the training process, the selected physical GPU becomes visible as local `cuda:0`.

---

### 3.4 `model/model_loader.py`

Model-loading module.

Responsibilities:

- Load tokenizer.
- Load base model from Hugging Face or local path.
- Load optional PEFT adapter.
- Handle different loading modes.

Current behavior:

#### `full`

- Loads the base model.
- If `adaptor_path` is supplied, loads adapter and merges it.
- Returns model, tokenizer, and merged flag.

#### `lora`

- Loads the base model.
- If `adaptor_path` is supplied, loads adapter and merges it.
- Returns model, tokenizer, and merged flag.

#### `adaptor`

- Requires `adaptor_path`.
- Loads base model.
- Loads PEFT adapter with `is_trainable=True`.
- Keeps adapter attached for training.
- Does not merge adapter before training.

Important design point:

For training an adapter, do not call `merge_and_unload()` before training, because that can remove trainable PEFT adapter parameters.

---

### 3.5 `dataset/dataset_loader.py`

Dataset ingestion module.

Responsibilities:

- Save uploaded files.
- Read supported dataset formats.
- Validate required columns.
- Apply prompt template.
- Save processed dataset to parquet.
- Return preview rows to frontend/API.

Supported formats:

```text
.csv
.json
.jsonl
.parquet
```

Required columns:

```text
question
answer
```

Prompt-template behavior:

The template is applied to the `question` column. The `answer` column remains separate because the unlearning data module later concatenates:

```python
question + answer + tokenizer.eos_token
```

Example template:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 July 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

The template must include:

```text
{question}
```

Dataset output:

- Uploaded raw file is saved.
- Processed/template-applied parquet is saved.
- Processed parquet path should be used in training config.

---

### 3.6 `config/training_config.py`

Builds the final orchestrator JSON from validated FastAPI input.

Responsibilities:

- Convert frontend/API fields into one training config.
- Keep model, dataset, GPU, and hyperparameters grouped.
- Provide exactly the structure expected by `orchestrator.py`.

Final JSON shape:

```json
{
  "model": {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "adaptor_path": "/path/to/adapter",
    "method": "adaptor",
    "hf_key": "optional_hf_token"
  },
  "dataset": {
    "forget_set_path": "uploaded_datasets/forget_x.parquet",
    "retain_set_path": "uploaded_datasets/retain_y.parquet"
  },
  "gpu": {
    "gpu_id": 0
  },
  "hyperparams": {
    "general": {
      "max_steps": 100,
      "epochs": null,
      "learning_rate": 0.0001,
      "context_length": 2048
    },
    "optimization": {
      "batch_size": 1,
      "grad_accum": 8,
      "weight_decay": 0.01
    },
    "schedule": {
      "save_steps": 10
    },
    "memory": {
      "assistant_completions_only": true
    },
    "lora_settings": {
      "target_modules": ["q_proj", "v_proj"]
    }
  }
}
```

---

### 3.7 `orchestrator.py`

Runtime training launcher.

Responsibilities:

- Receive final orchestrator JSON.
- Set `CUDA_VISIBLE_DEVICES`.
- Load model and tokenizer.
- Load forget and optional retain datasets.
- Build Hugging Face `TrainingArguments`.
- Select forget-only or forget-retain trainer.
- Start training.
- Save model/tokenizer output.
- Clear GPU memory after training.

Training modes:

#### Forget-only mode

Triggered when:

```python
retain_set_path is None
```

Uses:

```text
ForgetOnlyDataset
ForgetCollator
SimNPOForgetOnlyTrainer
```

#### Forget-retain mode

Triggered when:

```python
retain_set_path is not None
```

Uses:

```text
ForgetRetainDataset
RetainCollator
SimNPOForgetRetainTrainer
```

Important integration detail:

`unlearning/snpo/trainer.py` imports `losses` as a local module. Therefore, `orchestrator.py` adds:

```python
unlearning/snpo
```

to `sys.path` before importing the trainer.

---

### 3.8 `unlearning/data_helpers/data_module.py`

Core dataset logic.

This file converts dataframe rows into tokenized model inputs.

Important behavior:

```python
full_text = question + answer + tokenizer.eos_token
```

Labels for the question part are masked with `-100`, so only assistant answer tokens contribute to the loss.

Dataset classes:

```text
ForgetOnlyDataset
ForgetRetainDataset
```

`ForgetRetainDataset` samples forget examples sequentially and retain examples randomly.

This module should remain unchanged unless the data format or loss-masking logic changes.

---

### 3.9 `unlearning/data_helpers/collators.py`

Batch collation logic.

Available collators:

```text
ForgetCollator
RetainCollator
```

`ForgetCollator` returns a dictionary:

```python
{
  "input_ids": ...,
  "labels": ...,
  "attention_mask": ...
}
```

`RetainCollator` returns paired forget/retain batches.

This module should remain unchanged unless the trainer input format changes.

---

### 3.10 `unlearning/snpo/trainer.py`

Core SimNPO training logic.

Current trainers:

```text
SimNPOForgetOnlyTrainer
SimNPOForgetRetainTrainer
```

Depending on the current version, trainer implementation may be either:

1. Direct subclasses of Hugging Face `Trainer`, or
2. Wrapper classes that internally build a Hugging Face `Trainer`.

The orchestrator should be kept compatible with the active local trainer version.

---

## 4. API Flow

### Step 1: Start backend

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

### Step 2: Upload dataset

Endpoint:

```text
POST /dataset/upload
```

Inputs:

- `forget_set`: required file
- `retain_set`: optional file
- `prompt_template`: required text

Output:

- Forget dataset processed parquet path.
- Optional retain dataset processed parquet path.
- Preview rows.
- Row counts.

The frontend should store the returned processed paths and use them in the training request.

---

### Step 3: Build config without running training

Endpoint:

```text
POST /config/build
```

Purpose:

- Validate complete training configuration.
- Return final orchestrator JSON.
- Useful for frontend preview/debugging.

---

### Step 4: Run training

Endpoint:

```text
POST /train/run
```

Purpose:

- Build orchestrator config.
- Start a dedicated training child process through `training_process.py`.
- Call `run_orchestrator(config)` inside that child process.
- Return output directory and metrics after completion.

The HTTP request remains synchronous and waits until training finishes. Only one training process can be active at a time.

The active run can be stopped with:

```text
POST /train/stop
```

This terminates the training child process while leaving FastAPI running. The pending `/train/run` request then returns with `status = "stopped"`.

Future behavior should be asynchronous:

- Start job.
- Return `job_id`.
- Stream logs/progress via polling, Server-Sent Events, or WebSocket.

---

## 5. Current Backend Flow Diagram

```text
Frontend / Swagger / curl
        |
        v
FastAPI Routes
        |
        |-- /dataset/upload
        |       |
        |       v
        |   dataset_loader.py
        |       |
        |       v
        |   processed parquet paths
        |
        |-- /config/build
        |       |
        |       v
        |   training_config.py
        |       |
        |       v
        |   orchestrator JSON
        |
        |-- /train/run
                |
                v
        training_process.py
                |
                |-- spawn one training child process
                |-- terminate it on /train/stop
                v
            orchestrator.py (child process)
                |
                |-- set CUDA_VISIBLE_DEVICES
                |-- load model/tokenizer
                |-- read processed datasets
                |-- build Dataset object
                |-- build Trainer
                |-- train
                |-- save model/tokenizer
                v
            output directory + metrics
```

---

## 6. Example Training Request

```json
{
  "model_name": "meta-llama/Llama-2-7b-hf",
  "adaptor_path": "/path/to/adapter",
  "hf_key": null,
  "method": "adaptor",
  "gpu_id": 0,
  "forget_set_path": "uploaded_datasets/forget_abc.parquet",
  "retain_set_path": "uploaded_datasets/retain_xyz.parquet",
  "hyperparams": {
    "general": {
      "max_steps": 100,
      "learning_rate": "1e-4",
      "context_length": 2048
    },
    "optimization": {
      "batch_size": 1,
      "grad_accum": 8,
      "weight_decay": 0.01
    },
    "schedule": {
      "save_steps": 10
    },
    "memory": {
      "assistant_completions_only": true
    }
  }
}
```

For `method = "lora"`:

```json
"lora_settings": {
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
}
```

---

## 7. Frontend Recommendation

Because this project is running on a university server without sudo access, the first frontend should avoid a heavy TypeScript/Node setup.

Recommended order:

### Option A: Streamlit first

Best for fast internal UI.

Advantages:

- Python-only.
- Can be installed inside the existing `uv` virtual environment.
- Easy file upload widgets.
- Easy forms for hyperparameters.
- Easy API calls to FastAPI with `requests`.

Install:

```bash
uv pip install streamlit requests
```

Run:

```bash
streamlit run frontend_app.py --server.address 0.0.0.0 --server.port 8501
```

Suggested use:

- Build first prototype UI.
- Upload forget/retain datasets.
- Enter model config.
- Enter hyperparameters.
- Submit to FastAPI.

### Option B: Gradio

Good for ML demos and simple model-control interfaces.

Install:

```bash
uv pip install gradio requests
```

Run a Python UI that calls FastAPI endpoints.

### Option C: React + TypeScript later

Use this only after the backend stabilizes.

Advantages:

- Better long-term frontend.
- Better state management.
- Better user experience.

Disadvantages for current setup:

- Requires Node/npm tooling.
- More learning overhead.
- More files and deployment complexity.

If Node is already available on the server, Vite can scaffold a React/TypeScript frontend without sudo:

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

If Node is not available, avoid TypeScript for now and start with Streamlit.

---

## 8. Suggested Next Steps

### Immediate next step

Build a minimal Streamlit frontend:

1. Dataset upload page.
2. Model config form.
3. Hyperparameter form.
4. Config preview page.
5. Run training button.

### Backend next step

Add job management:

```text
POST /train/start
GET /train/status/{job_id}
GET /train/logs/{job_id}
```

Instead of blocking inside `/train/run`.

### Evaluation next step

Add an `evaluation/` module:

```text
evaluation
├── __init__.py
├── evaluator.py
└── metrics.py
```

Possible endpoints:

```text
POST /evaluate/run
GET /evaluate/status/{job_id}
```

Evaluation config should include:

- Model output path.
- Evaluation dataset path.
- Metrics.
- Batch size.
- Max generation tokens.
- Device/GPU id.

---

## 9. Notes for Codex

When editing this project:

1. Do not modify `unlearning/` unless explicitly asked.
2. Keep API schemas in `schemas.py`.
3. Keep route handlers thin.
4. Put business logic in modules:
   - `dataset/`
   - `model/`
   - `config/`
   - `evaluation/`
   - `orchestrator.py`
5. The final object passed to training should always be a single JSON-like dictionary.
6. Processed dataset paths, not raw uploaded paths, should be used for training.
7. For training adapters, keep PEFT adapter attached and trainable.
8. For selected GPU, set `CUDA_VISIBLE_DEVICES` before loading model.
9. Future frontend should first use Streamlit unless a full React/TypeScript app is required.
