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
- Evaluation of a saved unlearned model on the forget and retain sets.

The next planned modules are:

- Training progress streaming from trainer callbacks back to FastAPI.

---

## 2. Current Project Tree

```text
├── config
│   ├── __init__.py
│   ├── api_config.py
│   └── training_config.py
├── dataset
│   ├── __init__.py
│   └── dataset_loader.py
├── eval
│   ├── __init__.py
│   └── eval_utils.py
├── eval_orchestrator.py
├── evaluation_process.py
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
│   ├── evaluation_routes.py
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
- Register route modules under the canonical `/api` prefix.
- Configure CORS from `CORS_ALLOWED_ORIGINS`.
- Provide `/api/health` and the backward-compatible `/` health endpoint.

The route modules retain their domain prefixes and are composed under one API router:

```python
api_router = APIRouter(prefix="/api")
api_router.include_router(model_router)
api_router.include_router(dataset_router)
api_router.include_router(config_router)
api_router.include_router(train_router)
app.include_router(api_router)
```

The original unprefixed routes are also registered as hidden compatibility aliases, and `/docs`, `/redoc`, and `/openapi.json` remain available through compatibility redirects or aliases. New clients should use `/api/*`; the compatibility paths can be removed in a future breaking release after external callers have migrated.

### API and CORS configuration

`config/api_config.py` reads `CORS_ALLOWED_ORIGINS` as a comma-separated list of explicit browser origins. If the variable is unset, local development allows:

```text
http://localhost:5173
http://127.0.0.1:5173
```

Wildcard origins are rejected. The current application does not use cookie or HTTP-auth credentials, so `allow_credentials` is `False`. Allowed methods and headers are limited to those used by the current JSON and multipart API calls.

Production deployments should set `CORS_ALLOWED_ORIGINS` to the deployed frontend origin or origins. An empty value disables cross-origin browser access, which is appropriate when a reverse proxy serves the frontend and `/api` from the same origin.

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

### 3.11 Evaluation pipeline

`POST /api/evaluation/start` accepts the `orchestrator_config` and
`training_result` returned by `/api/train/run`, plus a required user-selected
sentence-transformers repository name or local path. It returns a job id.
`GET /api/evaluation/status/{job_id}` returns the current phase, the full
progress history, and eventually the comparison result. The older blocking
`POST /api/evaluation/run` remains available for compatibility.

`eval_orchestrator.py`:

- Loads the original pre-unlearning model, calculates conditional probability,
  perplexity, and generations for both datasets, persists them, and removes the
  model.
- Loads the saved unlearnt full model or PEFT adapter, calculates and persists
  the same language-model outputs, and removes the model.
- Reloads the processed forget and retain datasets used for training.
- Derives bounded generation lengths when `num_tokens` is absent.
- Only after both language models have been removed, loads the requested
  sentence-transformers model and calculates retain-set cosine similarity.
- Calculates ROUGE-L for both datasets and aggregates pre/post forget quality
  and model utility from the persisted columns.
- Writes four row-level tables under `<model_output_dir>/evaluation/`.

`evaluation_process.py` owns one background evaluation child process and stores
timestamped progress events in memory for the status endpoint. Training and
evaluation are mutually exclusive to prevent concurrent GPU workloads.

A retain set is required because model utility cannot be computed without it.

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
POST /api/dataset/upload
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
POST /api/config/build
```

Purpose:

- Validate complete training configuration.
- Return final orchestrator JSON.
- Useful for frontend preview/debugging.

---

### Step 4: Run training

Endpoint:

```text
POST /api/train/run
```

Purpose:

- Build orchestrator config.
- Start a dedicated training child process through `training_process.py`.
- Call `run_orchestrator(config)` inside that child process.
- Return output directory and metrics after completion.

The HTTP request remains synchronous and waits until training finishes. Only one training process can be active at a time.

The active run can be stopped with:

```text
POST /api/train/stop
```

This terminates the training child process while leaving FastAPI running. The pending `/api/train/run` request then returns with `status = "stopped"`.

Future behavior should be asynchronous:

- Start job.
- Return `job_id`.
- Stream logs/progress via polling, Server-Sent Events, or WebSocket.

---

### Step 5: Evaluate the trained model

Endpoints:

```text
POST /api/evaluation/start
GET  /api/evaluation/status/{job_id}
```

Request:

```json
{
  "orchestrator_config": {},
  "training_result": {
    "output_dir": "outputs/run/forget_retain"
  },
  "embedding_model_name": "/models/my-sentence-transformer",
  "max_new_tokens": 256
}
```

The frontend supplies the complete training response objects. The embedding
model is required input; it may be a Hugging Face repository name or a local
path. The generation limit defaults to 256.

---

## 5. Current Backend Flow Diagram

```text
Frontend / Swagger / curl
        |
        v
FastAPI Routes
        |
        |-- /api/dataset/upload
        |       |
        |       v
        |   dataset_loader.py
        |       |
        |       v
        |   processed parquet paths
        |
        |-- /api/config/build
        |       |
        |       v
        |   training_config.py
        |       |
        |       v
        |   orchestrator JSON
        |
        |-- /api/train/run
                |
                v
        training_process.py
                |
                |-- spawn one training child process
                |-- terminate it on /api/train/stop
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
        |
        |-- /api/evaluation/start
                |
                v
        evaluation_process.py
                |
                v
        eval_orchestrator.py (child process)
                |
                |-- score + remove pre-unlearning model
                |-- score + remove unlearnt model
                |-- load embedding model and compute similarity
                |-- compute ROUGE-L, forget quality, model utility
                v
            progress status + comparison metrics + four parquet files
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

## 7. Frontend Integration

The implemented frontend is the Vite, React, and TypeScript application in `front end/`. Browser requests flow through `front end/src/apiConfig.ts` and `front end/src/api.ts`:

```text
React components
      |
      v
api.ts + apiConfig.ts
      |
      | HTTP/JSON to VITE_API_URL + /api
      v
FastAPI route modules
      |
      v
Existing config, dataset, model, and training services
```

`VITE_API_URL` is the backend origin, without the `/api` suffix. It defaults to `http://localhost:8000`; setting it to an empty string selects same-origin `/api` URLs for a reverse-proxy deployment. Vite substitutes this value when the development server starts or the production bundle is built.

The older `VITE_API_BASE_URL` setting remains a frontend compatibility fallback, but new configuration should use `VITE_API_URL`. The optional Vite `/api` proxy no longer strips the prefix because FastAPI now exposes `/api` routes directly.

---

## 8. Suggested Next Steps

### Deployment next step

Serve the built frontend and FastAPI behind a production reverse proxy, set `VITE_API_URL` for the chosen public API origin, and set `CORS_ALLOWED_ORIGINS` to the exact public frontend origin when the two origins differ.

### Backend next step

Add job management:

```text
POST /api/train/start
GET /api/train/status/{job_id}
GET /api/train/logs/{job_id}
```

Instead of blocking inside `/api/train/run`.

### Evaluation next step

Move the synchronous evaluation call to the same future job/status API used by
training and add an evaluation stop endpoint.

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
9. Frontend requests should go through `front end/src/api.ts` and `apiConfig.ts`; do not hardcode backend URLs in components.
