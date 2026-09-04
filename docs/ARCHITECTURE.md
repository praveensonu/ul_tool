# LLM Unlearning FastAPI Backend Architecture

## 1. Purpose

This project is a FastAPI backend for launching LLM unlearning runs from structured API inputs.

The backend currently supports:

- GPU inspection and GPU selection.
- Hugging Face model loading.
- Optional PEFT/LoRA adapter loading.
- Dataset upload for forget and optional retain sets.
- Prompt-template formatting for question/answer datasets.
- RASLIK and GRACE forget/retain extraction from cached gradients.
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
├── data_selection
│   ├── __init__.py
│   ├── caching.py
│   ├── selection.py
│   ├── MP_main.py
│   └── RASLIK
│       ├── data_loader.py
│       ├── engine.py
│       └── ...
├── ex_configs
│   └── caching.json
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
│   ├── base.py
│   ├── data_helpers
│   │   ├── collators.py
│   │   ├── data_module.py
│   │   └── idk.jsonl
│   ├── dpo
│   │   ├── losses.py
│   │   └── trainer.py
│   ├── ga
│   │   ├── losses.py
│   │   └── trainer.py
│   ├── gd
│   │   ├── losses.py
│   │   └── trainer.py
│   ├── methods.py
│   ├── npo
│   │   ├── losses.py
│   │   └── trainer.py
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

The extraction workflow uses the same reader but calls
`prepare_raslik_dataframe()` instead. This accepts either `question`/`answer`
or already-normalized `prompt`/`generation` data. It creates an `id` column
when absent, rejects empty, duplicate, or path-unsafe ids, and writes
normalized JSONL containing `id`, `prompt`, and `generation`. For ordinary
question/answer input, `prompt` is the template-applied question and
`generation` is the answer. The original `question` and `answer` columns are
retained so selected rows can be consumed directly by the unlearning code.

---

### 3.5.1 RASLIK and GRACE dataset selection

Selecting **Extract** in the frontend starts a background job with:

```text
POST /api/dataset/extract/start
GET  /api/dataset/extract/status/{job_id}
POST /api/dataset/extract/cancel/{job_id}
```

The older blocking `POST /api/dataset/extract` endpoint remains available for
compatibility.

The multipart form contract is:

| Field | Required | Meaning |
|---|---:|---|
| `full_dataset` | yes | Full training pool (`csv`, `json`, `jsonl`, or `parquet`) |
| `poison_set` | yes | Small poison dataset in a supported format |
| `prompt_template` | yes | Template containing `{question}` |
| `experiment_name` | no | Human-readable prefix; defaults to `selection` |
| `model_name` | yes | Hugging Face model name or local path |
| `max_length` | yes | Positive sequence length used while caching |
| `adaptor_path` | no | PEFT/LoRA adapter path |
| `selection_method` | yes | `raslik` or `grace` |
| `forget_size` | yes | Positive forget-set size |
| `retain_size` | yes | Positive retain-set size |
| `grace_top_n` | GRACE | Size of the forget candidate pool |
| `grace_num_clusters` | GRACE | Number of retain-pool clusters |
| `keep_gradients` | no | Keep cached tensors after selection; defaults to `false` |

Both uploads are normalized to JSONL as described above before any GPU work
begins. A sanitized experiment prefix plus a random suffix isolates each run.

`data_selection/caching.py` generates one caching config for each normalized
dataset and runs `data_selection/MP_main.py --config_path ...` sequentially.
The generated configs preserve the checked-in `ex_configs/caching.json`
defaults while overriding the normalized data path, gradient path, model,
optional LoRA path, and maximum length. Compressed gradients are stored at:

```text
outputs/gradients/<experiment_name>/training
outputs/gradients/<experiment_name>/poison
```

After caching, `data_selection/selection.py` applies one of two selectors:

- **RASLIK** computes each training sample's inner product with the average
  poison gradient. This is equivalent to averaging that sample's inner
  products with every poison gradient. The highest-scoring ids form the
  forget set and the lowest-scoring non-forget ids form the retain set.
- **GRACE** ranks a configurable top-n candidate pool, applies non-negative
  OMP (NNOMP) against the average poison gradient for forget selection, and
  excludes the entire top-n pool from retain selection. It projects the
  remaining gradients away from the average poison direction, clusters the
  projected gradients with K-means, and applies OMP between each cluster
  centroid and its samples. Retain quotas are distributed as evenly as
  cluster capacity allows, with deterministic fallback filling if OMP stops
  before the requested count.

Parameter validation runs before gradient caching. RASLIK requires
`forget_size + retain_size <= full dataset size`. GRACE requires
`grace_top_n >= forget_size`, `grace_top_n + retain_size <= full dataset
size`, and a cluster count no greater than either `retain_size` or the pool
remaining after top-n exclusion. Training-gradient filenames must also match
the normalized full-dataset ids exactly.

The selected rows are read from the normalized full dataset by `id` and
written directly as:

```text
outputs/gradients/<experiment_name>/selected/<method>/forget.parquet
outputs/gradients/<experiment_name>/selected/<method>/retain.parquet
outputs/gradients/<experiment_name>/selected/<method>/selection.json
outputs/gradients/<experiment_name>/selected/<method>/average_poison_gradient.pt
```

The API returns these paths in the same fields used by direct dataset upload,
along with previews, row counts, cache paths, config paths, and the resolved
experiment name. The frontend stores the extraction response as its dataset
response, so no second upload is required before unlearning.

By default, the training and poison gradient directories and the saved average
poison gradient are removed immediately after successful selection. The
selected parquet files, selection metadata, configs, and logs remain. When
`keep_gradients=true`, all gradient tensors remain and the result reports
`gradients_retained=true`.

`POST /api/dataset/cache-gradients` remains available for cache-only use. It
accepts the common dataset/model fields but does not run selection.

Gradient caching and selection are GPU-intensive. The background job records
progress for dataset preparation, full-dataset caching, poison-set caching,
selection, gradient retention/removal, and completion. A process lock permits
only one cache/selection request at a time, and training and evaluation reject
startup while it is active. Conversely, cache/selection requests are rejected
while training or evaluation is active.

Cancellation sets a cooperative cancellation flag. The active `MP_main.py`
process group receives `SIGTERM` and is given time to exit before a `SIGKILL`
fallback. Partial cached gradients and partial selected outputs are removed,
and the job finishes with `cancelled` rather than `failed`.

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

Unlearning method selection:

`orchestrator_config["unlearning"]["method"]` picks the method, and
`build_unlearning_run()` resolves it to a dataset, collator, trainer class and the
trainer's hardcoded arguments:

| Method | Dataset | Collator | Trainer |
|---|---|---|---|
| `grad_ascent` | `ForgetOnlyDataset` | `ForgetCollator` | `GradAscentTrainer` |
| `grad_diff` | `ForgetRetainDataset` | `RetainCollator` | `GradDiffTrainer` |
| `npo` | `ForgetRetainDataset` | `RetainCollator` | `NPOTrainer` |
| `dpo` | `IdkForgetRetainDataset` | `DpoRetainCollator` | `DPOTrainer` |
| `simnpo` | `ForgetRetainDataset` | `RetainCollator` | `SimNPOForgetRetainTrainer` |

`grad_diff`, `npo` and `dpo` require a retain set; the schema rejects a request without
one. `grad_ascent` has no retain term and ignores a retain set if one was uploaded.
`simnpo` falls back to `ForgetOnlyDataset` + `SimNPOForgetOnlyTrainer` when no retain set
is given.


`run_type` — and therefore the output subdirectory — is
`{method}_{forget_only|forget_retain}`.

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
IdkForgetRetainDataset
```

`ForgetRetainDataset` samples forget examples sequentially and retain examples randomly.

`IdkForgetRetainDataset` (used by DPO) adds a third, preferred answer per forget example.
It reads that answer from an `alternate` column when the forget data has one, otherwise it
samples a random line from `unlearning/data_helpers/idk.jsonl` — a verbatim copy of the
100 "I don't know" responses open-unlearning uses for its own DPO runs.

This module should remain unchanged unless the data format or loss-masking logic changes.

---

### 3.9 `unlearning/data_helpers/collators.py`

Batch collation logic.

Available collators:

```text
ForgetCollator
RetainCollator
DpoRetainCollator
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

`DpoRetainCollator` returns forget/alternate/retain triples in that order.

This module should remain unchanged unless the trainer input format changes.

---

### 3.10 `unlearning/<method>/trainer.py`

One folder per unlearning method, each with `losses.py` (the math) and `trainer.py` (a
Hugging Face `Trainer` subclass overriding `compute_loss`). Algorithms and defaults follow
[open-unlearning](https://github.com/locuslab/open-unlearning/tree/main/src/trainer/unlearn).

```text
unlearning/ga/trainer.py     GradAscentTrainer(Trainer)
unlearning/gd/trainer.py     GradDiffTrainer(Trainer)
unlearning/npo/trainer.py    NPOTrainer(GradDiffTrainer)
unlearning/dpo/trainer.py    DPOTrainer(GradDiffTrainer)
unlearning/snpo/trainer.py   SimNPOForgetOnlyTrainer(Trainer)
                             SimNPOForgetRetainTrainer(GradDiffTrainer)
```

`GradDiffTrainer` is the base for every retain-aware method: it owns `compute_retain_loss`
(`NLL` or `KL`) and `_prepare_ref_model`, a frozen `deepcopy` of the model. NPO and DPO
always build that reference model, so they hold **two** copies of the model in memory.

`unlearning/gd/losses.py` also holds the primitives shared across methods —
`compute_batch_nll`, `compute_dpo_loss`, `compute_kl_divergence` and `to_model_inputs`.

Every trainer descends from `UnlearnTrainer` (`unlearning/base.py`), which mirrors
open-unlearning's base class: it routes `prediction_step` back to the stock
`Trainer.compute_loss` (an eval batch is a plain LM batch, not a forget/retain
structure) and clears `model_accepts_loss_kwargs` so that the forget-only and
forget/retain batch formats normalise by `gradient_accumulation_steps` identically.

---

### 3.11 Evaluation pipeline

`POST /api/evaluation/start` accepts the `orchestrator_config` and
`training_result` returned by `/api/train/run`, plus a required user-selected
sentence-transformers repository name or local path. It returns a job id.
`GET /api/evaluation/status/{job_id}` returns the current phase, the full
progress history, and eventually the comparison result. The older blocking
`POST /api/evaluation/run` remains available for compatibility.
`POST /api/evaluation/cancel/{job_id}` requests cooperative cancellation. The
evaluation stops at its next progress boundary, allowing active model-loading
phases to run their `finally` cleanup and release GPU memory.

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
timestamped progress events in memory for the status endpoint. Training,
evaluation, and gradient caching/selection are mutually exclusive to prevent
concurrent GPU workloads.

A retain set is required because model utility cannot be computed without it.

---

## 4. API Flow

### Step 1: Start backend

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

### Step 2: Prepare datasets

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

For the alternative **Extract** workflow, use:

```text
POST /api/dataset/extract/start
GET  /api/dataset/extract/status/{job_id}
POST /api/dataset/extract/cancel/{job_id}
```

Inputs are the full dataset, poison set, prompt template, model name/path,
maximum length, optional adapter path, selection method, forget size, retain
size, and the GRACE-only top-n and cluster fields. The request normalizes both
uploads, caches gradients once for the full training data and once for the
poison data, applies RASLIK or GRACE selection, and returns the generated
forget/retain parquet paths in the same fields as direct upload. The optional
`keep_gradients` form field controls whether intermediate gradient tensors are
retained after a successful extraction.

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
        |-- /api/dataset/extract/start + status + cancel
        |       |
        |       v
        |   normalize full + poison data to JSONL
        |       |
        |       v
        |   MP_main.py twice (training, poison)
        |       |
        |       v
        |   RASLIK or GRACE selection
        |       |
        |       v
        |   forget.parquet + retain.parquet
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

Add finer-grained row-level progress and cooperative cancellation checks to
long model-scoring batches.

---

## 9. Notes for Codex

When editing this project:

1. Do not modify `unlearning/` unless explicitly asked.
2. Keep API schemas in `schemas.py`.
3. Keep route handlers thin.
4. Put business logic in modules:
   - `dataset/`
   - `data_selection/`
   - `model/`
   - `config/`
   - `evaluation/`
   - `orchestrator.py`
5. The final object passed to training should always be a single JSON-like dictionary.
6. Processed or extracted parquet paths, not raw uploaded paths, should be used for training.
7. For training adapters, keep PEFT adapter attached and trainable.
8. For selected GPU, set `CUDA_VISIBLE_DEVICES` before loading model.
9. Frontend requests should go through `front end/src/api.ts` and `apiConfig.ts`; do not hardcode backend URLs in components.
10. Keep RASLIK/GRACE orchestration in `data_selection/caching.py` and selection math in `data_selection/selection.py`.
