# Ascent Unlearning Tool

Ascent Unlearning Tool is a FastAPI backend with a Vite, React, and TypeScript frontend for configuring and launching LLM unlearning runs. The application supports dataset upload, prompt-template formatting, model and GPU settings, training configuration preview, and training launch through the backend API.

## Repository Layout

```text
.
├── main.py                  # FastAPI app entry point
├── routes/                  # API route modules
├── schemas.py               # Pydantic request and response schemas
├── config/                  # Orchestrator config builder
├── dataset/                 # Upload, validation, and preprocessing helpers
├── gpu/                     # GPU inspection utilities
├── model/                   # Model and adapter loading helpers
├── unlearning/              # Core unlearning/training code
├── front end/               # Vite React frontend
└── docs/                    # Architecture and run documentation
```

Generated files are intentionally ignored by Git, including `.venv/`, `outputs/`, `uploaded_datasets/`, Python bytecode, and frontend build output.

## Requirements

- Python environment with the backend dependencies installed.
- Docker for the frontend development server.
- A CUDA-capable environment if you want to run GPU training.

This repository currently includes a local `.venv/` workflow. The checked-in `requirements.txt` contains environment-specific pins, so use the existing environment when available or recreate dependencies carefully for your machine.

## Quick Start

The backend uses the repository's Python environment and the frontend runs in Docker. From the repository root, start both services with:

```bash
./start.sh
```

The command builds the frontend image, then starts both development servers with their logs in the current terminal:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`

Browser requests go directly to the backend's `/api` routes, with FastAPI CORS configured for the development frontend origin. Press `Ctrl+C` once to stop both services; the supervisor terminates the backend process group and stops the frontend container.

The defaults can be overridden when needed:

```bash
BACKEND_PORT=8080 FRONTEND_PORT=5174 ./start.sh
```

`PYTHON_BIN`, `BACKEND_HOST`, `BACKEND_PORT`, `FRONTEND_PORT`, `FRONTEND_IMAGE`, `FRONTEND_CONTAINER`, `VITE_API_URL`, `CORS_ALLOWED_ORIGINS`, and `API_PROXY_TARGET` are supported. When ports change, the default frontend API URL and CORS origins follow them automatically.

## Development: Run Services Individually

Running each service separately remains useful for focused development and debugging.

### Backend

Start the backend from the repository root:

```bash
.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Check it at `http://localhost:8000`:

```bash
curl -s http://127.0.0.1:8000/api/health
```

Expected response:

```json
{"message":"LLM training API is running"}
```

### Frontend

The frontend is intended to run through Docker, so host-level `npm` is not required.

Build the frontend image from the repository root:

```bash
docker build -t ascent-unlearning-frontend "front end"
```

Start the frontend container:

```bash
docker run -d \
  --name ascent-unlearning-frontend-dev \
  -p 5173:5173 \
  -e VITE_API_URL=http://localhost:8000 \
  ascent-unlearning-frontend
```

Open the app:

```text
http://localhost:5173
```

Start the backend before using the browser UI. The browser calls `http://localhost:8000/api` directly. If the backend is not running, actions such as dataset upload fail with a network error.

### Frontend Docker Compose Option

If Docker Compose is available:

```bash
cd "front end"
docker compose up --build
```

Stop it with:

```bash
cd "front end"
docker compose down
```

### Stop Individually Started Services

Stop the frontend container:

```bash
docker rm -f ascent-unlearning-frontend-dev
```

If the backend is running in the foreground, stop it with `Ctrl+C`.

If the backend was started in the background:

```bash
ps aux | grep "uvicorn main:app"
kill <PID>
```

## Frontend Health Checks

Check the frontend container:

```bash
docker ps --filter name=ascent-unlearning-frontend-dev
docker logs --tail 80 ascent-unlearning-frontend-dev
```

Check the backend's canonical health route:

```bash
curl -s http://127.0.0.1:8000/api/health
```

Expected response:

```json
{"message":"LLM training API is running"}
```

## Dataset Format

Uploaded datasets can be CSV, JSON, JSONL, or Parquet files. Each dataset must contain:

- `question`
- `answer`

The prompt template must contain `{question}`. During upload, the backend applies the template to each question and writes processed Parquet files under `uploaded_datasets/`.

## Main API Endpoints

- `GET /api/health` - backend health check.
- `POST /api/dataset/upload` - upload a required forget set, optional retain set, and prompt template.
- `POST /api/config/build` - validate a training request and return the orchestrator config.
- `POST /api/train/run` - build the config and run training in a dedicated child process.
- `POST /api/train/stop` - terminate the active training process without stopping the API.
- `POST /api/evaluation/start` - start pre/post evaluation as a background job.
- `GET /api/evaluation/status/{job_id}` - poll evaluation progress and results.
- `POST /api/evaluation/run` - blocking compatibility endpoint.

The original unprefixed endpoints remain available as compatibility aliases, but new integrations should use `/api/*`. API documentation is available at `http://localhost:8000/api/docs`.

Evaluation is a separate fifth frontend stage. It consumes the
`orchestrator_config` and `training_result` returned by training and requires a
sentence-transformers model name or local path. The original and unlearnt
language models are scored and removed one at a time before the embedding model
is loaded. The dashboard compares forget quality, model utility, perplexity,
conditional probability, ROUGE-L, and retain-set cosine similarity. Both forget
and retain datasets are required.

The frontend uses these endpoints to provide the main workflow: upload datasets, preview processed rows, configure model and training settings, inspect the generated orchestrator JSON, launch training, and stop an active run.

## Training Configuration Notes

- Supported model loading methods are `full`, `lora`, and `adaptor`.
- Supported unlearning methods are `grad_ascent`, `grad_diff`, `npo`, `dpo`, and `simnpo`,
  selected with `unlearning_method` (defaults to `simnpo`). 
- `retain_set_path` is required for `grad_diff`, `npo`, and `dpo`. `grad_ascent` has no
  retain term and ignores one.
- `dpo` needs a preferred answer per forget row: it uses an `alternate` column when the
  forget dataset has one, otherwise it samples from `unlearning/data_helpers/idk.jsonl`.
- Exactly one of `max_steps` or `epochs` must be set.
- `context_length` must be a power of two.
- `assistant_completions_only` is always `true`.
- `lora_settings` is required for `lora` and only allowed for `lora`.
- `adaptor_path` is required for `adaptor`.

More detail is available in:

- `docs/RUN_INSTRUCTIONS.md`
- `docs/ARCHITECTURE.md`
- `docs/FRONTEND.md`

## Resources

See `docs/resources.md` for product and UI references.

## Plan

See `docs/plan.md` for the project plan.
