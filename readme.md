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

## Run The Backend

Start the backend from the repository root:

```bash
.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend listens at:

```text
http://localhost:8000
```

Check that it is running:

```bash
curl -s http://127.0.0.1:8000/
```

Expected response:

```json
{"message":"LLM training API is running"}
```

## Run The Frontend

The frontend is intended to run through Docker, so host-level `npm` is not required.

Build the frontend image from the repository root:

```bash
docker build -t ascent-unlearning-frontend "front end"
```

Start the frontend container:

```bash
docker run -d \
  --name ascent-unlearning-frontend-dev \
  --add-host=host.docker.internal:host-gateway \
  -p 5173:5173 \
  -e VITE_API_BASE_URL=/api \
  -e API_PROXY_TARGET=http://host.docker.internal:8000 \
  ascent-unlearning-frontend
```

Open the app:

```text
http://localhost:5173
```

Start the backend before using the browser UI. The frontend proxies browser requests from `http://localhost:5173/api` to `http://host.docker.internal:8000`. If the backend is not running, actions such as dataset upload may fail with `502 Bad Gateway`.

## Docker Compose Option

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

## Stop Services

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

Check that the frontend proxy can reach the backend:

```bash
docker exec ascent-unlearning-frontend-dev wget -qO- http://127.0.0.1:5173/api/
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

- `GET /` - backend health check.
- `POST /dataset/upload` - upload a required forget set, optional retain set, and prompt template.
- `POST /config/build` - validate a training request and return the orchestrator config.
- `POST /train/run` - build the config and run training.

The frontend uses these endpoints to provide the main workflow: upload datasets, preview processed rows, configure model and training settings, inspect the generated orchestrator JSON, and launch training.

## Training Configuration Notes

- Supported methods are `full`, `lora`, and `adaptor`.
- Exactly one of `max_steps` or `epochs` must be set.
- `context_length` must be a power of two.
- `assistant_completions_only` is always `true`.
- `lora_settings` is required for `lora` and only allowed for `lora`.
- `adaptor_path` is required for `adaptor`.

More detail is available in:

- `docs/RUN_INSTRUCTIONS.md`
- `docs/ARCHITECTURE.md`
- `docs/FRONTEND.md`
