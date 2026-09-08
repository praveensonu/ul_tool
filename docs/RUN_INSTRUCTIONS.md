# Run Instructions

This project can run without host-level `npm`. The frontend is intended to run through Docker.

## Start the Entire Application

From the repository root, run:

```bash
./start.sh
```

This builds the existing frontend Docker image and starts the existing backend and frontend development commands concurrently. Both logs remain attached to the terminal. Open:

```text
http://localhost:5173
```

The browser calls `http://localhost:8000/api` directly, and FastAPI allows the development frontend origin `http://localhost:5173`. Press `Ctrl+C` to terminate the backend process group and stop the frontend container.

Ports and environment settings can be overridden if necessary:

```bash
BACKEND_PORT=8081 FRONTEND_PORT=5174 ./start.sh
```

`FRONTEND_PORT` is passed to Vite and used for both sides of the Docker port
mapping. In this example, Vite listens on port `5174` inside the container and
Docker publishes `5174:5174` on the host. The frontend API URL and backend CORS
origins also follow the overridden ports unless explicitly configured.

See the root `README.md` for all supported overrides.

## Run Services Individually

Use the following commands when developing or debugging one service at a time. Start the backend before the frontend. If the backend is not running, browser actions such as dataset upload can fail with a network error such as:

```text
Failed to fetch
```

This normally means the configured `VITE_API_URL` is unreachable. A browser CORS error instead means the frontend origin is missing from `CORS_ALLOWED_ORIGINS`.

### Start Backend

From the repository root:

```bash
.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend listens on:

```text
http://localhost:8000
```

Check the backend:

```bash
curl -s http://127.0.0.1:8000/api/health
```

Expected response:

```json
{"message":"LLM training API is running"}
```

After a successful training response, the frontend unlocks its Evaluation
stage. Enter a sentence-transformers repository name or local path, start the
job, and keep the page open while it polls `/api/evaluation/status/{job_id}`.
Evaluation requires the training run to include a retain set.

### Stop Backend

If the backend is running in the foreground, stop it with:

```bash
Ctrl+C
```

If it was started in the background, find and stop the process:

```bash
ps aux | grep "uvicorn main:app"
kill <PID>
```

### Build Frontend Image

You only need to rebuild after changing frontend files or dependencies.

From the repository root:

```bash
docker build -t ascent-unlearning-frontend "front end"
```

### Start Frontend Container

From the repository root:

```bash
docker run -d \
  --name ascent-unlearning-frontend-dev \
  -p 5173:5173 \
  -e FRONTEND_PORT=5173 \
  -e VITE_API_URL=http://localhost:8000 \
  ascent-unlearning-frontend
```

Open:

```text
http://localhost:5173
```

The frontend expects the backend to be running on port `8000`. `localhost` is correct here because `VITE_API_URL` is used by the browser, not by the frontend container.

To run the frontend on another port, use the same value for the host mapping,
container mapping, and `FRONTEND_PORT`:

```bash
docker run -d \
  --name ascent-unlearning-frontend-dev \
  -p 5174:5174 \
  -e FRONTEND_PORT=5174 \
  -e VITE_API_URL=http://localhost:8081 \
  ascent-unlearning-frontend
```

### Stop Frontend Container

```bash
docker rm -f ascent-unlearning-frontend-dev
```

### Restart Frontend Container

```bash
docker rm -f ascent-unlearning-frontend-dev
docker run -d \
  --name ascent-unlearning-frontend-dev \
  -p 5173:5173 \
  -e FRONTEND_PORT=5173 \
  -e VITE_API_URL=http://localhost:8000 \
  ascent-unlearning-frontend
```

### Check Frontend Container

```bash
docker ps --filter name=ascent-unlearning-frontend-dev
docker logs --tail 80 ascent-unlearning-frontend-dev
```

Check the canonical API health endpoint from the host:

```bash
curl -s http://127.0.0.1:8000/api/health
```

Expected response:

```json
{"message":"LLM training API is running"}
```

If this fails, start the backend with:

```bash
.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Use A Different Backend URL

Change `VITE_API_URL` when starting the container. This value must be a URL the user's browser can reach:

```bash
docker run -d \
  --name ascent-unlearning-frontend-dev \
  -p 5173:5173 \
  -e FRONTEND_PORT=5173 \
  -e VITE_API_URL=http://192.168.1.20:8000 \
  ascent-unlearning-frontend
```

Also include `http://localhost:5173` (or the actual frontend origin) in the backend's comma-separated `CORS_ALLOWED_ORIGINS` value.

## Environment Configuration

The checked-in examples are `.env.example` for FastAPI and `front end/.env.example` for Vite.

- `CORS_ALLOWED_ORIGINS`: comma-separated exact browser origins allowed by FastAPI. It defaults to `http://localhost:5173,http://127.0.0.1:5173`. Set it in the backend process environment; Uvicorn does not read the example file automatically.
- `FRONTEND_PORT`: Vite's listening port and both sides of the Docker port mapping when using `start.sh` or Docker Compose. It defaults to `5173`.
- `VITE_API_URL`: backend origin visible to the browser, without `/api`. It defaults to `http://localhost:8000` in the frontend client.
- `API_PROXY_TARGET`: optional Vite proxy target. It is only needed when `VITE_API_URL` is empty and the browser uses same-origin `/api` URLs.

For a same-origin production deployment, build the frontend with an empty `VITE_API_URL`, route `/api` to FastAPI in the reverse proxy, and set `CORS_ALLOWED_ORIGINS` to an empty value. For different public origins, set both variables to the exact public URLs; do not use `*`.

### Docker Compose Optional

If Docker Compose is available:

```bash
cd "front end"
docker compose up --build
```

To use another frontend port with Compose, pass `FRONTEND_PORT` to the command:

```bash
cd "front end"
FRONTEND_PORT=5174 docker compose up --build
```

Stop it with:

```bash
cd "front end"
docker compose down
```

This server currently has the base `docker` CLI available, but may not have the `docker compose` plugin installed.

### Shut Individually Started Services Down

Stop the frontend:

```bash
docker rm -f ascent-unlearning-frontend-dev
```

Stop the backend if it is running in the foreground:

```bash
Ctrl+C
```

If the backend was started in the background:

```bash
ps aux | grep "uvicorn main:app"
kill <PID>
```

### Batching and evaluation result files

The Data step exposes **Gradient batch size per GPU** (default `2`). This is sent
as the `gradient_batch_size` multipart field to caching/extraction endpoints and
stored in the generated RASLIK configuration under `influence.gradient_batch_size`.
The cache batches gradient computation using vectorized per-example derivatives,
then applies RapidGrad compression to the batch. Every sample still produces its
own gradient file; gradients are not averaged across examples. Existing standalone
RASLIK configs default to `1`. Batched mode supports caching-only runs with one
RapidGrad projection size and without DeepSpeed. Set the value to `1` for other
RASLIK modes, limited GPU memory, or models whose backward operations do not support
vectorized gradients. Worker failures now fail the job rather than retry forever.

The Evaluation step exposes **Evaluation batch size** (default `4`), sent as JSON
`batch_size` to `/api/evaluation/start` or `/api/evaluation/run`. It controls
language-model generation, conditional probability, and perplexity. Padding and
prompt tokens are excluded from each sample's scoring loss; the metric definition
is unchanged. Generation uses left padding and keeps each row's token limit.
Embedding similarity retains its separate `embedding_batch_size` setting (default
`32`). Batch size is independent of the training batch size, and larger batches
require more GPU memory. Evaluation reports progress and checks cancellation
between batches. The previously selected GPU allocation still applies.

The frontend sends the project name as `experiment_name`. After each successful
evaluation, the full JSON result is appended as one line to
`outputs/results/<experiment-name>.jsonl`; spaces and unsafe filename characters
are replaced with hyphens. Repeated evaluations append additional records with
`completed_at` timestamps. The response includes the location in
`output_files.results_jsonl_path`, and the UI displays it. Existing per-sample
Parquet output files remain available. API clients can set `experiment_name`
explicitly; if omitted, the orchestrator uses the configuration's experiment name
or the parent directory name of the trained model.

### MMLU and GPQA benchmarks

Check **Include benchmark evaluation (MMLU and GPQA)** to run EleutherAI
`lm_eval` against each already-loaded model, after
its forget/retain scoring and before releasing it. The checkbox defaults to off
and is disabled while evaluation is running. API clients can opt in with
`include_benchmarks: true`; omitted or false skips both benchmarks and returns
`benchmarks: null` for each model. The selection is saved with the project.
Benchmark evaluation uses the first selected GPU
and the evaluation batch size. Benchmarks run once per model, including when a
separate multi-GPU generation phase is needed; no extra model load is introduced.

Install the updated requirements (`lm_eval[hf]==0.4.13` is included). Benchmark
datasets are downloaded on first use and subsequently use the Hugging Face cache.
GPQA requires accepting the dataset terms at
<https://huggingface.co/datasets/Idavidrein/gpqa>. Use the project's configured HF
key with access to that dataset, or an existing `HF_TOKEN`/Hugging Face login.
Progress and cancellation checks occur between benchmark scoring batches.
A benchmark failure fails evaluation instead of silently reporting a missing score.

The fixed benchmark protocol is:

- **MMLU:** `mmlu`, 5-shot, full test set, sample-weighted global `acc` from the
  harness's MMLU group.
- **GPQA:** `gpqa_main_zeroshot`, 0-shot, full GPQA Main set, global `acc`.

Both use multiple-choice likelihood scoring with the harness task prompts and
fixed seeds, without applying a chat template. No sample limits are applied.
Only `benchmarks.mmlu` and `benchmarks.gpqa` (accuracy on a 0–1 scale) are returned
under each of `pre_unlearning` and `post_unlearning`, displayed in the dashboard,
and saved in the experiment JSONL record. Subject scores, individual answers,
normalized-accuracy alternatives, and standard errors are not displayed or saved.

See the upstream [Python API](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md)
and [GPQA task documentation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gpqa/README.md).
