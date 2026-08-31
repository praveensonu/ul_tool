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
BACKEND_PORT=8080 FRONTEND_PORT=5174 ./start.sh
```

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
  -e VITE_API_URL=http://localhost:8000 \
  ascent-unlearning-frontend
```

Open:

```text
http://localhost:5173
```

The frontend expects the backend to be running on port `8000`. `localhost` is correct here because `VITE_API_URL` is used by the browser, not by the frontend container.

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
  -e VITE_API_URL=http://192.168.1.20:8000 \
  ascent-unlearning-frontend
```

Also include `http://localhost:5173` (or the actual frontend origin) in the backend's comma-separated `CORS_ALLOWED_ORIGINS` value.

## Environment Configuration

The checked-in examples are `.env.example` for FastAPI and `front end/.env.example` for Vite.

- `CORS_ALLOWED_ORIGINS`: comma-separated exact browser origins allowed by FastAPI. It defaults to `http://localhost:5173,http://127.0.0.1:5173`. Set it in the backend process environment; Uvicorn does not read the example file automatically.
- `VITE_API_URL`: backend origin visible to the browser, without `/api`. It defaults to `http://localhost:8000` in the frontend client.
- `API_PROXY_TARGET`: optional Vite proxy target. It is only needed when `VITE_API_URL` is empty and the browser uses same-origin `/api` URLs.

For a same-origin production deployment, build the frontend with an empty `VITE_API_URL`, route `/api` to FastAPI in the reverse proxy, and set `CORS_ALLOWED_ORIGINS` to an empty value. For different public origins, set both variables to the exact public URLs; do not use `*`.

### Docker Compose Optional

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
