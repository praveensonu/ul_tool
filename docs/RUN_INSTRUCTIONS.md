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

The frontend proxies API calls from `http://localhost:5173/api` to the backend on port `8000`. Press `Ctrl+C` to terminate the backend process group and stop the frontend container.

Ports and environment settings can be overridden if necessary:

```bash
BACKEND_PORT=8080 FRONTEND_PORT=5174 ./start.sh
```

See the root `README.md` for all supported overrides.

## Run Services Individually

Use the following commands when developing or debugging one service at a time. Start the backend before the frontend. If the backend is not running, browser actions such as dataset upload can fail with:

```text
Request failed with status 502
```

That `502` means the Vite frontend container is running, but its proxy cannot reach the FastAPI backend.

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
curl -s http://127.0.0.1:8000/
```

Expected response:

```json
{"message":"LLM training API is running"}
```

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
  --add-host=host.docker.internal:host-gateway \
  -p 5173:5173 \
  -e VITE_API_BASE_URL=/api \
  -e API_PROXY_TARGET=http://host.docker.internal:8000 \
  ascent-unlearning-frontend
```

Open:

```text
http://localhost:5173
```

The frontend expects the backend to be running on port `8000`.

### Stop Frontend Container

```bash
docker rm -f ascent-unlearning-frontend-dev
```

### Restart Frontend Container

```bash
docker rm -f ascent-unlearning-frontend-dev
docker run -d \
  --name ascent-unlearning-frontend-dev \
  --add-host=host.docker.internal:host-gateway \
  -p 5173:5173 \
  -e VITE_API_BASE_URL=/api \
  -e API_PROXY_TARGET=http://host.docker.internal:8000 \
  ascent-unlearning-frontend
```

### Check Frontend Container

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

If this returns `502 Bad Gateway`, start the backend with:

```bash
.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Use A Different Backend URL

Change `API_PROXY_TARGET` when starting the container:

```bash
docker run -d \
  --name ascent-unlearning-frontend-dev \
  --add-host=host.docker.internal:host-gateway \
  -p 5173:5173 \
  -e VITE_API_BASE_URL=/api \
  -e API_PROXY_TARGET=http://192.168.1.20:8000 \
  ascent-unlearning-frontend
```

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
