# Ascent Unlearning Frontend

Vite and TypeScript frontend for the FastAPI unlearning backend.

## Run with Docker

Start the backend from the repository root:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Start the frontend in another shell with the base Docker CLI:

```bash
cd "front end"
docker build -t ascent-unlearning-frontend .
docker run --rm -it \
  --add-host=host.docker.internal:host-gateway \
  -p 5173:5173 \
  -e VITE_API_BASE_URL=/api \
  -e API_PROXY_TARGET=http://host.docker.internal:8000 \
  ascent-unlearning-frontend
```

Open `http://localhost:5173`.

The Vite dev server proxies browser calls from `/api` to `http://host.docker.internal:8000` inside Docker. To point at another backend, change `API_PROXY_TARGET` in the `docker run` command:

```bash
API_PROXY_TARGET=http://192.168.1.20:8000
```

If Docker Compose is available on your machine, this folder also includes `docker-compose.yml`, so `docker compose up --build` works from inside `front end`.

## What it supports

- Upload forget and optional retain datasets.
- Preview processed rows and saved parquet paths.
- Configure model method, GPU, adapter path, and Hugging Face token.
- Configure validated hyperparameters for max steps or epochs.
- Build the orchestrator config through `/config/build`.
- Launch training through `/train/run`.
