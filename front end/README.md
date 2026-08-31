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
  -p 5173:5173 \
  -e VITE_API_URL=http://localhost:8000 \
  ascent-unlearning-frontend
```

Open `http://localhost:5173`.

The centralized client in `src/apiConfig.ts` appends `/api` to `VITE_API_URL`, and `src/api.ts` contains the reusable request functions. The URL must be reachable from the user's browser. To point at another backend, change it in the `docker run` command:

```bash
VITE_API_URL=http://192.168.1.20:8000
```

The backend allows `http://localhost:5173` by default. Set its comma-separated `CORS_ALLOWED_ORIGINS` environment variable when the frontend uses another origin. For a same-origin reverse-proxy deployment, build with an empty `VITE_API_URL`; the optional `API_PROXY_TARGET` setting supports the same arrangement during Vite development.

If Docker Compose is available on your machine, this folder also includes `docker-compose.yml`, so `docker compose up --build` works from inside `front end`.

## What it supports

- Upload forget and optional retain datasets.
- Preview processed rows and saved parquet paths.
- Configure model method, GPU, adapter path, and Hugging Face token.
- Configure validated hyperparameters for max steps or epochs.
- Build the orchestrator config through `/api/config/build`.
- Launch training through `/api/train/run`.
- Stop an active training process through `/api/train/stop`.
- Continue to a separate evaluation stage after unlearning succeeds.
- Select a sentence-transformers model by repository name or local path.
- Follow evaluation progress through `/api/evaluation/status/{job_id}`.
- Compare pre/post forget quality, utility, perplexity, ROUGE-L, conditional
  probability, and cosine similarity in the dashboard.
