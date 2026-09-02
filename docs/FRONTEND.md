# Frontend Architecture

The frontend lives in `front end/` and is a Vite, React, and TypeScript application for the LLM unlearning FastAPI backend.

## Goals

- Provide a single workflow for dataset upload, configuration preview, and training launch.
- Avoid requiring host-level `npm`, since this project may run on a shared server without sudo access.
- Keep the interface minimal and work-focused with neutral colors and no purple gradients.
- Use the backend validation rules instead of duplicating all training constraints in the browser.

## Technology

```text
front end/
├── Dockerfile
├── README.md
├── docker-compose.yml
├── index.html
├── package.json
├── src
│   ├── App.tsx
│   ├── api.ts
│   ├── apiConfig.ts
│   ├── main.tsx
│   ├── styles.css
│   ├── types.ts
│   └── vite-env.d.ts
├── tsconfig.json
└── vite.config.ts
```

Core dependencies:

- `vite`
- `typescript`
- `react`
- `react-dom`
- `lucide-react`

The Docker image uses `node:22-alpine` and runs the Vite dev server on port `5173`.

## Backend Integration

`front end/src/apiConfig.ts` is the single source of truth for the backend URL. It reads the backend origin from:

```text
VITE_API_URL=http://localhost:8000
```

The helper appends the canonical `/api` prefix and exports `apiUrl(path)`. `front end/src/api.ts` owns request/response handling, and components import its typed functions instead of calling `fetch` or embedding backend URLs.

Example:

```ts
const response = await fetch(apiUrl("config/build"), {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload)
});
```

For a same-origin production deployment, set `VITE_API_URL` to an empty string so requests use `/api/*`. Vite environment variables are substituted when its dev server starts or when the frontend is built, so production builds must receive the intended value.

`VITE_API_BASE_URL` remains a compatibility fallback for existing deployments. New deployments should use `VITE_API_URL`. Vite also retains an optional same-origin development proxy configured by:

```text
API_PROXY_TARGET=http://127.0.0.1:8000
```

The proxy is used when `VITE_API_URL` is empty. It forwards `/api` unchanged because the backend now exposes the same prefix.

## API Calls

The API helper file is `front end/src/api.ts`.

The frontend uses these backend endpoints:

```text
GET  /api/health
POST /api/dataset/upload
POST /api/config/build
POST /api/train/run
POST /api/train/stop
POST /api/evaluation/start
GET  /api/evaluation/status/{job_id}
```

### Dataset Upload

`POST /api/dataset/upload` receives:

- `forget_set`: required file.
- `retain_set`: optional file.
- `prompt_template`: required text.

The frontend expects the response to include:

- `forget_set_path`
- `retain_set_path`
- row counts
- preview rows

These processed parquet paths are then passed into `/api/config/build` and `/api/train/run`.

### Config Build

`POST /api/config/build` validates the complete training request and returns the final orchestrator JSON.

The UI displays this JSON so the user can inspect the exact payload before launching training.

### Training Run

`POST /api/train/run` sends the same validated training request and waits for the backend response. The backend runs the training workload in a dedicated child process.

While the request is active, the frontend replaces the run button with a stop button. `POST /api/train/stop` terminates the active training child process without stopping FastAPI. The original run request then returns a stopped result. The request remains synchronous, so the frontend shows a running state until training finishes or is stopped. The architecture document notes future progress streaming; when that is added, this UI can be extended with a live progress panel.

### Evaluation

Evaluation is a separate fifth project stage unlocked after successful
unlearning. The user supplies a sentence-transformers repository name or local
path. The page starts a background job and polls its status endpoint to show
model loading, language-model scoring, model removal, cosine similarity,
ROUGE-L, and final aggregation updates.

On completion, grouped comparison plots and a detailed table show pre- and
post-unlearning forget quality, model utility, dataset perplexities,
conditional probabilities, ROUGE-L, and retain-set cosine similarity. Blue is
used consistently for pre-unlearning and green for post-unlearning.

## Local Development

The normal development origins are:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`

The browser calls FastAPI directly at `http://localhost:8000/api/*`. FastAPI explicitly allows the frontend origin through `CORS_ALLOWED_ORIGINS`; credentials are disabled because the application does not use cookies or HTTP authentication. Copy `.env.example` when a different frontend environment is needed, and restart Vite after changing its environment values.

## UI Workflow

The main application is implemented in `front end/src/App.tsx`.

The first screen is the actual training console, not a landing page.

Primary areas:

- Dataset upload and prompt template.
- Model settings.
- Hyperparameters.
- Dataset preview.
- Orchestrator config preview.
- Training result output.
- Evaluation configuration, live progress, and pre/post comparison dashboard.

## Form Behavior

The frontend keeps a local request payload in React state after datasets are uploaded.

Important rules reflected in the UI:

- Forget dataset is required.
- Retain dataset is optional, except for the unlearning methods that need one.
- Model loading method can be `full`, `lora`, or `adaptor`.
- LoRA target modules appear only when method is `lora`.
- The unlearning method dropdown on the hyperparameters stage is populated from
  `GET /api/config/unlearning-methods`.
- `grad_diff`, `npo`, and `dpo` block the hyperparameters stage until a retain set is
  selected.
- The schedule uses either `max_steps` or `epochs`, not both.
- `assistant_completions_only` is always sent as `true`.
- Context length is selected from power-of-two values.

Backend Pydantic validation remains the source of truth for invalid combinations.

## Styling

Styles are in `front end/src/styles.css`.

The palette is intentionally restrained:

- Off-white background.
- White surfaces.
- Muted green primary actions.
- Neutral borders.
- Dark green-black code preview.
- Soft red error state.

There are no gradient backgrounds and no purple color theme.

## Backend Adjustment

The frontend needs processed dataset paths from upload responses. To support that, the backend dataset upload response includes:

```python
forget_set_path: str
retain_set_path: Optional[str]
```

These fields are defined in `schemas.py` and returned from `routes/dataset_routes.py`.

## Build Verification

The frontend has been verified inside Docker with:

```bash
docker build -t ascent-unlearning-frontend-test "front end"
docker run --rm ascent-unlearning-frontend-test npm run build
```

The backend schema change has been checked with:

```bash
python -m py_compile schemas.py routes/dataset_routes.py
```
