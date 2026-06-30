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

Browser requests use `VITE_API_BASE_URL`, which defaults to:

```text
/api
```

During local development, Vite proxies `/api` to the backend target configured by:

```text
API_PROXY_TARGET=http://host.docker.internal:8000
```

The proxy is defined in `front end/vite.config.ts`.

## API Calls

The API helper file is `front end/src/api.ts`.

The frontend uses these backend endpoints:

```text
GET  /
POST /dataset/upload
POST /config/build
POST /train/run
```

### Dataset Upload

`POST /dataset/upload` receives:

- `forget_set`: required file.
- `retain_set`: optional file.
- `prompt_template`: required text.

The frontend expects the response to include:

- `forget_set_path`
- `retain_set_path`
- row counts
- preview rows

These processed parquet paths are then passed into `/config/build` and `/train/run`.

### Config Build

`POST /config/build` validates the complete training request and returns the final orchestrator JSON.

The UI displays this JSON so the user can inspect the exact payload before launching training.

### Training Run

`POST /train/run` sends the same validated training request and waits for the backend response.

The current backend is synchronous, so the frontend shows a running state until the request returns. The architecture document notes future progress streaming; when that is added, this UI can be extended with a live progress panel.

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

## Form Behavior

The frontend keeps a local request payload in React state after datasets are uploaded.

Important rules reflected in the UI:

- Forget dataset is required.
- Retain dataset is optional.
- Method can be `full`, `lora`, or `adaptor`.
- LoRA target modules appear only when method is `lora`.
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
