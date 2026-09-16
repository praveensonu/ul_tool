# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY ["front end/package.json", "front end/package-lock.json", "./"]
RUN npm ci
COPY ["front end/", "./"]
RUN VITE_API_URL= npm run build

FROM ghcr.io/astral-sh/uv:0.8.22 AS uv

FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/hf-cache \
    CORS_ALLOWED_ORIGINS="" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx tini build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY requirements.txt ./
RUN uv venv /app/.venv \
    && uv pip install --no-cache --python /app/.venv/bin/python -r requirements.txt
COPY . .
COPY --from=frontend /frontend/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/nginx.conf
RUN mkdir -p /app/outputs /app/uploaded_datasets /app/hf-cache \
    && chmod +x /app/docker/entrypoint.sh
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=4).read()"]
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/app/docker/entrypoint.sh"]
