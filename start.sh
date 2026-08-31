#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="${ROOT_DIR}/front end"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
FRONTEND_IMAGE="${FRONTEND_IMAGE:-ascent-unlearning-frontend}"
FRONTEND_CONTAINER="${FRONTEND_CONTAINER:-ascent-unlearning-frontend-dev}"
VITE_API_URL="${VITE_API_URL-http://localhost:${BACKEND_PORT}}"
API_PROXY_TARGET="${API_PROXY_TARGET:-http://host.docker.internal:${BACKEND_PORT}}"
CORS_ALLOWED_ORIGINS="${CORS_ALLOWED_ORIGINS-http://localhost:${FRONTEND_PORT},http://127.0.0.1:${FRONTEND_PORT}}"

backend_pid=""
frontend_pid=""

cleanup() {
  local exit_status=$?

  trap - EXIT INT TERM
  set +e

  if [[ -n "${backend_pid}" ]] && kill -0 -- "-${backend_pid}" 2>/dev/null; then
    # The backend runs in its own process group so the uvicorn reloader and server
    # process receive the same termination signal.
    kill -TERM -- "-${backend_pid}" 2>/dev/null
  elif [[ -n "${backend_pid}" ]] && kill -0 "${backend_pid}" 2>/dev/null; then
    kill -TERM "${backend_pid}" 2>/dev/null
  fi

  if [[ -n "${frontend_pid}" ]]; then
    docker stop --time 10 "${FRONTEND_CONTAINER}" >/dev/null 2>&1
    kill -TERM "${frontend_pid}" 2>/dev/null
  fi

  [[ -n "${backend_pid}" ]] && wait "${backend_pid}" 2>/dev/null
  [[ -n "${frontend_pid}" ]] && wait "${frontend_pid}" 2>/dev/null

  # Cover the narrow case where Docker created the container while the first
  # stop request and the docker client were racing during early shutdown.
  if [[ -n "${frontend_pid}" ]] && docker container inspect "${FRONTEND_CONTAINER}" >/dev/null 2>&1; then
    docker rm -f "${FRONTEND_CONTAINER}" >/dev/null 2>&1
  fi

  exit "${exit_status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Backend Python executable not found: %s\n' "${PYTHON_BIN}" >&2
  printf 'Create .venv or set PYTHON_BIN to a Python environment with the backend dependencies.\n' >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  printf 'Docker is required to run the frontend.\n' >&2
  exit 1
fi

if docker container inspect "${FRONTEND_CONTAINER}" >/dev/null 2>&1; then
  printf 'A Docker container named %s already exists. Remove it before starting the application.\n' "${FRONTEND_CONTAINER}" >&2
  exit 1
fi

printf 'Building frontend image %s...\n' "${FRONTEND_IMAGE}"
docker build -t "${FRONTEND_IMAGE}" "${FRONTEND_DIR}"

printf 'Starting backend on http://localhost:%s...\n' "${BACKEND_PORT}"
# Job control gives this background job its own process group. Turning it off
# again keeps the rest of the supervisor non-interactive and quiet.
set -m
(
  cd "${ROOT_DIR}"
  CORS_ALLOWED_ORIGINS="${CORS_ALLOWED_ORIGINS}" exec "${PYTHON_BIN}" -m uvicorn main:app \
    --host "${BACKEND_HOST}" \
    --port "${BACKEND_PORT}" \
    --reload
) &
backend_pid=$!
set +m

printf 'Starting frontend on http://localhost:%s...\n' "${FRONTEND_PORT}"
docker run --rm \
  --name "${FRONTEND_CONTAINER}" \
  --add-host=host.docker.internal:host-gateway \
  -p "${FRONTEND_PORT}:5173" \
  -e "VITE_API_URL=${VITE_API_URL}" \
  -e "API_PROXY_TARGET=${API_PROXY_TARGET}" \
  "${FRONTEND_IMAGE}" &
frontend_pid=$!

printf 'Both services are running. Press Ctrl+C to stop them.\n'

set +e
wait -n "${backend_pid}" "${frontend_pid}"
service_status=$?
set -e

if (( service_status != 0 )); then
  printf 'A service exited with status %s; stopping the other service.\n' "${service_status}" >&2
else
  printf 'A service stopped; stopping the other service.\n'
fi

exit "${service_status}"
