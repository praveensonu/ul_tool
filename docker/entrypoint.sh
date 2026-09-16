#!/usr/bin/env bash
set -Eeuo pipefail

backend_pid=""
frontend_pid=""
cleanup() {
    trap - EXIT INT TERM
    [[ -z "$backend_pid" ]] || kill -TERM "$backend_pid" 2>/dev/null || true
    [[ -z "$frontend_pid" ]] || kill -TERM "$frontend_pid" 2>/dev/null || true
    wait || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python -m uvicorn main:app --host 127.0.0.1 --port 8000 &
backend_pid=$!
nginx -g 'daemon off;' &
frontend_pid=$!

# Stop the container if either service exits; tini forwards container signals.
wait -n "$backend_pid" "$frontend_pid"
