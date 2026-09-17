#!/usr/bin/env bash
set -Eeuo pipefail

declare -a service_pids=()

cleanup() {
  trap - EXIT INT TERM
  for pid in "${service_pids[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done
  wait "${service_pids[@]}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

setsid uv run fastapi dev &
service_pids+=("$!")

setsid npm --prefix web run dev &
service_pids+=("$!")

wait -n "${service_pids[@]}"
