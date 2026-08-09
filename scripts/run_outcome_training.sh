#!/bin/bash
set -euo pipefail

REPO_DIR="/Users/matthewbarlowe/code/python/mlb"
UV_BIN="/usr/local/bin/uv"

resolve_mlflow_uri() {
  if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    printf '%s' "$MLFLOW_TRACKING_URI"
    return 0
  fi
  zsh -ic 'printf %s "$MLFLOW_TRACKING_URI"' </dev/null 2>/dev/null
}

MLFLOW_URI="$(resolve_mlflow_uri)"
if [[ -z "$MLFLOW_URI" ]]; then
  echo "MLFLOW_TRACKING_URI is not set in this shell or your zsh config" >&2
  exit 1
fi
export MLFLOW_TRACKING_URI="$MLFLOW_URI"

if [[ "${1:-}" == "--print-mlflow-uri" ]]; then
  printf '%s\n' "$MLFLOW_TRACKING_URI"
  exit 0
fi

cd "$REPO_DIR"
exec "$UV_BIN" run python scripts/train_outcome_models.py "$@"
