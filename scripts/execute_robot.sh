#!/usr/bin/env bash
set -euo pipefail

OPENPI_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${OPENPI_WORKSPACE}/.." && pwd)"
D1_PYTHON_NAME="${D1_PYTHON:-python}"
PREVIEW_ARG=--image-preview
[[ "${IMAGE_PREVIEW:-1}" == "1" ]] || PREVIEW_ARG=--no-image-preview

cd "${REPO_ROOT}"
exec "${D1_PYTHON_NAME}" "${OPENPI_WORKSPACE}/inference/openpi_d1_client_fast.py" \
  --host "${OPENPI_HOST:-127.0.0.1}" \
  --port "${OPENPI_PORT:-5555}" \
  --prompt "${OPENPI_PROMPT:-Pick up the doll and place it on the plate.}" \
  --max-steps "${OPENPI_MAX_STEPS:-1000}" \
  --control-hz "${OPENPI_CONTROL_HZ:-10}" \
  --chunk-steps "${OPENPI_CHUNK_STEPS:-1}" \
  "${PREVIEW_ARG}" \
  --move-head-on-start \
  --log-every "${OPENPI_LOG_EVERY:-1}" \
  --execute \
  "$@"
