#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "${OPENPI_ROOT}/.." && pwd)"
D1_PYTHON_NAME="${D1_PYTHON:-python}"

if ! D1_PYTHON="$(command -v "${D1_PYTHON_NAME}")"; then
  echo "ERROR: D1_PYTHON is not executable: ${D1_PYTHON_NAME}" >&2
  exit 1
fi
ARGS=(
  --host "${OPENPI_HOST:-127.0.0.1}"
  --port "${OPENPI_PORT:-5555}"
  --prompt "${OPENPI_PROMPT:-Pick up the doll and place it on the plate.}"
  --dataset-root "${DATASET_ROOT:-${REPO_ROOT}/data_processed/pick_up_the_doll_and_place_it_on_the_plate}"
  --dataset-episode "${DATASET_EPISODE:-0}"
  --control-hz "${CONTROL_HZ:-10}"
  --max-steps "${MAX_STEPS:-500}"
  --chunk-steps "${OPENPI_CHUNK_STEPS:-20}"
  --log-every "${OPENPI_LOG_EVERY:-1}"
)

[[ "${MUJOCO_VIEWER:-1}" == "1" ]] && ARGS+=(--viewer) || ARGS+=(--no-viewer)
[[ "${IMAGE_PREVIEW:-1}" == "1" ]] && ARGS+=(--image-preview) || ARGS+=(--no-image-preview)
[[ "${IMAGE_CROP:-policy}" == "none" ]] && ARGS+=(--no-image-crop)
[[ -n "${DATASET_CAMERA_KEY:-}" ]] && ARGS+=(--camera-key "${DATASET_CAMERA_KEY}")
[[ -n "${D1_URDF:-}" ]] && ARGS+=(--urdf "${D1_URDF}")

cd "${REPO_ROOT}"
exec "${D1_PYTHON}" "${OPENPI_ROOT}/inference/reply/d1_mujoco_dataset_client.py" "${ARGS[@]}" "$@"
