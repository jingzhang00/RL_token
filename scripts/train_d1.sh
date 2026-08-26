#!/usr/bin/env bash
set -euo pipefail

OPENPI_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-${OPENPI_WORKSPACE}/openpi}"
OPENPI_PYTHON="${OPENPI_PYTHON:-${OPENPI_ROOT}/.venv/bin/python}"

if [[ ! -x "${OPENPI_PYTHON}" ]]; then
    echo "OPENPI_PYTHON is not executable: ${OPENPI_PYTHON}" >&2
    exit 1
fi

WANDB_ENV_FILE="${WANDB_ENV_FILE:-${OPENPI_ROOT}/wandb.env}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
    set -a
    source "${WANDB_ENV_FILE}"
    set +a
fi

CONFIG_NAME="${CONFIG_NAME:-pi05_d1}"
EXP_NAME="${EXP_NAME:-d1}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${OPENPI_ROOT}/checkpoints}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${KEEP_PERIOD:-1000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FSDP_DEVICES="${FSDP_DEVICES:-2}"
PROJECT_NAME="${PROJECT_NAME:-openpi}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
OVERWRITE="${OVERWRITE:-true}"
RESUME="${RESUME:-false}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

export D1_REPO_ID="${D1_REPO_ID:-${OPENPI_WORKSPACE}/../data_processed/pick_up_the_doll_and_place_it_on_the_plate}"
export PI05_BASE_CKPT="${PI05_BASE_CKPT:-/home/ai01/Downloads/imitation_learning/openpi/train/model/openpi-assets/checkpoints/pi05_base/params}"

case "${WANDB_ENABLED,,}" in
    1|true|yes) WANDB_FLAG="--wandb-enabled" ;;
    0|false|no) WANDB_FLAG="--no-wandb-enabled" ;;
    *) echo "WANDB_ENABLED must be true or false" >&2; exit 2 ;;
esac

case "${OVERWRITE,,}" in
    1|true|yes) OVERWRITE_FLAG="--overwrite" ;;
    0|false|no) OVERWRITE_FLAG="--no-overwrite" ;;
    *) echo "OVERWRITE must be true or false" >&2; exit 2 ;;
esac

case "${RESUME,,}" in
    1|true|yes) RESUME_FLAG="--resume" ;;
    0|false|no) RESUME_FLAG="--no-resume" ;;
    *) echo "RESUME must be true or false" >&2; exit 2 ;;
esac

cd "${OPENPI_WORKSPACE}"
export PYTHONPATH="${OPENPI_WORKSPACE}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"

exec env \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION}" \
    "${OPENPI_PYTHON}" -m d1.train "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --project-name="${PROJECT_NAME}" \
    --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}" \
    --num-train-steps="${NUM_TRAIN_STEPS}" \
    --batch-size="${BATCH_SIZE}" \
    --save-interval="${SAVE_INTERVAL}" \
    --keep-period="${KEEP_PERIOD}" \
    --num-workers="${NUM_WORKERS}" \
    --fsdp-devices="${FSDP_DEVICES}" \
    "${WANDB_FLAG}" \
    "${OVERWRITE_FLAG}" \
    "${RESUME_FLAG}"
