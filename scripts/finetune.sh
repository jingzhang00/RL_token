#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-${WORKSPACE}/openpi}"
PYTHON="${OPENPI_PYTHON:-${WORKSPACE}/.venv/bin/python}"
CONFIG="${RLT_CONFIG:-${WORKSPACE}/rlt/config/config.yaml}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "OPENPI_PYTHON is not executable: ${PYTHON}" >&2
    exit 1
fi

WANDB_ENV_FILE="${WANDB_ENV_FILE:-${WORKSPACE}/wandb.env}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
    set -a
    source "${WANDB_ENV_FILE}"
    set +a
fi

cd "${WORKSPACE}"
export RLT_CONFIG="${CONFIG}"
export PYTHONPATH="${WORKSPACE}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"

CONFIG_NAME="$("${PYTHON}" -c 'from rlt.settings import load_config; print(load_config().vla["config_name"])')"
ENABLED="$("${PYTHON}" -c 'from rlt.settings import load_config; print(str(load_config().vla_finetune.get("enabled", True)).lower())')"
if [[ "${ENABLED}" != "true" ]]; then
    echo "vla.finetune.enabled is false in ${CONFIG}; refusing to start finetuning" >&2
    exit 2
fi

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
exec env \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION}" \
    "${PYTHON}" -m robots.d1.train "${CONFIG_NAME}" "$@"
