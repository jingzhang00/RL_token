#!/usr/bin/env bash
set -euo pipefail

OPENPI_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-${OPENPI_WORKSPACE}/openpi}"
OPENPI_PYTHON="${OPENPI_PYTHON:-${OPENPI_WORKSPACE}/.venv/bin/python}"
RLT_CONFIG="${RLT_CONFIG:-${OPENPI_WORKSPACE}/rlt/config/config.yaml}"

if [[ ! -x "${OPENPI_PYTHON}" ]]; then
    echo "OPENPI_PYTHON is not executable: ${OPENPI_PYTHON}" >&2
    exit 1
fi

WANDB_ENV_FILE="${WANDB_ENV_FILE:-${OPENPI_WORKSPACE}/wandb.env}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
    set -a
    source "${WANDB_ENV_FILE}"
    set +a
fi

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

cd "${OPENPI_WORKSPACE}"
export PYTHONPATH="${OPENPI_WORKSPACE}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"

export RLT_CONFIG

case "${1:-}" in
    cache)
        shift
        exec "${OPENPI_PYTHON}" -m rlt.train_token cache --config "${RLT_CONFIG}" "$@"
        ;;
    train)
        shift
        exec env \
            XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION}" \
            "${OPENPI_PYTHON}" -m rlt.train_token train --config "${RLT_CONFIG}" "$@"
        ;;
    "")
        ;;
    *)
        echo "usage: $0 [cache|train] [options]" >&2
        exit 2
        ;;
esac

"${OPENPI_PYTHON}" -m rlt.train_token cache --config "${RLT_CONFIG}" "$@"
exec env \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION}" \
    "${OPENPI_PYTHON}" -m rlt.train_token train --config "${RLT_CONFIG}"
