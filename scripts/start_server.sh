#!/usr/bin/env bash
set -euo pipefail

OPENPI_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_PYTHON="${OPENPI_PYTHON:-${OPENPI_WORKSPACE}/.venv/bin/python}"
OPENPI_ROOT="${OPENPI_ROOT:-${OPENPI_WORKSPACE}/openpi}"
RLT_CONFIG="${RLT_CONFIG:-${OPENPI_WORKSPACE}/rlt/config/config.yaml}"
export RLT_CONFIG
POLICY_CONFIG="${POLICY_CONFIG:-$(${OPENPI_PYTHON} -c 'from rlt.settings import load_config; print(load_config().vla["config_name"])')}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(${OPENPI_PYTHON} -c 'from rlt.settings import load_config; print(load_config().vla_checkpoint)')}"
DEFAULT_PROMPT="$(${OPENPI_PYTHON} -c 'from rlt.settings import load_config; print(load_config().vla["prompt"])')"


export OPENPI_WORKSPACE
export OPENPI_ROOT
export CHECKPOINT_DIR
export PYTHONPATH="${OPENPI_WORKSPACE}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

cd "${OPENPI_WORKSPACE}"
echo "Starting OpenPI D1 server from ${OPENPI_WORKSPACE}"
exec "${OPENPI_PYTHON}" "${OPENPI_WORKSPACE}/inference/serve_d1_policy.py" \
  --host "${OPENPI_HOST:-127.0.0.1}" \
  --port "${OPENPI_PORT:-5555}" \
  --config "${POLICY_CONFIG}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  "$@"
