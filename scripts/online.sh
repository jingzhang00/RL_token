#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${OPENPI_PYTHON:-${WORKSPACE}/.venv/bin/python}"
CONFIG="${RLT_CONFIG:-${WORKSPACE}/rlt/config/config.yaml}"

cd "${WORKSPACE}"
export RLT_CONFIG="${CONFIG}"
export PYTHONPATH="${WORKSPACE}:${WORKSPACE}/openpi/src:${WORKSPACE}/openpi/packages/openpi-client/src:${PYTHONPATH:-}"

TOKEN_CHECKPOINT="${RLT_TOKEN_CHECKPOINT:-$("${PYTHON}" -c 'from pathlib import Path; from rlt.settings import load_config; c=load_config(); print(Path(c.rl_token["output"]) / ("step_" + str(c.rl_token["steps"]) + ".pt"))')}"
FEATURE_HOST="$("${PYTHON}" -c 'from rlt.settings import load_config; print(load_config().runtime["feature_host"])')"
FEATURE_PORT="$("${PYTHON}" -c 'from rlt.settings import load_config; print(load_config().runtime["feature_port"])')"
FEATURE_LOG="${RLT_FEATURE_LOG:-/tmp/rlt_feature_server.log}"

cleanup() {
    if [[ -n "${FEATURE_PID:-}" ]] && kill -0 "${FEATURE_PID}" 2>/dev/null; then
        kill "${FEATURE_PID}" 2>/dev/null || true
        wait "${FEATURE_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "${RLT_START_FEATURE_SERVER:-true}" == "true" ]]; then
    "${PYTHON}" -m rlt.serve_features \
        --config "${CONFIG}" \
        --token "${TOKEN_CHECKPOINT}" \
        --host "${FEATURE_HOST}" \
        --port "${FEATURE_PORT}" \
        >"${FEATURE_LOG}" 2>&1 &
    FEATURE_PID=$!
    for _ in $(seq 1 100); do
        if ! kill -0 "${FEATURE_PID}" 2>/dev/null; then
            sed -n '1,80p' "${FEATURE_LOG}" >&2
            exit 1
        fi
        if (echo >/dev/tcp/${FEATURE_HOST}/${FEATURE_PORT}) 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
fi

exec "${PYTHON}" -m robots.d1.online_rl_d1 \
    --config "${CONFIG}" \
    --host "${FEATURE_HOST}" \
    --port "${FEATURE_PORT}" \
    "$@"
