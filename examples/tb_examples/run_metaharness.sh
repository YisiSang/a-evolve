#!/usr/bin/env bash
set -euo pipefail

# MetaHarness experiment on Terminal-Bench 2.0
#
# Reproduces the experiment from "Meta-Harness: End-to-End Optimization
# of Model Harnesses" (Lee et al., 2026, arXiv:2603.28052).
#
# Usage:
#   bash examples/tb_examples/run_metaharness.sh <RUN_NAME>
#   bash examples/tb_examples/run_metaharness.sh opus_v1 --workers 8
#   bash examples/tb_examples/run_metaharness.sh test --eval-sample-size 2 --max-cycles 1 --workers 1
#   nohup bash examples/tb_examples/run_metaharness.sh opus_v1 &

RUN_NAME="${1:?Usage: $0 <RUN_NAME> [--workers N] [--max-cycles N] [--eval-sample-size N] [--solver-model MODEL] [--phase PHASE]}"
shift

CONFIG="examples/configs/metaharness_tbench2.yaml"
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)          EXTRA_FLAGS="$EXTRA_FLAGS --workers $2";          shift 2 ;;
        --max-cycles)       EXTRA_FLAGS="$EXTRA_FLAGS --max-cycles $2";       shift 2 ;;
        --eval-sample-size) EXTRA_FLAGS="$EXTRA_FLAGS --eval-sample-size $2"; shift 2 ;;
        --solver-model)     EXTRA_FLAGS="$EXTRA_FLAGS --solver-model $2";     shift 2 ;;
        --phase)            EXTRA_FLAGS="$EXTRA_FLAGS --phase $2";            shift 2 ;;
        --task-limit)       EXTRA_FLAGS="$EXTRA_FLAGS --task-limit $2";       shift 2 ;;
        --config)           CONFIG="$2";                                      shift 2 ;;
        --skip-tasks)       EXTRA_FLAGS="$EXTRA_FLAGS --skip-tasks $2";       shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

LOG_DIR="logs/metaharness_${RUN_NAME}"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  MetaHarness Experiment: ${RUN_NAME}"
echo "  Config:  ${CONFIG}"
echo "  Logs:    ${LOG_DIR}"
echo "============================================================"
echo ""

# --- Prerequisites ---
echo ">>> Checking prerequisites..."

# Claude Code CLI
if ! command -v claude &>/dev/null; then
    echo "ERROR: 'claude' CLI not found. Install it: https://docs.anthropic.com/en/docs/claude-code"
    exit 1
fi
echo "  claude CLI: $(claude --version 2>/dev/null || echo 'found')"

# Docker
if ! command -v docker &>/dev/null; then
    echo "ERROR: 'docker' not found."
    exit 1
fi
echo "  docker: $(docker --version | head -1)"

# uv
if ! command -v uv &>/dev/null; then
    echo "ERROR: 'uv' not found."
    exit 1
fi
echo "  uv: $(uv --version)"

echo ""

# --- Download challenges ---
echo ">>> Ensuring Terminal-Bench 2.0 challenges are downloaded..."
bash agent_evolve/benchmarks/tb2/download_challenges.sh 2>&1 | tail -2
echo ""

# --- Run experiment ---
echo ">>> Running MetaHarness experiment..."
UV_CACHE_DIR=/tmp/uv_cache uv run python examples/tb_examples/run_metaharness.py \
    --config "$CONFIG" \
    --run-name "$RUN_NAME" \
    $EXTRA_FLAGS \
    2>&1 | tee "$LOG_DIR/experiment.log"

echo ""
echo "============================================================"
echo "  Experiment complete: ${RUN_NAME}"
echo "  Workspace: evolution_workdir/metaharness_${RUN_NAME}"
echo "  Logs:      ${LOG_DIR}/experiment.log"
echo "============================================================"
