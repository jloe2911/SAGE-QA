#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUTF8=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python experiments/cross_encoder_reranking_dev_v1/run_experiment.py run
