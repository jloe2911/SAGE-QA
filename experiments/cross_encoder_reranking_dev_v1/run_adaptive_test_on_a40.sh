#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUTF8=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CROSS_DIR="outputs/development_runs/question_candidate_cross_encoder_v1"
CROSS_POLICY="outputs/development_runs/question_candidate_cross_encoder_v1_cross_encoder_adaptive_k"
FINAL_POLICY="outputs/development_runs/question_candidate_cross_encoder_v1_final_sageqa_adaptive_k"
OUTPUT_DIR="outputs/final_results/question_candidate_cross_encoder_v1_adaptive_test_a40"

python evaluation/preflight_cross_encoder_a40_test.py \
  --require-a40 \
  --cross-encoder-policy-dir "$CROSS_POLICY" \
  --final-sageqa-policy-dir "$FINAL_POLICY" \
  --output "$OUTPUT_DIR/a40_preflight_manifest.json"

python -m pytest -q -p no:cacheprovider \
  tests/test_adaptive_support_aggregation_v2.py \
  experiments/cross_encoder_reranking_dev_v1/test_experiment.py

python evaluation/run_cross_encoder_test_retrieval.py generate \
  --cross-encoder-dir "$CROSS_DIR" \
  --cross-encoder-policy-dir "$CROSS_POLICY" \
  --final-sageqa-policy-dir "$FINAL_POLICY" \
  --output-dir "$OUTPUT_DIR"

python evaluation/run_cross_encoder_test_retrieval.py evaluate \
  --cross-encoder-dir "$CROSS_DIR" \
  --cross-encoder-policy-dir "$CROSS_POLICY" \
  --final-sageqa-policy-dir "$FINAL_POLICY" \
  --output-dir "$OUTPUT_DIR"
