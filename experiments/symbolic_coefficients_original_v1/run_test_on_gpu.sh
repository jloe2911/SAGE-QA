#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${SAGEQA_DEV_REFERENCE:?Set SAGEQA_DEV_REFERENCE to the extracted immutable DEV reference directory}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUTF8=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RUNNER="experiments/symbolic_coefficients_original_v1/test_pipeline.py"
DATA="data/production_generator_d_v1"
CROSS="outputs/development_runs/question_candidate_cross_encoder_v1"
OUT="outputs/final_results/symbolic_coefficients_original_v1_test"

git diff --exit-code -- \
  experiments/symbolic_coefficients_original_v1/test_pipeline.py \
  experiments/symbolic_coefficients_original_v1/original_symbolic_features.py \
  evaluation/adaptive_support_aggregation.py \
  evaluation/adaptive_support_aggregation_v2.py \
  data_processing/retrieval_contracts.py \
  data_processing/build_hotpot_subgraph_dataset.py \
  data_processing/build_2wiki_subgraph_dataset.py

python -m pytest -q -p no:cacheprovider \
  tests/test_symbolic_coefficients_original_v1.py \
  tests/test_symbolic_coefficients_original_v1_test_pipeline.py

python "$RUNNER" preflight \
  --require-a40 \
  --reference-dir "$SAGEQA_DEV_REFERENCE" \
  --data-root "$DATA" \
  --cross-encoder-dir "$CROSS" \
  --output-dir "$OUT"

python "$RUNNER" generate \
  --reference-dir "$SAGEQA_DEV_REFERENCE" \
  --data-root "$DATA" \
  --cross-encoder-dir "$CROSS" \
  --output-dir "$OUT"

# Evaluation is intentionally not automatic. Run the evaluate command supplied
# in the handoff only after the complete prediction freeze has been reviewed.
