# GPU execution handoff

Run from the repository root in a clean Linux GPU environment containing the
original v1.0.0 artifacts. These commands do not regenerate the Cross-Encoder
predictions; the GPU checkpoint is provenance input only.

```bash
set -euo pipefail

EXP=experiments/symbolic_coefficients_original_v1
DATA=data/production_generator_d_v1
CE=outputs/development_runs/question_candidate_cross_encoder_v1
RUN=outputs/development_runs/symbolic_coefficients_original_v1

# 0. Lightweight code validation (does not run the DEV pipeline).
python -m pytest -q tests/test_symbolic_coefficients_original_v1.py
python -m compileall -q "$EXP"

# 1. Validate original frozen DEV artifacts and their recorded hashes.
python "$EXP/prepare_dev_cache.py" validate-inputs \
  --predictions "$CE/dev_predictions_frozen.jsonl" \
  --prediction-freeze "$CE/prediction_freeze.json" \
  --data-root "$DATA" \
  --source-manifest "$CE/preflight_manifest.json"

# 2. Prepare and validate the DEV feature cache.
python "$EXP/prepare_dev_cache.py" prepare \
  --predictions "$CE/dev_predictions_frozen.jsonl" \
  --prediction-freeze "$CE/prediction_freeze.json" \
  --data-root "$DATA" \
  --source-manifest "$CE/preflight_manifest.json" \
  --output-dir "$RUN/cache_v1"
python "$EXP/prepare_dev_cache.py" validate-cache \
  --cache "$RUN/cache_v1/dev_feature_cache.jsonl" \
  --manifest "$RUN/cache_v1/cache_manifest.json"

# 3. Optimize Text-Chain coefficients only.
python "$EXP/optimize.py" search \
  --cache "$RUN/cache_v1/dev_feature_cache.jsonl" \
  --cache-manifest "$RUN/cache_v1/cache_manifest.json" \
  --domain text \
  --output-dir "$RUN/text_search_v1"

# 4. Optimize Proof-aware coefficients only.
python "$EXP/optimize.py" search \
  --cache "$RUN/cache_v1/dev_feature_cache.jsonl" \
  --cache-manifest "$RUN/cache_v1/cache_manifest.json" \
  --domain ontology \
  --output-dir "$RUN/ontology_search_v1"

# 5. Compare selected and original coefficients.
python "$EXP/optimize.py" compare \
  --text-result "$RUN/text_search_v1/text_optimization.json" \
  --ontology-result "$RUN/ontology_search_v1/ontology_optimization.json" \
  --output "$RUN/original_vs_selected_dev.json"

# 6. Freeze the selected coefficients and provenance.
python "$EXP/optimize.py" freeze \
  --text-result "$RUN/text_search_v1/text_optimization.json" \
  --ontology-result "$RUN/ontology_search_v1/ontology_optimization.json" \
  --cache-manifest "$RUN/cache_v1/cache_manifest.json" \
  --output-dir "$RUN/frozen_coefficients_v1"

# 7. Generate selected-coefficient DEV rankings only.
python "$EXP/materialize_dev_rankings.py" \
  --cache "$RUN/cache_v1/dev_feature_cache.jsonl" \
  --cache-manifest "$RUN/cache_v1/cache_manifest.json" \
  --coefficients "$RUN/frozen_coefficients_v1/selected_coefficients.json" \
  --freeze-manifest "$RUN/frozen_coefficients_v1/freeze_manifest.json" \
  --output-dir "$RUN/dev_rankings_v1"

# 8. Refit adaptive-k from the new DEV rankings. Do not add --overwrite.
# This gate must be clean: it prevents use of uncommitted abandoned-v1.1
# reader/evaluation-policy edits in the adaptive-k refit.
git diff --exit-code -- \
  evaluation/fit_production_dev_adaptive_k.py \
  evaluation/adaptive_support_aggregation.py \
  evaluation/adaptive_support_aggregation_v2.py \
  evaluation/evaluate_adaptive_support_aggregation_v2.py
python evaluation/fit_production_dev_adaptive_k.py \
  --input-dir "$RUN/dev_rankings_v1" \
  --output-dir "$RUN/adaptive_k_v1" \
  --ranking-method final_sageqa
```

Required inputs are the ten original
`data/production_generator_d_v1/<dataset>/dev_subgraph_retrieval.jsonl` files,
`dev_predictions_frozen.jsonl`, `prediction_freeze.json`, and
`preflight_manifest.json`. Preserve the Cross-Encoder
`checkpoint_final/model.safetensors` whose SHA-256 is recorded by
`prediction_freeze.json`; none of these commands loads or reruns it.
