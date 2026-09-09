#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Override PYTHON_BIN if the virtual environment
# uses a different path. Pass a dataset name to resume from that dataset.
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RESUME_FROM="${1:-}"
CHECKPOINT_ROOT="checkpoints/production_generator_d_v2_hard_pair"
LOG_ROOT="outputs/final_runs/production_generator_d_v2_hard_pair/training_logs"

datasets=(
  "HotpotQA:hotpotqa"
  "2WikiMultiHopQA:2wiki"
  "FamilyOWL_1hop:familyowl_1hop"
  "FamilyOWL_2hop:familyowl_2hop"
  "pizza_100_1hop:pizza_100_1hop"
  "pizza_100_2hop:pizza_100_2hop"
  "pizza_250_1hop:pizza_250_1hop"
  "pizza_250_2hop:pizza_250_2hop"
  "OWL2Bench_1hop:OWL2Bench_1hop"
  "OWL2Bench_2hop:OWL2Bench_2hop"
)

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi

if [[ -n "$RESUME_FROM" ]]; then
  found=false
  for entry in "${datasets[@]}"; do
    if [[ "${entry%%:*}" == "$RESUME_FROM" ]]; then
      found=true
      break
    fi
  done
  if [[ "$found" != true ]]; then
    echo "Unknown resume dataset: $RESUME_FROM" >&2
    exit 2
  fi
fi

mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUTF8=1

started=false
for entry in "${datasets[@]}"; do
  dataset="${entry%%:*}"
  slug="${entry#*:}"

  if [[ -z "$RESUME_FROM" || "$dataset" == "$RESUME_FROM" ]]; then
    started=true
  fi
  if [[ "$started" != true ]]; then
    continue
  fi

  train_path="data/production_generator_d_v1/$dataset/train_subgraph_retrieval.jsonl"
  dev_path="data/production_generator_d_v1/$dataset/dev_subgraph_retrieval.jsonl"
  save_dir="$CHECKPOINT_ROOT/$slug"
  log_path="$LOG_ROOT/$slug.log"

  echo "Starting dataset: $dataset"
  "$PYTHON_BIN" training/train_gnn_subgraph_retriever.py \
    --train-path "$train_path" \
    --dev-path "$dev_path" \
    --save-dir "$save_dir" \
    --model-name google/bert_uncased_L-2_H-128_A-2 \
    --lr 2e-5 \
    --epochs 3 \
    --max-length 128 \
    --candidate-batch-size 512 \
    --freeze-encoder \
    --ranking-margin 0.2 \
    --ranking-weight 2.0 \
    --bce-weight 0.2 \
    --listwise-weight 0.0 \
    --max-pairs 512 \
    --score-mode neural \
    --size-penalty 0.01 \
    --hard-pair-reservation \
    2>&1 | tee "$log_path"
done
