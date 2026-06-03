# SAGE-QA

SAGE-QA is a retrieval-augmented question answering pipeline for multi-hop
reasoning over text and ontology-style knowledge graphs. This repository
contains the code used for the manuscript experiments on HotpotQA,
2WikiMultiHopQA, FamilyOWL, and OWL2Bench.

The pipeline builds retrieval datasets, trains a GNN subgraph retriever,
evaluates lexical/GNN/SAGE-QA/GNN-RAG retrieval variants, generates answers with
an OpenAI-compatible LLM reader, and computes final answer, support, retrieval,
and joint metrics.

## Reproduction Checklist

This repository is intended to support result reproduction.

- Main runner: `experiments/run_experiments.py`
- Dataset builders:
  - `data/build_subgraph_training_data.py` for FamilyOWL
  - `data/build_subgraph_training_data.py` for OWL2Bench
  - `data_processing/build_hotpot_subgraph_dataset.py` for HotpotQA
  - `data_processing/build_2wiki_subgraph_dataset.py` for 2WikiMultiHopQA
- Evaluation scripts: `evaluation/`
- GNN training script: `training/train_gnn_subgraph_retriever.py`
- Adapted GNN-RAG baseline: `third_party/GNN-RAG/`
- Expected final result files:
  - `outputs/full_results/full_pipeline_results.csv`
  - `outputs/full_results/full_pipeline_results.json`
- Per-dataset/method outputs:
  - `outputs/full_results/<Dataset>/<method>/test_details.json`
  - `outputs/full_results/<Dataset>/<method>/llm_answers*.jsonl`
  - `outputs/full_results/<Dataset>/<method>/metrics*.json`

Large benchmark files, processed datasets, checkpoints, and generated outputs
are not committed by default. They can be regenerated with the commands below.

## Repository Layout

```text
data/                       Raw and processed datasets
data_processing/            HotpotQA, 2Wiki, and GNN-RAG data adapters
training/                   GNN retriever training and symbolic utilities
models/                     GNN and symbolic retriever components
generation/                 LLM answer generation scripts
evaluation/                 Retrieval, QA, support, and error evaluators
experiments/                End-to-end experiment runner
third_party/GNN-RAG/        Adapted upstream GNN-RAG baseline
patches/                    Notes and patch for local GNN-RAG modifications
checkpoints/                Generated model checkpoints
outputs/                    Generated experiment outputs
```

## Environment

This repository was prepared on Windows with PowerShell and Python 3.12. The same
commands can be translated to Bash by changing activation syntax and line
continuations.

Recommended setup:

- Python 3.12, tested with Python 3.12.7
- A virtual environment using `requirements.txt`
- Network access for Hugging Face downloads unless data and models are cached
- `OPENAI_API_KEY` or `OPENROUTER_API_KEY` for LLM-based KG construction and
  answer generation
- CUDA-capable GPU for faster GNN training; CPU is usable for smoke tests

Create the environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `hf download` or Parquet loading is unavailable in a fresh environment,
install the missing helper packages:

```powershell
pip install "huggingface_hub[cli]" pyarrow
```

Set one LLM API key:

```powershell
$env:OPENROUTER_API_KEY="sk-or-..."
# or
$env:OPENAI_API_KEY="sk-..."
```

The code accepts provider-prefixed model names:

- `openrouter:openai/gpt-4.1-mini`
- `openrouter:google/gemini-2.5-flash-lite`
- `openai:gpt-4.1-mini`

Without a prefix, OpenRouter is used when `OPENROUTER_API_KEY` is set;
otherwise OpenAI is used.

## Data Policy

Committed or expected raw assets:

- Source code
- `data/raw/family.zip`
- `data/raw/owl2bench.zip`
- GNN-RAG local adaptation files under `third_party/GNN-RAG/`
- Local GNN-RAG patch notes under `patches/`

Downloaded externally:

- HotpotQA Parquet files
- 2WikiMultiHopQA Parquet files

Generated locally:

- `data/FamilyOWL_1hop/`
- `data/FamilyOWL_2hop/`
- `data/OWL2Bench_1hop/`
- `data/OWL2Bench_2hop/`
- `data/HotpotQA/`
- `data/2WikiMultiHopQA/`
- `checkpoints/`
- `outputs/`

## Download Raw Data

Extract FamilyOWL:

```powershell
Expand-Archive -Path data/raw/family.zip -DestinationPath data/raw -Force
```

Extract OWL2Bench:

```powershell
Expand-Archive -Path data/raw/owl2bench.zip -DestinationPath data/raw -Force
```

Download HotpotQA and 2WikiMultiHopQA with the Hugging Face CLI:

```powershell
hf download hotpotqa/hotpot_qa distractor/ `
  --repo-type dataset `
  --local-dir data/raw/hotpot_qa

hf download framolfese/2WikiMultihopQA data/ `
  --repo-type dataset `
  --local-dir data/raw/2WikiMultihopQA
```

The text benchmark commands below use the labeled validation Parquet file as the
evaluation source. The public 2Wiki test Parquet file in this distribution does
not include usable gold answers/supporting facts for supervised evaluation.

## Build Retrieval Datasets

### FamilyOWL

```powershell
python data/build_subgraph_training_data.py `
  --input-json data/raw/family/FamilyOWL_1hop.json `
  --output-dir data

python data/build_subgraph_training_data.py `
  --input-json data/raw/family/FamilyOWL_2hop.json `
  --output-dir data
```

This creates:

```text
data/FamilyOWL_1hop/{train,dev,test}_subgraph_retrieval.jsonl
data/FamilyOWL_2hop/{train,dev,test}_subgraph_retrieval.jsonl
data/FamilyOWL_*/{train,dev,test}_answer_only_no_explanation.jsonl
```

Answer-only binary examples are used only for OWL answer EM/F1. They are
excluded from support, retrieval, and joint metrics.

### OWL2Bench

OWL2Bench uses the same ontology-style retrieval builder as FamilyOWL:

```powershell
python data/build_subgraph_training_data.py `
  --input-json data/raw/owl2bench/OWL2Bench_1hop.json `
  --output-dir data

python data/build_subgraph_training_data.py `
  --input-json data/raw/owl2bench/OWL2Bench_2hop.json `
  --output-dir data
```

This creates:

```text
data/OWL2Bench_1hop/{train,dev,test}_subgraph_retrieval.jsonl
data/OWL2Bench_2hop/{train,dev,test}_subgraph_retrieval.jsonl
data/OWL2Bench_*/{train,dev,test}_answer_only_no_explanation.jsonl
```

### HotpotQA

```powershell
python data_processing/build_hotpot_subgraph_dataset.py `
  --train-file data/raw/hotpot_qa/distractor/train-00000-of-00002.parquet data/raw/hotpot_qa/distractor/train-00001-of-00002.parquet `
  --dev-file data/raw/hotpot_qa/distractor/validation-00000-of-00001.parquet `
  --output-dir data/HotpotQA `
  --max-train-examples 3000 `
  --max-dev-examples 500 `
  --max-test-examples 1000 `
  --max-sentences-per-example 30 `
  --max-subgraph-size 3 `
  --max-kg-bridge-triples 64 `
  --kg-construction-backend llm `
  --kg-construction-model openrouter:google/gemini-2.5-flash-lite `
  --kg-construction-workers 3 `
  --kg-request-timeout 90 `
  --kg-max-retries 6 `
  --kg-retry-initial-sleep 10 `
  --max-candidates-per-question 256 `
  --seed 42
```

### 2WikiMultiHopQA

```powershell
python data_processing/build_2wiki_subgraph_dataset.py `
  --train-file data/raw/2WikiMultihopQA/data/train-00000-of-00002.parquet data/raw/2WikiMultihopQA/data/train-00001-of-00002.parquet `
  --dev-file data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet `
  --output-dir data/2WikiMultiHopQA `
  --max-train-examples 3000 `
  --max-dev-examples 500 `
  --max-test-examples 1000 `
  --max-sentences-per-example 30 `
  --max-subgraph-size 4 `
  --max-kg-bridge-triples 64 `
  --kg-construction-backend llm_with_provided `
  --kg-construction-model openrouter:google/gemini-2.5-flash-lite `
  --kg-construction-workers 3 `
  --kg-request-timeout 90 `
  --kg-max-retries 6 `
  --kg-retry-initial-sleep 10 `
  --max-candidates-per-question 256 `
  --seed 42
```

The text builders create:

```text
data/HotpotQA/{train,dev,test}_subgraph_retrieval.jsonl
data/2WikiMultiHopQA/{train,dev,test}_subgraph_retrieval.jsonl
```

LLM KG construction caches successful calls in `<output-dir>/kg_cache/`, so
interrupted dataset builds can be resumed without repeating completed examples.
For a faster text build without API calls, omit the `--kg-construction-*`
options. The default `auto` mode uses provided triples when present and
otherwise falls back to deterministic Wikipedia-title/answer bridge triples.

Check KG cache progress:

```powershell
Get-ChildItem data/HotpotQA/kg_cache/*.jsonl, data/2WikiMultiHopQA/kg_cache/*.jsonl `
  -ErrorAction SilentlyContinue |
  ForEach-Object { "$($_.FullName): $((Get-Content $_.FullName).Count)" }
```

## Smoke Test

Run these commands after building the FamilyOWL retrieval data. They avoid LLM
answer generation and are useful before launching the full suite.

Lexical retrieval smoke test:

```powershell
python evaluation/eval_subgraph_baselines.py `
  --test-path data/FamilyOWL_1hop/test_subgraph_retrieval.jsonl `
  --baseline lexical `
  --input-format hybrid `
  --output-dir outputs/smoke_test/familyowl_1hop_lexical `
  --source-name FamilyOWL_1hop
```

Small GNN smoke test:

```powershell
python training/train_gnn_subgraph_retriever.py `
  --train-path data/FamilyOWL_1hop/train_subgraph_retrieval.jsonl `
  --dev-path data/FamilyOWL_1hop/dev_subgraph_retrieval.jsonl `
  --save-dir checkpoints/smoke_familyowl_1hop `
  --epochs 1 `
  --max-train-examples 25 `
  --max-dev-examples 25 `
  --candidate-batch-size 128 `
  --source-name FamilyOWL_1hop

python evaluation/eval_gnn_subgraph_retriever.py `
  --train-path data/FamilyOWL_1hop/train_subgraph_retrieval.jsonl `
  --dev-path data/FamilyOWL_1hop/dev_subgraph_retrieval.jsonl `
  --test-path data/FamilyOWL_1hop/test_subgraph_retrieval.jsonl `
  --checkpoint checkpoints/smoke_familyowl_1hop/best_model.pt `
  --max-test-examples 25 `
  --candidate-batch-size 128 `
  --source-name FamilyOWL_1hop `
  --save-details `
  --details-dir outputs/smoke_test/familyowl_1hop_gnn
```

## Reproduce Main Manuscript Results

After building all six retrieval datasets, run:

```powershell
python experiments/run_experiments.py `
  --datasets hotpotqa,2wiki,familyowl_1hop,familyowl_2hop,owl2bench_1hop,owl2bench_2hop `
  --methods auto `
  --top-k 3 `
  --reader-model openrouter:openai/gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512
```

To reuse existing details, predictions, metrics, checkpoints, and generated
answers where available:

```powershell
python experiments/run_experiments.py `
  --datasets hotpotqa,2wiki,familyowl_1hop,familyowl_2hop,owl2bench_1hop,owl2bench_2hop `
  --methods auto `
  --top-k 3 `
  --reader-model openrouter:openai/gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512 `
  --resume `
  --skip-llm-if-exists
```

`--methods auto` expands by dataset type:

- Text datasets: `lexical_subgraph`, `gnn_neural`, `sageqa_text_chain`,
  `gnn_rag`
- OWL datasets: `lexical_subgraph`, `gnn_neural`, `sageqa_proof`, `gnn_rag`

The full command writes:

```text
outputs/full_results/full_pipeline_results.csv
outputs/full_results/full_pipeline_results.json
outputs/full_results/<Dataset>/<method>/test_details.json
outputs/full_results/<Dataset>/<method>/metrics*.json
outputs/full_results/<Dataset>/<method>/llm_answers*.jsonl
```

If `full_pipeline_results.csv` is missing but per-method files exist, the
end-to-end runner did not finish the final aggregation step. Re-run the command
with `--resume --skip-llm-if-exists`.

## Running Subsets

Text-only reproduction:

```powershell
python experiments/run_experiments.py `
  --datasets hotpotqa,2wiki `
  --methods auto `
  --top-k 3 `
  --reader-model openrouter:openai/gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512 `
  --resume `
  --skip-llm-if-exists
```

FamilyOWL-only reproduction:

```powershell
python experiments/run_experiments.py `
  --datasets familyowl_1hop,familyowl_2hop `
  --methods auto `
  --top-k 3 `
  --reader-model openrouter:openai/gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512 `
  --resume `
  --skip-llm-if-exists
```

OWL2Bench-only reproduction:

```powershell
python experiments/run_experiments.py `
  --datasets owl2bench_1hop,owl2bench_2hop `
  --methods auto `
  --top-k 3 `
  --reader-model openrouter:openai/gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512 `
  --resume `
  --skip-llm-if-exists
```

All text, FamilyOWL, and OWL2Bench datasets:

```powershell
python experiments/run_experiments.py `
  --datasets hotpotqa,2wiki,familyowl_1hop,familyowl_2hop,owl2bench_1hop,owl2bench_2hop `
  --methods auto `
  --top-k 3 `
  --reader-model openrouter:openai/gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512 `
  --resume `
  --skip-llm-if-exists
```

GNN-RAG-only baseline:

```powershell
python experiments/run_experiments.py `
  --datasets hotpotqa,2wiki,familyowl_1hop,familyowl_2hop,owl2bench_1hop,owl2bench_2hop `
  --methods gnn_rag `
  --top-k 3 `
  --reader-model openrouter:openai/gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512 `
  --resume
```

Useful runner options:

- `--dry-run`: print commands without executing them
- `--force-train`: retrain even if a checkpoint already exists
- `--skip-llm`: skip answer generation
- `--skip-llm-if-exists`: reuse existing generated answers
- `--resume`: reuse existing details, gold subsets, predictions, and metrics
- `--resume-llm`: resume supported LLM answer generation jobs
- `--max-llm-examples N`: limit LLM answer generation for debugging
- `--gnn-rag-text-max-candidates N`: cap text candidates passed to GNN-RAG

## Metrics and Output Interpretation

The final CSV contains one row per dataset/method. The most important fields
for manuscript comparison are:

- `Answer_EM`, `Answer_F1`: final answer quality
- `Support_Prec`, `Support_Recall`, `Support_F1`: support quality
- `Joint_EM`, `Joint_F1`: combined answer/support quality where available
- `Exact_at_k`, `Contained_at_k`, `Support_Set_F1_at_k`: retrieval-side
  metrics where available
- `llm_rows`, `llm_empty`, `llm_errors`: generated or reused reader-answer
  counts

Per-method `test_details.json` files contain retrieved subgraphs/supports for
each example. Per-method `metrics*.json` files contain the exact evaluator
output used to build the final CSV.

## Error Analysis

FamilyOWL example:

```powershell
python evaluation/error_analysis_support_qa.py `
  --mode owl `
  --details outputs/full_results/FamilyOWL_2hop/gnn_sageqa_proof/test_details.json `
  --llm-answers outputs/full_results/FamilyOWL_2hop/gnn_sageqa_proof/llm_answers_top3_openrouter_openai_gpt_4_1_mini.jsonl `
  --top-k 3 `
  --output-csv outputs/full_results/FamilyOWL_2hop/gnn_sageqa_proof/error_analysis_top3.csv `
  --output-summary outputs/full_results/FamilyOWL_2hop/gnn_sageqa_proof/error_analysis_top3_summary.json
```

HotpotQA example:

```powershell
python evaluation/error_analysis_support_qa.py `
  --mode text `
  --details outputs/full_results/HotpotQA/gnn_sageqa_text_chain/test_details.json `
  --llm-answers outputs/full_results/HotpotQA/gnn_sageqa_text_chain/llm_answers_top3_openrouter_openai_gpt_4_1_mini.jsonl `
  --gold data/HotpotQA/hotpot_test_subset_gold.json `
  --predictions outputs/full_results/HotpotQA/gnn_sageqa_text_chain/hotpotqa_predictions_top3_openrouter_openai_gpt_4_1_mini.json `
  --top-k 3 `
  --output-csv outputs/full_results/HotpotQA/gnn_sageqa_text_chain/error_analysis_top3.csv `
  --output-summary outputs/full_results/HotpotQA/gnn_sageqa_text_chain/error_analysis_top3_summary.json
```

## Reproducibility Notes

- Text-QA sampling uses seed `42` in the commands above.
- The GNN scripts set Python and PyTorch seeds, but exact GPU determinism is
  not guaranteed across hardware, drivers, and dependency versions.
- LLM KG construction and LLM answer generation can vary as hosted models
  change. Cached KG and answer files give the closest run-to-run comparison.
- Full reproduction can take several hours. Runtime depends on GPU speed,
  Hugging Face cache state, LLM API latency, and whether checkpoints/outputs are
  reused.
- The runner includes auxiliary Pizza dataset keys. The manuscript command uses
  an explicit dataset list so auxiliary Pizza rows are not included by accident.
- `--skip-llm` is useful for retrieval-only checks but will not produce final
  answer metrics.

## GNN-RAG Adaptation

The GNN-RAG baseline adapts each dataset into the upstream `cmavro/GNN-RAG`
KGQA format, trains/evaluates the upstream ReaRev retriever, writes
`test.info`, calls:

```text
third_party/GNN-RAG/llm/src/qa_prediction/predict_answer.py
```

and converts upstream predictions back to the common evaluator format.

For text datasets, constructed `KG::subject::predicate::object` bridge units are
mapped into upstream KG triples and linked back to matching sentence evidence
nodes. KG triples enrich the GNN-RAG graph but are not counted as predicted
support sentences.

Local changes are documented in:

```text
patches/GNN-RAG-local-changes.md
patches/GNN-RAG-local-changes.patch
```

## License

This project is released under the MIT License. See `LICENSE` for details.
