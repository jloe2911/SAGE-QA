# SAGE-QA

SAGE-QA is a retrieval-augmented question answering pipeline for multi-hop
reasoning over text benchmarks and OWL-style datasets. The
repository contains the code used for the manuscript experiments on HotpotQA,
2WikiMultiHopQA, Family, Pizza, and OWL2Bench.

The pipeline converts each question into candidate evidence subgraphs, trains a
GNN subgraph retriever, compares lexical/GNN/SAGE-QA/GNN-RAG retrieval
variants, generates final answers with an OpenAI-compatible LLM reader, and
evaluates answer, support, retrieval, and joint metrics.

## Repository Overview

```text
data/                       Raw archives and generated retrieval datasets
data_processing/            Text benchmark builders and GNN-RAG adapters
training/                   GNN subgraph retriever training
models/                     GNN and symbolic retriever components
generation/                 LLM answer-generation scripts
evaluation/                 Retrieval, QA, support, and error evaluators
experiments/                End-to-end experiment runner
third_party/GNN-RAG/        Adapted GNN-RAG baseline
patches/                    Notes and patch for local GNN-RAG changes
checkpoints/                Generated model checkpoints
outputs/                    Generated experiment outputs
```

Key entry points:

- Main runner: `experiments/run_experiments.py`
- GNN training: `training/train_gnn_subgraph_retriever.py`
- Text builders:
  `data_processing/build_hotpot_subgraph_dataset.py` and
  `data_processing/build_2wiki_subgraph_dataset.py`
- Ontology builder: `data/build_subgraph_training_data.py`
- Final result aggregation output:
  `outputs/full_results/full_pipeline_results.csv`

## Requirements

This repository was prepared with Python 3.12 on Windows/PowerShell.

Recommended environment:

- Python 3.12, tested with Python 3.12.7
- CUDA-capable GPU for full GNN training; CPU is sufficient for smoke tests
- Network access for Hugging Face dataset downloads unless files are already
  cached locally
- `OPENAI_API_KEY` or `OPENROUTER_API_KEY` for LLM KG construction and final
  answer generation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Some fresh environments also need the Hugging Face CLI and Parquet backend:

```powershell
pip install "huggingface_hub[cli]" pyarrow
```

Set one API key for LLM calls:

```powershell
$env:OPENAI_API_KEY="sk-..."
# or
$env:OPENROUTER_API_KEY="sk-or-..."
```

Provider-prefixed model names are accepted, for example
`openai:gpt-4.1-mini`. Without a prefix, OpenRouter is used when
`OPENROUTER_API_KEY` is set; otherwise OpenAI is used.

## Data Policy

Expected committed assets:

- Source code
- `data/raw/family.zip`
- `data/raw/pizza_100.zip`
- `data/raw/pizza_250.zip`
- `data/raw/owl2bench.zip`
- Adapted GNN-RAG files under `third_party/GNN-RAG/`
- Local GNN-RAG change notes under `patches/`

Downloaded externally:

- HotpotQA Parquet files
- 2WikiMultiHopQA Parquet files

Generated locally:

- `data/FamilyOWL_1hop/`
- `data/FamilyOWL_2hop/`
- `data/pizza_100_1hop/`
- `data/pizza_100_2hop/`
- `data/pizza_250_1hop/`
- `data/pizza_250_2hop/`
- `data/OWL2Bench_1hop/`
- `data/OWL2Bench_2hop/`
- `data/HotpotQA/`
- `data/2WikiMultiHopQA/`
- `checkpoints/`
- `outputs/`

## Download Raw Data

Extract the bundled ontology benchmark archives:

```powershell
Expand-Archive -Path data/raw/family.zip -DestinationPath data/raw -Force
Expand-Archive -Path data/raw/pizza_100.zip -DestinationPath data/raw -Force
Expand-Archive -Path data/raw/pizza_250.zip -DestinationPath data/raw -Force
Expand-Archive -Path data/raw/owl2bench.zip -DestinationPath data/raw -Force
```

Download the text benchmarks:

```powershell
hf download hotpotqa/hotpot_qa distractor/ `
  --repo-type dataset `
  --local-dir data/raw/hotpot_qa

hf download framolfese/2WikiMultihopQA data/ `
  --repo-type dataset `
  --local-dir data/raw/2WikiMultihopQA
```

For 2WikiMultiHopQA, the commands below use the labeled validation Parquet file
as the evaluation source because the public test Parquet file in this
distribution does not include usable gold answers/supporting facts for
supervised evaluation.

## Build Retrieval Datasets

Build Family:

```powershell
python data/build_subgraph_training_data.py `
  --input-json data/raw/family/FamilyOWL_1hop.json `
  --output-dir data

python data/build_subgraph_training_data.py `
  --input-json data/raw/family/FamilyOWL_2hop.json `
  --output-dir data
```

Build Pizza:

```powershell
python data/build_subgraph_training_data.py `
  --input-json data/raw/pizza_100/pizza_100_1hop.json `
  --output-dir data

python data/build_subgraph_training_data.py `
  --input-json data/raw/pizza_100/pizza_100_2hop.json `
  --output-dir data

python data/build_subgraph_training_data.py `
  --input-json data/raw/pizza_250/pizza_250_1hop.json `
  --output-dir data

python data/build_subgraph_training_data.py `
  --input-json data/raw/pizza_250/pizza_250_2hop.json `
  --output-dir data
```

Build OWL2Bench:

```powershell
python data/build_subgraph_training_data.py `
  --input-json data/raw/owl2bench/OWL2Bench_1hop.json `
  --output-dir data

python data/build_subgraph_training_data.py `
  --input-json data/raw/owl2bench/OWL2Bench_2hop.json `
  --output-dir data
```

Build HotpotQA:

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
  --kg-construction-model openai:gpt-4.1-mini `
  --kg-construction-workers 3 `
  --kg-request-timeout 90 `
  --kg-max-retries 6 `
  --kg-retry-initial-sleep 10 `
  --max-candidates-per-question 256 `
  --seed 42
```

Build 2WikiMultiHopQA:

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
  --kg-construction-model openai:gpt-4.1-mini `
  --kg-construction-workers 3 `
  --kg-request-timeout 90 `
  --kg-max-retries 6 `
  --kg-retry-initial-sleep 10 `
  --max-candidates-per-question 256 `
  --seed 42
```

The builders produce `{train,dev,test}_subgraph_retrieval.jsonl` files. The OWL
builder also writes `{train,dev,test}_answer_only_no_explanation.jsonl`, which
is used only for OWL answer EM/F1 and is excluded from support, retrieval, and
joint metrics.

LLM KG construction caches successful calls under `<output-dir>/kg_cache/`, so
interrupted text builds can be resumed without repeating completed examples.
For quick checks without API calls, use `--kg-construction-backend deterministic`
and small `--max-*-examples` values.

## Reproduce Main Results

After building all six retrieval datasets, run the main manuscript command:

```powershell
python experiments/run_experiments.py `
  --datasets hotpotqa,2wiki,familyowl_1hop,familyowl_2hop,pizza_100_1hop,pizza_100_2hop,pizza_250_1hop,pizza_250_2hop,owl2bench_1hop,owl2bench_2hop `
  --methods auto `
  --top-k 3 `
  --reader-model openai:gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512
```

To reuse existing checkpoints, details, predictions, metrics, and generated
answers:

```powershell
python experiments/run_experiments.py `
  --datasets hotpotqa,2wiki,familyowl_1hop,familyowl_2hop,pizza_100_1hop,pizza_100_2hop,pizza_250_1hop,pizza_250_2hop,owl2bench_1hop,owl2bench_2hop `
  --methods auto `
  --top-k 3 `
  --reader-model openai:gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512 `
  --resume `
  --skip-llm-if-exists
```

`--methods auto` expands by dataset type:

- Text datasets: `lexical_subgraph`, `gnn_neural`,
  `sageqa_text_chain`, `gnn_rag`
- OWL datasets: `lexical_subgraph`, `gnn_neural`,
  `sageqa_proof`, `gnn_rag`

The main run writes:

```text
outputs/full_results/full_pipeline_results.csv
outputs/full_results/full_pipeline_results.json
outputs/full_results/<Dataset>/<method>/test_details.json
outputs/full_results/<Dataset>/<method>/metrics*.json
outputs/full_results/<Dataset>/<method>/llm_answers*.jsonl
```

If `full_pipeline_results.csv` is missing but per-method files exist, the run
likely stopped before final aggregation. Re-run the command with
`--resume --skip-llm-if-exists`.

## Useful Reproduction Subsets

Text-only:

```powershell
python experiments/run_experiments.py `
  --datasets hotpotqa,2wiki `
  --methods auto `
  --top-k 3 `
  --reader-model openai:gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512 `
  --resume `
  --skip-llm-if-exists
```

Ontology-only:

```powershell
python experiments/run_experiments.py `
  --datasets familyowl_1hop,familyowl_2hop,pizza_100_1hop,pizza_100_2hop,pizza_250_1hop,pizza_250_2hop,owl2bench_1hop,owl2bench_2hop `
  --methods auto `
  --top-k 3 `
  --reader-model openai:gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512 `
  --resume `
  --skip-llm-if-exists
```

GNN-RAG baseline only:

```powershell
python experiments/run_experiments.py `
  --datasets hotpotqa,2wiki,familyowl_1hop,familyowl_2hop,pizza_100_1hop,pizza_100_2hop,pizza_250_1hop,pizza_250_2hop,owl2bench_1hop,owl2bench_2hop `
  --methods gnn_rag `
  --top-k 3 `
  --reader-model openai:gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512 `
  --resume
```

Frequently used runner flags:

- `--dry-run`: print commands without executing them
- `--force-train`: retrain even if a checkpoint exists
- `--skip-llm`: skip answer generation
- `--skip-llm-if-exists`: reuse existing answer files
- `--resume`: reuse existing details, gold subsets, predictions, and metrics
- `--resume-llm`: resume supported LLM generation jobs
- `--max-llm-examples N`: limit LLM calls for debugging
- `--reader-top-k N`: pass a different number of retrieved candidates to the
  reader than the evaluation `--top-k`
- `--support-top-k N`: union a different number of candidates for text support
  export
- `--gnn-rag-text-max-candidates N`: cap text candidates passed to GNN-RAG

## Metrics

The final CSV has one row per dataset/method. The main columns are:

- `Answer_EM`, `Answer_F1`: final answer quality
- `Support_Prec`, `Support_Recall`, `Support_F1`: support quality
- `Joint_EM`, `Joint_F1`: combined answer/support metrics where available
- `Retrieval_Precision_at_k`, `Retrieval_Recall_at_k`,
  `Retrieval_F1_at_k`: retrieval-side evidence overlap
- `Exact_Evidence_Set_at_k`: whether the top-k evidence union exactly matches a
  gold evidence set
- `Complete_Evidence_Recall_at_k`: whether the top-k evidence union contains a
  complete gold evidence set
- `llm_rows`, `llm_empty`, `llm_errors`: generated or reused reader-answer
  counts

Per-method `test_details.json` files contain ranked retrieved subgraphs and
supports for each example. Per-method `metrics*.json` files contain the exact
evaluator output used to build the final CSV.

## Text Benchmark Representation

HotpotQA and 2WikiMultiHopQA are evaluated as text-support retrieval tasks. Raw
benchmark context is flattened into sentence evidence units:

```text
SENT::<title>::<sentence_id>::<sentence text>
```

Constructed or provided KG triples are auxiliary graph context:

```text
KG::<subject>::<predicate>::<object>
```

Only `SENT::...` units are exported as predicted support and compared against
official gold supporting facts. `KG::...` units can enrich the retriever/GNN
graph but are not counted as support facts.

Generated text candidate rows include:

- `example_id`: shared id for all candidate rows derived from one QA example
- `question`, `answer`: raw question and gold answer
- `subgraph_units`: candidate sentence support set scored by the retriever
- `graph_context_units`: optional KG context nodes
- `raw_supporting_facts`: original benchmark support labels
- `gold_support_units`: gold support labels resolved to `SENT::...` units
- `label`: positive when the candidate contains a complete gold support set
- `rank_target`: soft support-overlap target
- `symbolic_features`: gold-free lexical/structural features

More details and worked examples are in `TEXT_BENCHMARK_RETRIEVAL.md`.

## GNN-RAG Adaptation

The GNN-RAG baseline adapts each dataset into the upstream
`cmavro/GNN-RAG` KGQA format, trains/evaluates the upstream ReaRev retriever,
writes `test.info`, calls:

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

## Error Analysis

Family example:

```powershell
python evaluation/error_analysis_support_qa.py `
  --mode owl `
  --details outputs/full_results/FamilyOWL_2hop/gnn_sageqa_proof/test_details.json `
  --llm-answers outputs/full_results/FamilyOWL_2hop/gnn_sageqa_proof/llm_answers_top3_openai_gpt_4_1_mini.jsonl `
  --top-k 3 `
  --output-csv outputs/full_results/FamilyOWL_2hop/gnn_sageqa_proof/error_analysis_top3.csv `
  --output-summary outputs/full_results/FamilyOWL_2hop/gnn_sageqa_proof/error_analysis_top3_summary.json
```

HotpotQA example:

```powershell
python evaluation/error_analysis_support_qa.py `
  --mode text `
  --details outputs/full_results/HotpotQA/gnn_sageqa_text_chain/test_details.json `
  --llm-answers outputs/full_results/HotpotQA/gnn_sageqa_text_chain/llm_answers_top3_openai_gpt_4_1_mini.jsonl `
  --gold data/HotpotQA/hotpot_test_subset_gold.json `
  --predictions outputs/full_results/HotpotQA/gnn_sageqa_text_chain/hotpotqa_predictions_top3_openai_gpt_4_1_mini.json `
  --top-k 3 `
  --output-csv outputs/full_results/HotpotQA/gnn_sageqa_text_chain/error_analysis_top3.csv `
  --output-summary outputs/full_results/HotpotQA/gnn_sageqa_text_chain/error_analysis_top3_summary.json
```

Adjust the answer/prediction filenames if the reader model tag differs.

## Reproducibility Notes

- Text-QA sampling uses seed `42` in the commands above.
- The GNN scripts set Python and PyTorch seeds, but exact GPU determinism is not
  guaranteed across hardware, drivers, and dependency versions.
- Hosted LLM outputs can vary as provider-side models change. Reusing cached KG
  and answer files gives the closest run-to-run comparison.
- Full reproduction can take several hours. Runtime depends on GPU speed,
  Hugging Face cache state, LLM API latency, and whether checkpoints/outputs
  are reused.
- `--skip-llm` is useful for retrieval-only checks but does not produce final
  answer metrics.

## License

This project is released under the MIT License. See `LICENSE` for details.
