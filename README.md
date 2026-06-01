# SAGE-QA

This repository contains the code and data-processing scripts for the SAGE-QA
experiments. It supports the manuscript experiments on HotpotQA,
2WikiMultiHopQA, and FamilyOWL, including retrieval-data construction, GNN
retriever training, symbolic retrieval variants, GNN-RAG adaptation, LLM answer
generation, final QA evaluation, and error analysis.

## Reproduction Overview

This repository is intended to reproduce the manuscript's main SAGE-QA pipeline
results for four datasets:

- HotpotQA
- 2WikiMultiHopQA
- FamilyOWL_1hop
- FamilyOWL_2hop

The main entry point is `experiments/run_experiments.py`. It builds or reuses
GNN checkpoints, evaluates retrieval methods, generates LLM answers, computes
QA/support/joint metrics, and writes the combined result tables under
`outputs/full_results/`.

The expected workflow is:

1. Install dependencies.
2. Download or extract the required input data.
3. Build the retrieval datasets.
4. Run either the smoke test or the full manuscript experiment suite.
5. Compare `outputs/full_results/full_pipeline_results.csv` with the manuscript
   tables.

The committed repository includes code, the small FamilyOWL raw archive, the
GNN-RAG adaptation code under `third_party/GNN-RAG/`, and local patch notes.
Large external benchmarks, generated processed datasets, generated outputs, and
checkpoints are intentionally not committed.

## Repository Layout

- `data_processing/`: builders for HotpotQA and 2WikiMultiHopQA retrieval data
- `data/`: raw inputs and processed retrieval datasets
- `training/`: training utilities for GNN and symbolic components
- `experiments/`: unified experiment runner
- `evaluation/`: QA and support/error-analysis scripts
- `models/`, `generation/`, `utils/`: model, answer-generation, and shared code
- `third_party/GNN-RAG/`: adapted upstream GNN-RAG baseline code
- `outputs/`: experiment outputs, predictions, and combined result files
- `checkpoints/`: trained or reused model checkpoints

## Environment

The repository was prepared and tested on Windows with PowerShell commands. The
commands can be translated to Bash on Linux/macOS by replacing PowerShell line
continuations and activation syntax.

Recommended environment:

- Python 3.12, tested with Python 3.12.7
- A virtual environment with packages from `requirements.txt`
- Network access for Hugging Face dataset/model downloads unless already cached
- A valid `OPENAI_API_KEY` for LLM answer generation
- A CUDA-capable GPU for faster GNN training, although small runs can execute on
  CPU

Full reproduction can take several hours depending on hardware, Hugging Face
cache state, OpenAI API latency, and whether checkpoints already exist. The
text-QA benchmarks use 3,000 training examples and 1,000 test examples in the
manuscript commands below. GNN-RAG and LLM answer generation are usually the
slowest stages.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The experiments use Hugging Face models/tokenizers and OpenAI-backed answer
generation. Make sure the required Hugging Face models are available locally or
online. Set `OPENAI_API_KEY` before running any command that generates LLM
answers:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

The GNN-RAG baseline uses the adapted upstream code under
`third_party/GNN-RAG`. Local changes are documented in:

- `patches/GNN-RAG-local-changes.md`
- `patches/GNN-RAG-local-changes.patch`

## Input Data

Raw input files live under `data/raw/`. The repository tracks the small
FamilyOWL archive at `data/raw/family.zip`; extract it before building the OWL
retrieval datasets:

```powershell
Expand-Archive -Path data/raw/family.zip -DestinationPath data/raw -Force
```

HotpotQA and 2WikiMultiHopQA are not committed. Download their parquet inputs
from Hugging Face into the paths expected by the text-QA builders:

```powershell
hf download hotpotqa/hotpot_qa distractor/ `
  --repo-type dataset `
  --local-dir data/raw/hotpot_qa

hf download framolfese/2WikiMultihopQA data/ `
  --repo-type dataset `
  --local-dir data/raw/2WikiMultihopQA
```

The builders read these raw benchmark files and write processed retrieval
datasets under `data/`.

Data and output policy:

- Committed: source code, `data/raw/family.zip`, and GNN-RAG adaptation files.
- Downloaded externally: HotpotQA and 2WikiMultiHopQA parquet files.
- Generated locally: `data/FamilyOWL_*`, `data/HotpotQA`,
  `data/2WikiMultiHopQA`, `outputs/`, and `checkpoints/`.
- Optional cached outputs/checkpoints may be reused with `--resume` and
  `--skip-llm-if-exists` when they are available locally.

## Build FamilyOWL Retrieval Data

FamilyOWL examples contain ontology context, SPARQL/query information, answers,
and gold explanations. The OWL builder converts each raw benchmark JSON file
into a retrieval dataset directory named after the JSON stem:

- `data/<dataset>/train_subgraph_retrieval.jsonl`
- `data/<dataset>/dev_subgraph_retrieval.jsonl`
- `data/<dataset>/test_subgraph_retrieval.jsonl`
- `data/<dataset>/*_answer_only_no_explanation.jsonl`

Build the manuscript FamilyOWL datasets:

```powershell
python data/build_subgraph_training_data.py --input-json data/raw/family/FamilyOWL_1hop.json --output-dir data

python data/build_subgraph_training_data.py --input-json data/raw/family/FamilyOWL_2hop.json --output-dir data
```

By default the builder includes gold supports, adds query-aligned OWL context
axioms, and uses bounded beam search to generate connected candidate subgraphs
with no fixed size cap. Runtime is controlled by the beam width and
candidate-pool cap, and negatives are capped at 200 per example.

Optional controls:

```powershell
python data/build_subgraph_training_data.py `
  --input-json data/raw/family/FamilyOWL_2hop.json `
  --output-dir data `
  --max-subgraph-size 0 `
  --candidate-beam-width 96 `
  --max-candidate-subgraphs 320
```

Use `--max-subgraph-size N` only when intentionally applying a fixed support
depth cap. The default `0` means no fixed cap.

The symbolic composer can also load weights fitted offline from retrieval rows:

```powershell
python training/learn_symbolic_composer_weights.py `
  --train-path data/FamilyOWL_2hop/train_subgraph_retrieval.jsonl `
  --output checkpoints/symbolic_composer_weights.json
```

Binary QAs without gold explanations are written to the
`*_answer_only_no_explanation.jsonl` files. They are used only for OWL answer
EM/F1 and are excluded from support, retrieval, and joint metrics.

## Build Text-QA Retrieval Data

HotpotQA and 2WikiMultiHopQA use a separate text-QA construction path. These
builders read downloaded benchmark parquet files and construct sentence-level
candidate subgraphs. Unlike the OWL datasets, the raw data here is natural-text
context and supporting facts rather than ontology axioms.

Text-QA rows include KG bridge nodes for GNN message passing. Both builders
accept KG-style triples from `evidences`, `evidence`, `kg_triples`, or
`triples` fields and add them as `KG::subject::predicate::object` nodes in
`graph_context_units`. These bridge nodes are available to the GNN graph but
are not counted as predicted support sentences.

For 2WikiMultiHopQA, use the official test parquet as the test source:

```powershell
python data_processing/build_2wiki_subgraph_dataset.py `
  --train-file data/raw/2WikiMultihopQA/data/train-00000-of-00002.parquet data/raw/2WikiMultihopQA/data/train-00001-of-00002.parquet `
  --dev-file data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet `
  --test-file data/raw/2WikiMultihopQA/data/test-00000-of-00001.parquet `
  --output-dir data/2WikiMultiHopQA `
  --max-train-examples 3000 `
  --max-dev-examples 500 `
  --max-test-examples 1000 `
  --max-sentences-per-example 30 `
  --max-subgraph-size 4 `
  --max-kg-bridge-triples 64 `
  --max-candidates-per-question 256 `
  --seed 42
```

For HotpotQA, the labeled validation parquet is commonly used as the evaluation
source. This command omits `--test-file`, so the test retrieval file is sampled
from `--dev-file`.

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
  --max-candidates-per-question 256 `
  --seed 42
```

The generated retrieval files are:

- `data/2WikiMultiHopQA/train_subgraph_retrieval.jsonl`
- `data/2WikiMultiHopQA/dev_subgraph_retrieval.jsonl`
- `data/2WikiMultiHopQA/test_subgraph_retrieval.jsonl`
- `data/HotpotQA/train_subgraph_retrieval.jsonl`
- `data/HotpotQA/dev_subgraph_retrieval.jsonl`
- `data/HotpotQA/test_subgraph_retrieval.jsonl`

For final runs, `--max-candidates-per-question 256` is the recommended text-QA
setting. It keeps a broad candidate pool while avoiding very large JSONL files.

## Smoke Test

Before running the full suite, verify data loading, retrieval evaluation, GNN
training, and result writing without OpenAI API calls by running a
retrieval-only FamilyOWL check.

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

These smoke tests assume the FamilyOWL retrieval data has already been built.
They do not evaluate final LLM answer quality.

## Reproduce Main Results

After setup and data construction, run the manuscript experiment suite:

```powershell
python experiments/run_experiments.py `
  --datasets hotpotqa,2wiki,familyowl_1hop,familyowl_2hop `
  --methods auto `
  --top-k 3 `
  --reader-model gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512
```

To reuse existing retrieval details, predictions, metrics, and generated LLM
answers where available:

```powershell
python experiments/run_experiments.py `
  --datasets hotpotqa,2wiki,familyowl_1hop,familyowl_2hop `
  --methods auto `
  --top-k 3 `
  --reader-model gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512 `
  --resume `
  --skip-llm-if-exists
```

The final combined result files are written to:

- `outputs/full_results/full_pipeline_results.csv`
- `outputs/full_results/full_pipeline_results.json`

The expected manuscript rows are the four datasets listed in the reproduction
overview crossed with the default methods selected by `--methods auto`. If the
CSV also contains Pizza rows, those are auxiliary experiments and are not part
of the main manuscript reproduction command shown above.

Representative values from a completed `gpt-4.1-mini`, `top-k=3` run are:

| Dataset | Method | Answer EM | Support F1 | Exact@k |
| --- | --- | ---: | ---: | ---: |
| FamilyOWL_1hop | sageqa Proof | 0.523 | 0.583 | 0.914 |
| FamilyOWL_2hop | sageqa Proof | 0.854 | 0.276 | 1.000 |

Small numerical differences can occur across hardware, dependency versions,
checkpoint initialization, and LLM responses. Reusing existing outputs with
`--resume` and `--skip-llm-if-exists` gives the closest comparison to a cached
run.

## Reproducibility Notes

- Dataset sampling uses seed `42` in the text-QA build commands.
- The GNN training scripts set fixed Python and PyTorch seeds, but exact GPU
  determinism is not guaranteed across platforms.
- LLM-based answer generation uses OpenAI models and may vary over time.
- Commands that include `--skip-llm` avoid OpenAI API calls and therefore do not
  produce final answer metrics.
- Commands that include `--skip-llm-if-exists` reuse existing generated answer
  files when present.
- The default runner includes auxiliary Pizza dataset keys if `--datasets` is
  omitted. Use the manuscript command's explicit dataset list for paper results.

## Run Selected Experiments

The unified runner is:

- `experiments/run_experiments.py`

It trains or reuses GNN checkpoints, evaluates retrieval methods, generates LLM
answers, evaluates final QA metrics, and writes combined results.

Supported manuscript dataset keys:

- `hotpotqa`: text-QA
- `2wiki`: text-QA
- `familyowl_1hop`: OWL
- `familyowl_2hop`: OWL

When `--methods auto` is used, the runner selects defaults by dataset type:

- text-QA: `lexical_subgraph`, `gnn_neural`, `sageqa_text_chain`, `gnn_rag`
- OWL: `lexical_subgraph`, `gnn_neural`, `sageqa_proof`, `gnn_rag`

Run only text-QA datasets:

```powershell
python experiments/run_experiments.py `
  --datasets hotpotqa,2wiki `
  --methods auto `
  --top-k 3 `
  --reader-model gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512 `
  --skip-llm-if-exists `
  --resume
```

Run only FamilyOWL datasets:

```powershell
python experiments/run_experiments.py `
  --datasets familyowl_1hop,familyowl_2hop `
  --methods auto `
  --top-k 3 `
  --reader-model gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512 `
  --skip-llm-if-exists `
  --resume
```

Run the adapted GNN-RAG pipeline:

```powershell
python experiments/run_experiments.py `
  --datasets hotpotqa,2wiki,familyowl_1hop,familyowl_2hop `
  --methods gnn_rag `
  --top-k 3 `
  --reader-model gpt-4.1-mini `
  --epochs 3 `
  --candidate-batch-size 512 `
  --resume
```

This baseline adapts each dataset into the upstream `cmavro/GNN-RAG` KGQA data
format, trains/evaluates the upstream ReaRev retriever, writes its `test.info`,
calls `third_party/GNN-RAG/llm/src/qa_prediction/predict_answer.py`, then
converts the upstream `predictions.jsonl` back to the common evaluator.

Useful options:

- `--dry-run`: print commands without executing them
- `--force-train`: retrain even if a checkpoint already exists
- `--skip-llm`: skip answer generation
- `--skip-llm-if-exists`: reuse existing generated answers
- `--resume`: reuse existing retrieval details, gold subsets, predictions, and metrics
- `--resume-llm`: resume OWL LLM answer generation when supported
- `--max-llm-examples N`: limit LLM answer generation for debugging
- `--datasets hotpotqa` or `--datasets familyowl_1hop`: run one dataset

The pipeline writes checkpoints under `checkpoints/`, method outputs under
`outputs/full_results/<dataset>/`, and combined result files under
`outputs/full_results/`.

## Error Analysis

The error-analysis script writes a per-example CSV and a JSON summary of error
categories.

FamilyOWL example:

```powershell
python evaluation/error_analysis_support_qa.py `
  --mode owl `
  --details outputs/full_results/FamilyOWL_2hop/gnn_sageqa_proof/test_details.json `
  --llm-answers outputs/full_results/FamilyOWL_2hop/gnn_sageqa_proof/llm_answers_top3_gpt_4_1_mini.jsonl `
  --top-k 3 `
  --output-csv outputs/full_results/FamilyOWL_2hop/gnn_sageqa_proof/error_analysis_top3.csv `
  --output-summary outputs/full_results/FamilyOWL_2hop/gnn_sageqa_proof/error_analysis_top3_summary.json
```

HotpotQA example:

```powershell
python evaluation/error_analysis_support_qa.py `
  --mode text `
  --details outputs/full_results/HotpotQA/gnn_sageqa_text_chain/test_details.json `
  --llm-answers outputs/full_results/HotpotQA/gnn_sageqa_text_chain/llm_answers_top3_gpt_4_1_mini.jsonl `
  --gold data/HotpotQA/hotpot_test_subset_gold.json `
  --predictions outputs/full_results/HotpotQA/gnn_sageqa_text_chain/hotpotqa_predictions_top3_gpt_4_1_mini.json `
  --top-k 3 `
  --output-csv outputs/full_results/HotpotQA/gnn_sageqa_text_chain/error_analysis_top3.csv `
  --output-summary outputs/full_results/HotpotQA/gnn_sageqa_text_chain/error_analysis_top3_summary.json
```

## Notes

- The reproduction commands above use only the datasets included in the final
  manuscript experiments.
- Large external benchmark files must be downloaded separately because they are
  not committed to the repository.
- Existing checkpoints and outputs can be reused with `--resume` and
  `--skip-llm-if-exists` to avoid rerunning expensive stages.
- LLM-based answer generation requires a valid `OPENAI_API_KEY`.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
