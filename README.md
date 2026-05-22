# NeSyQA

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Input Data

Raw input files live under `data/raw/`. The repository tracks only the small
OWL input archive at `data/raw/family.zip`; extract it before building the OWL
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
datasets to the other directories under `data/`.

## Build OWL Retrieval Data

OWL-style datasets are built from the project benchmark JSON files. This is a
different construction path from HotpotQA and 2WikiMultiHopQA: OWL examples
contain ontology context, SPARQL/query information, answers, and gold
explanations.

The OWL builder converts each raw benchmark JSON file into one retrieval dataset
directory named after the JSON stem:

- `data/<dataset>/train_subgraph_retrieval.jsonl`
- `data/<dataset>/dev_subgraph_retrieval.jsonl`
- `data/<dataset>/test_subgraph_retrieval.jsonl`

Build from raw benchmark files:

```powershell
python data/build_subgraph_training_data.py  --input-json data/raw/family/FamilyOWL_1hop.json --output-dir data

python data/build_subgraph_training_data.py  --input-json data/raw/family/FamilyOWL_2hop.json --output-dir data
```

By default the builder includes gold supports, adds query-aligned OWL context axioms, enumerates connected candidate subgraphs up to size 3, and caps negatives at 200 per example.

### OWL Dataset Format

New raw dataset files should follow the same structure as the existing benchmark JSON files. Each top-level item should include:

- `OWL Context`: RDF/XML context used to build candidate support axioms
- `Task Type` and `Answer Type`: metadata for reporting
- `QAs`: question-answer entries

Each QA entry should include:

- `NL Question` or `ABS Question`
- `SPARQL Query`
- `Answer`
- `Explanations` or `Minimum Explanation`

## Build Text-QA Retrieval Data

HotpotQA and 2WikiMultiHopQA use a separate text-QA construction path. These
builders read downloaded benchmark parquet files and construct sentence-level
candidate subgraphs. Unlike the OWL datasets, the raw data here is natural-text
context and supporting facts rather than ontology axioms.

Both builders accept one or more paths for `--train-file`. The generated
retrieval splits are sampled as follows:

- train from `--train-file`
- dev from `--dev-file`
- test from `--test-file` when provided
- test from `--dev-file` when `--test-file` is omitted

For 2WikiMultiHopQA, use the official test parquet as the test source:

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
  --max-candidates-per-question 512 `
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
  --max-candidates-per-question 512 `
  --seed 42
```

The builders write:

- `data/2WikiMultiHopQA/train_subgraph_retrieval.jsonl`
- `data/2WikiMultiHopQA/dev_subgraph_retrieval.jsonl`
- `data/2WikiMultiHopQA/test_subgraph_retrieval.jsonl`
- `data/HotpotQA/train_subgraph_retrieval.jsonl`
- `data/HotpotQA/dev_subgraph_retrieval.jsonl`
- `data/HotpotQA/test_subgraph_retrieval.jsonl`

## Run Experiments

There is now one experiment runner for both OWL and text-QA datasets:

- `experiments/run_experiments.py`

The dataset construction remains separate, but experiment execution is unified.
The runner trains or reuses GNN checkpoints, evaluates retrieval methods,
generates LLM answers, evaluates final QA metrics, and writes combined results.

Set `OPENAI_API_KEY` before running answer generation:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

Run all configured datasets with the default method set for each dataset type:

```powershell
python experiments/run_experiments.py `
  --datasets hotpotqa,2wiki,familyowl_1hop,familyowl_2hop `
  --methods auto `
  --top-k 3 `
  --reader-model gpt-4.1-mini `
  --encoder-model google/bert_uncased_L-2_H-128_A-2 `
  --epochs 3 `
  --candidate-batch-size 512
```

Supported dataset keys:

- `hotpotqa`: text-QA
- `2wiki`: text-QA
- `familyowl_1hop`: OWL
- `familyowl_2hop`: OWL

With `--methods auto`, the runner chooses methods by dataset type:

- text-QA: `lexical_subgraph`, `gnn_neural`, `nesyqa_text_chain`
- OWL: `lexical_subgraph`, `gnn_neural`, `nesyqa_compact`

Available methods:

- `lexical_subgraph`: lexical sentence-subgraph baseline
- `gnn_neural`: trained GNN subgraph retriever
- `nesyqa_text_chain`: GNN retriever plus text-chain symbolic adjustment for HotpotQA/2WikiMultiHopQA
- `nesyqa_compact`: GNN retriever plus compact symbolic adjustment for OWL datasets

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

Run only OWL datasets:

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

Useful options:

- `--dry-run`: print commands without executing them
- `--force-train`: retrain even if a checkpoint already exists
- `--skip-llm`: skip answer generation
- `--skip-llm-if-exists`: reuse existing generated answers
- `--resume`: reuse existing retrieval details, gold subsets, predictions, and metrics
- `--resume-llm`: resume OWL LLM answer generation when supported
- `--max-llm-examples N`: limit LLM answer generation for debugging
- `--datasets hotpotqa` or `--datasets familyowl_1hop`: run one dataset
- `--methods nesyqa_text_chain` or `--methods nesyqa_compact`: run one method

The pipeline writes checkpoints under `checkpoints/`, method outputs under
`outputs/full_results/<dataset>/`, and combined result files:

- `outputs/full_results/full_pipeline_results.csv`
- `outputs/full_results/full_pipeline_results.json`

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
