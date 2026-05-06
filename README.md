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

## Project Structure

```text
NeSyQA/
|- data/
|  |- build_subgraph_training_data.py
|  |- FamilyOWL_1hop/
|  |  |- train_subgraph_retrieval.jsonl
|  |  |- dev_subgraph_retrieval.jsonl
|  |  \- test_subgraph_retrieval.jsonl
|  \- FamilyOWL_2hop/
|     |- train_subgraph_retrieval.jsonl
|     |- dev_subgraph_retrieval.jsonl
|     \- test_subgraph_retrieval.jsonl
|- evaluation/
|  |- eval_subgraph_baselines.py
|  |- eval_axiom_retriever_baseline.py
|  |- eval_gnn_subgraph_retriever.py
|  \- collect_final_results.py
|- experiments/
|  \- run_final_pipeline.py
|- models/
|  |- gnn_subgraph_retriever.py
|  \- symbolic_composer.py
|- training/
|  \- train_gnn_subgraph_retriever.py
|- utils/
|  |- eval_splits.py
|  |- model_loader.py
|  \- tokenizer.py
|- requirements.txt
|- LICENSE
\- README.md
```

## Build Final Data

The final pipeline consumes:

- `data/FamilyOWL_1hop/train_subgraph_retrieval.jsonl`
- `data/FamilyOWL_1hop/dev_subgraph_retrieval.jsonl`
- `data/FamilyOWL_1hop/test_subgraph_retrieval.jsonl`
- `data/FamilyOWL_2hop/train_subgraph_retrieval.jsonl`
- `data/FamilyOWL_2hop/dev_subgraph_retrieval.jsonl`
- `data/FamilyOWL_2hop/test_subgraph_retrieval.jsonl`

Create them from the raw benchmark files:

```bash
python data/build_subgraph_training_data.py `
  --input-json FamilyOWL_1hop.json FamilyOWL_2hop.json `
  --output-dir data
```

For a quick smoke test without rebuilding everything:

```bash
python data/build_subgraph_training_data.py `
  --input-json FamilyOWL_1hop.json `
  --output-dir data/_smoke `
  --max-groups 2 `
  --max-context-units 10 `
  --max-negative-per-example 5
```

By default the builder includes gold supports, adds query-aligned OWL context axioms, enumerates connected candidate subgraphs up to size 3, and caps negatives at 200 per example.

## Run Final Experiments

```bash
python experiments/run_final_pipeline.py `
  --data-root data `
  --results-root outputs/final_results `
  --checkpoints-root checkpoints `
  --epochs 3 `
  --freeze-encoder `
  --candidate-batch-size 512 `
  --input-format hybrid
```

The pipeline runs both `FamilyOWL_1hop` and `FamilyOWL_2hop` from separate split files under `data/`.

## Outputs

The final pipeline writes:

- model checkpoints under `checkpoints/gnn_subgraph_ranker_1hop` and `checkpoints/gnn_subgraph_ranker_2hop`
- per-dataset metrics under `outputs/final_results/FamilyOWL_1hop` and `outputs/final_results/FamilyOWL_2hop`
- combined tables at `outputs/final_results/final_summary.csv` and `outputs/final_results/final_summary.tex`

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
