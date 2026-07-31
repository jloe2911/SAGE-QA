# GNN Retrieval Development Protocol

This protocol makes FamilyOWL and 2Wiki GNN iterations fast, reproducible, and
comparable. It samples questions, never candidate rows: every selected question
keeps its complete candidate pool, so ranking metrics retain their meaning.

## Fixed development selections

The initial selections use seed 42 and contain 64 train, 32 dev, and 32 test
questions per dataset. Exact raw group/question or record IDs, source sizes,
and stratum counts are stored in each `sample_manifest.json`.

Create or refresh the materialized files only when the upstream candidate
dataset changes:

```powershell
python data_processing/create_gnn_dev_sample.py `
  --input-dir data/FamilyOWL_1hop `
  --output-dir data/development/gnn_samples/familyowl_1hop `
  --train-examples 64 --dev-examples 32 --test-examples 32 --seed 42

python data_processing/create_gnn_dev_sample.py `
  --input-dir data/2WikiMultiHopQA `
  --output-dir data/development/gnn_samples/2wiki `
  --train-examples 64 --dev-examples 32 --test-examples 32 --seed 42
```

These files bootstrap the selection manifest. They are not the data used for
training after candidate-construction code changes. Rebuild the selected raw
questions with the current builders:

```powershell
python data/build_subgraph_training_data.py `
  --input-json data/raw/family/FamilyOWL_1hop.json `
  --output-dir data/development/gnn_rebuilt `
  --selection-manifest data/development/gnn_samples/familyowl_1hop/sample_manifest.json `
  --max-subgraph-size 7 `
  --max-context-units 40 `
  --max-negative-per-example 200 `
  --candidate-beam-width 96 `
  --max-candidate-subgraphs 320

python data_processing/build_2wiki_subgraph_dataset.py `
  --train-file data/raw/2WikiMultihopQA/data/train-00000-of-00002.parquet data/raw/2WikiMultihopQA/data/train-00001-of-00002.parquet `
  --dev-file data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet `
  --output-dir data/development/gnn_rebuilt/2WikiMultiHopQA `
  --selection-manifest data/development/gnn_samples/2wiki/sample_manifest.json `
  --max-kg-candidate-triples 30 `
  --max-subgraph-size 4 `
  --max-kg-bridge-triples 64 `
  --kg-construction-backend llm `
  --kg-construction-model openai:gpt-4.1-mini `
  --kg-construction-workers 3 `
  --candidate-beam-width 96 `
  --max-candidates-per-question 256 `
  --kg-cache-dir data/2WikiMultiHopQA_shared_v3/kg_cache `
  --seed 42
```

The 2Wiki source currently uses the labeled validation data for both its dev
and test source. The selection strips builder-added split markers when checking
leakage and excludes any dev source record from the test slice. LLM KG
construction is cached, so rebuilding candidates does not repeat completed API
calls.

## Train and evaluate

Use a unique experiment name so checkpoints and detailed rankings remain
comparable:

```powershell
$sample = "data/development/gnn_rebuilt/FamilyOWL_1hop"
$run = "checkpoints/development/gnn_rebuilt/family_v3_leaky"

python training/train_gnn_subgraph_retriever.py `
  --train-path "$sample/train_subgraph_retrieval.jsonl" `
  --dev-path "$sample/dev_subgraph_retrieval.jsonl" `
  --save-dir $run `
  --epochs 5 `
  --candidate-batch-size 256 `
  --architecture-version 3 `
  --freeze-encoder

python evaluation/eval_gnn_subgraph_retriever.py `
  --train-path "$sample/train_subgraph_retrieval.jsonl" `
  --dev-path "$sample/dev_subgraph_retrieval.jsonl" `
  --test-path "$sample/test_subgraph_retrieval.jsonl" `
  --checkpoint "$run/best_model.pt" `
  --candidate-batch-size 256 `
  --save-details `
  --details-dir "outputs/development_runs/gnn_samples/family_baseline"
```

For 2Wiki, use `data/development/gnn_rebuilt/2WikiMultiHopQA`. Do not use the
runtime `--max-*-examples` flags with these files: they are already fixed
selections, and another sample layer makes comparisons harder to audit.

Training excludes questions with no rankable positive/negative pair. They
remain in dev/test evaluation because missing upstream evidence is a real
end-to-end failure.

Both datasets use one coefficient-free graded listwise objective. Candidate
target probability is its gold set-F1 divided by the sum of candidate set-F1
values for that question. Exact proofs therefore have the strongest individual
target, useful partial proofs retain signal, and irrelevant candidates receive
none. This makes the training rule identical across ontology-native and
text-derived graphs while respecting missing upstream evidence. Checkpoints
are selected on dev `set_f1@3`, then `precision@3` and exact MRR as
tie-breakers.

The 2Wiki extractor uses page-aware sentence selection and versioned cache
signatures. After an extraction change, incompatible cache entries are rebuilt
instead of silently reused. Rows expose `matched_gold_kg_units`,
`missing_gold_kg_units`, and selected/all context counts for diagnostics.

For an interactive rebuild, inspection, training, and evaluation workflow,
open `notebooks/gnn_development_workbench.ipynb`. Its control cells call the
maintained command-line builders and trainer; the notebook does not duplicate
their implementation.

## Decision rule

Tune on train/dev only. The primary checkpoint target is `set_f1@3`, which
balances precision and recall for the evidence union actually passed
downstream. Use `precision@3` and exact MRR as tie-breakers and continue to
report `recall@3` so a smaller evidence set is not mistaken for a better
retriever.
Also report:

- `best_precision@1`, `best_recall@1`, and `best_set_f1@1` for ranking quality;
- `exact_hit@1` for minimal-proof selection;
- `precision@3`, `recall@3`, and `set_f1@3` for downstream evidence quality.

Run the fixed test split only after a change improves dev precision without a
material recall regression. Because 32 test questions are deliberately small,
use it as an iteration gate, not as the final manuscript estimate. Confirm
promising changes on the full untouched evaluation set.

## Current diagnosis

Rebuilding and training the fixed samples exposed two earlier bottlenecks:

1. Architecture v2's classifier ReLU died and assigned every candidate the
   same score. Architecture v3 uses LeakyReLU and restores candidate-dependent
   ranking.
2. Current 2Wiki LLM KG extraction has only 0.272/0.255/0.266 mean gold-triple
   coverage on train/dev/test. Only 4/64 training questions have complete KG
   extraction, and only 31/64 have rankable supervision.

Fresh architecture-v3 test baselines are:

| Dataset | Best precision@1 | Union precision@3 | Union recall@3 |
| --- | ---: | ---: | ---: |
| FamilyOWL 1-hop | 0.266 | 0.233 | 0.885 |
| 2Wiki | 0.354 | 0.282 | 0.227 |

For FamilyOWL, ranking remains the main problem because exact candidates are
usually available. For 2Wiki, improve KG coverage before treating GNN tuning as
the primary lever: the retriever cannot recover facts absent from its candidate
graph.
