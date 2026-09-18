# SAGE-QA

SAGE-QA is a retrieval-augmented question-answering system for multi-hop reasoning over
text benchmarks and OWL-style datasets. This repository contains the published-paper
workflow and the current Thesis Chapter 7 workflow as two distinct, reproducible
protocols.

## Reproduce the published paper

The paper reports six settings: HotpotQA, 2WikiMultiHopQA, FamilyOWL 1-hop and 2-hop,
and OWL2Bench 1-hop and 2-hop. The historical pipeline combines lexical retrieval,
GraphSAGE ranking, SAGE-QA Text-Chain or Proof reasoning, and GNN-RAG through
[`experiments/run_experiments.py`](experiments/run_experiments.py).

Restore the six processed dataset roots, six GraphSAGE checkpoint roots, raw inputs,
and `outputs/full_results/` described by
[`release_manifests/paper_original/index.yaml`](release_manifests/paper_original/index.yaml).
Verify the supported historical contracts without an API call:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider `
  --basetemp .tmp/reproducibility/paper `
  tests/reproducibility/test_original_paper_contracts.py `
  tests/test_evaluate_owl_qa_predictions.py
```

The reproduction entry point for exactly the six reported settings is:

```powershell
.\.venv\Scripts\python.exe experiments/run_experiments.py `
  --datasets hotpotqa,2wiki,familyowl_1hop,familyowl_2hop,owl2bench_1hop,owl2bench_2hop `
  --methods auto --top-k 3
```

This scientific run may train or load GPU models and may call a hosted reader. Commit
`dbdbb50708bdc6c686ef82518ec71c1d1bf55985` is the strongest paper-source candidate,
but the exact published-result/source binding remains unproven. See
[`REPRODUCE.md`](REPRODUCE.md) and
[`docs/ORIGINAL_PAPER_SOURCE_REVISIONS.md`](docs/ORIGINAL_PAPER_SOURCE_REVISIONS.md).

## Reproduce Thesis Chapter 7

Chapter 7 reports ten settings: the six above plus Pizza 100 and Pizza 250 at 1-hop and
2-hop. Its main pipeline is:

```text
Generator D -> DistilBERT cross-encoder -> Text-Chain / Proof
            -> adaptive support aggregation -> support-grounded reader
```

The retained release also covers lexical, clean GNN-RAG, and full-context baselines;
the hard-pair GraphSAGE ablation; the complete ground-truth support reference condition;
and stage-wise error analysis. Restore the paths indexed by `thesis_final`,
`thesis_baselines`, `thesis_graph_ablation`, and `oracle`, then run:

```powershell
.\.venv\Scripts\python.exe evaluation/check_thesis_final_paths.py
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider `
  --basetemp .tmp/reproducibility/thesis-fast `
  tests/reproducibility --ignore=tests/reproducibility/test_original_paper_contracts.py
```

Exact stage commands and GPU/API boundaries are in [`REPRODUCE.md`](REPRODUCE.md).
Frozen retrieval, predictions, metrics, and manifests can be verified without calling an
external API. The indexed `884480be53c554edae70d3b2d8e781847590aa69` revision is the
scientific thesis-source baseline; later documentation-only commits do not redefine it.

## Results

[`RESULTS.md`](RESULTS.md) maps the published-paper results and each reported Chapter 7
result family to its frozen artifact root and release manifest.

## Artifact download

**REVIEWER_REPRODUCTION_BLOCKER:** no verified public download URL is currently bound to
the required paper or Chapter 7 artifact bundles. A fresh clone therefore cannot perform
complete artifact verification or scientific reproduction.

The required publication units are `paper_original_artifacts` and
`thesis_ch7_artifacts`, with checksums and extraction paths matching
[`docs/ARTIFACTS.md`](docs/ARTIFACTS.md). No URL is claimed until it has been published
and verified.

## Citation / License

Citation metadata is in [`CITATION.cff`](CITATION.cff). First-party code is released
under the [`MIT License`](LICENSE). The adapted GNN-RAG baseline is third-party code;
consult its upstream terms and
[`patches/GNN-RAG-local-changes.md`](patches/GNN-RAG-local-changes.md) before use or
redistribution.
