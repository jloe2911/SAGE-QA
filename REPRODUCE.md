# Reproducing and verifying SAGE-QA

This is the canonical reproduction guide. Verification reuses frozen artifacts and does
not call an external service. Scientific reproduction may require separately distributed
assets, a GPU, or provider credentials; those dependencies are stated at each boundary.

## Environment and dependency scopes

The recorded development environment uses Python 3.12.7. Dependency versions are
documented rather than modernized in this release-cleanup phase.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

| Use | Dependency source | Additional requirement |
|---|---|---|
| Verification only | `requirements-dev.txt` | Restored frozen assets for checks that inspect them |
| Final-thesis retrieval | `requirements.txt` | Pinned DistilBERT snapshot and frozen checkpoints; GPU for practical full inference |
| GPU experiments/training | `requirements.txt` | Compatible CUDA/PyTorch environment; exact results can vary by hardware |
| Answer generation | `requirements.txt` | Provider credentials and availability; not needed to verify cached answers |
| GNN-RAG baseline | `requirements.txt` plus `third_party/GNN-RAG/` environment/configuration | Upstream/baseline-specific assets and typically a GPU |
| Development/testing | `requirements-dev.txt` | Jupyter only for notebook work |

`requirements-dev.txt` includes `requirements.txt`. Neither file is an environment lock,
and hosted model deployments are external mutable dependencies.

## A. Verify thesis-final frozen results

This is the safest workflow and requires no API calls.

### 1. Restore required assets

Restore the separately distributed roots listed in [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md)
at their manifest-recorded paths. Verify their identity against the machine-readable
indexes under [`release_manifests/`](release_manifests/). The repository currently records
their external publication location as pending or unknown; a fresh clone alone is not a
complete artifact checkout.

### 2. Run the path preflight

```powershell
python evaluation/check_thesis_final_paths.py
```

This resolves and reports active paths without reading `.env`, invoking a model, calling
an API, or modifying an artifact.

### 3. Run FAST

```powershell
$base = New-Item -ItemType Directory -Force -Path .tmp/reproducibility
$bt = Join-Path $base.FullName ("sageqa-repro-fast-" + [guid]::NewGuid())
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp $bt `
  tests/reproducibility --ignore=tests/reproducibility/test_original_paper_contracts.py
```

FAST checks logical indexes, protected roots, selected artifact identities, thesis-final
routing, the gold firewall, and the separation of historical Gold Support from the
complete Gold Support oracle. The multi-gigabyte Generator D members are not fully hashed.

### 4. Optionally run STANDARD or FULL

Use the exact commands in
[`docs/REPRODUCIBILITY_TEST_BASELINE.md`](docs/REPRODUCIBILITY_TEST_BASELINE.md).
STANDARD adds supported final-thesis, baseline, and graph-ablation tests. FULL also hashes
all 66 Generator D members and runs supported original-paper and development contracts;
it is local and API-free but reads about 16.9 GB for the full artifact hash pass.

## B. Reproduce the final thesis pipeline

The frozen thesis protocol is:

```text
Generator D -> DistilBERT cross-encoder -> Text-Chain/Proof
            -> adaptive support aggregation -> support-grounded reader
```

Reproduction is intentionally separated by stage. Consult the frozen configurations and
embedded manifests before running any command; release indexes authorize verification,
not a new TEST run.

| Stage | Entry point or record | Runtime/dependency boundary |
|---|---|---|
| Generator D candidate construction | dataset builders in `data_processing/` and `data/build_subgraph_training_data.py`; frozen corpus at `data/production_generator_d_v1/` | Deterministic/local once raw inputs are present; large output; some historical text-building modes can be provider-dependent |
| Cross-encoder training/DEV evaluation | `experiments/cross_encoder_reranking_dev_v1/run_experiment.py` and its `PROTOCOL.md` | GPU-dependent; pinned DistilBERT snapshot required; DEV protocol only |
| Frozen TEST retrieval | `evaluation/run_cross_encoder_test_retrieval.py` and `evaluation/run_production_test_retrieval.py` | GPU-dependent; frozen checkpoint and policies required; do not run as routine verification |
| Retrieval export/evaluation | `evaluation/export_manuscript_retrieval_results.py` | Deterministic/local over frozen rankings; gold is joined only after ranking freeze |
| Reader-input preflight | `python generation/run_final_manuscript_answer_generation.py preflight` | Local; validates and freezes inputs; no API call |
| Answer generation | `python generation/run_final_manuscript_answer_generation.py generate` | Expensive and provider-dependent; requires credentials; provider drift can prevent byte-identical regeneration |
| Frozen answer evaluation | `python generation/run_final_manuscript_answer_generation.py evaluate --source-root .` | Local over complete frozen predictions; no API call |
| Canonical finalization | `python evaluation/finalize_final_manuscript_end_to_end.py` | Deterministic/local; verifies counts, hashes, denominators, and equal-dataset macro |

For exact thesis reporting, prefer the frozen, hashed retrieval and reader artifacts over a
new hosted-reader call. Baseline and graph-ablation workflows are separate logical bundles
and must not be substituted for the main method.

## C. Reproduce the published/original-paper protocol

The historical entry point is:

```powershell
python experiments/run_experiments.py --help
```

The full historical command and its dataset-specific options remain available in that
revision's README and runner. Do not substitute current thesis-final builders, checkpoint
selection, cross-encoder, adaptive policy, or reader orchestration.

The strongest source candidate is
`dbdbb50708bdc6c686ef82518ec71c1d1bf55985`, but its status is
`candidate_not_fully_proven`. Exact original reproduction cannot currently be guaranteed:
the six reported original processed datasets, six GraphSAGE checkpoint roots, the retained reported children and aggregate files under `outputs/full_results/`,
exact environment, and hosted-reader lineage are not bound by a complete commit-specific
manifest. Details are in
[`docs/ORIGINAL_PAPER_SOURCE_REVISIONS.md`](docs/ORIGINAL_PAPER_SOURCE_REVISIONS.md) and
[`release_manifests/paper_original/index.yaml`](release_manifests/paper_original/index.yaml).

## Safety boundary

Verification commands are read-only contract checks. Candidate generation, training,
retrieval inference, and answer generation are scientific workflows: they can be costly,
write new outputs, and may require external services. Do not run them merely to validate a
clone, and do not overwrite any frozen artifact root.
