# Reproducing SAGE-QA

The published-paper and Thesis Chapter 7 workflows are separate protocols. Do not route
one through the other or overwrite frozen artifact roots.

## A. Published paper

### Environment setup

The recorded repository environment uses Python 3.12.7. The requirement files are not a
complete historical lock.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

### Required external assets and paths

Restore `paper_original_artifacts` to the exact paths in
`release_manifests/paper_original/index.yaml`: six processed dataset roots under `data/`,
six `checkpoints/gnn_subgraph_ranker_*_full/` roots, required raw inputs, and the six
reported result children plus aggregates under `outputs/full_results/`.

No verified public download location is currently bound to this bundle. This is a
**REVIEWER_REPRODUCTION_BLOCKER**.

### Verify frozen results without API calls

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider `
  --basetemp .tmp/reproducibility/paper `
  tests/reproducibility/test_original_paper_contracts.py `
  tests/test_evaluate_owl_qa_predictions.py
```

This verifies source objects, retained paths, and evaluator behavior. It does not call a
hosted reader or regenerate results.

### Scientific reproduction

```powershell
.\.venv\Scripts\python.exe experiments/run_experiments.py `
  --datasets hotpotqa,2wiki,familyowl_1hop,familyowl_2hop,owl2bench_1hop,owl2bench_2hop `
  --methods auto --top-k 3
```

GraphSAGE training/inference and GNN-RAG are GPU-dependent for practical full runs. The
answer stage uses a hosted reader and requires provider credentials and availability;
provider drift may prevent byte-identical regeneration. Use restored frozen outputs for
API-free verification. `dbdbb507` remains the strongest source candidate, not a formally
proven exact publication revision.

## B. Thesis Chapter 7

### Environment setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

GPU reproduction also requires a compatible CUDA/PyTorch environment. The final
cross-encoder requires the pinned DistilBERT snapshot, supplied locally or through
`SAGEQA_DISTILBERT_PATH`.

### Required external assets and paths

Restore `thesis_ch7_artifacts` at the exact paths in the `thesis_final`,
`thesis_baselines`, `thesis_graph_ablation`, and `oracle` release indexes. Required roots
include `data/production_generator_d_v1/`, final cross-encoder and GraphSAGE/GNN-RAG
checkpoints, adaptive policies, and frozen outputs under `outputs/final_results/`.

No verified public download location is currently bound to this bundle. This is a
**REVIEWER_REPRODUCTION_BLOCKER**.

### Verify frozen results without API calls

```powershell
.\.venv\Scripts\python.exe evaluation/check_thesis_final_paths.py
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider `
  --basetemp .tmp/reproducibility/thesis-fast `
  tests/reproducibility --ignore=tests/reproducibility/test_original_paper_contracts.py
```

For STANDARD and the 66-member FULL hash pass, use the exact commands in
`docs/REPRODUCIBILITY_TEST_BASELINE.md`. These verification paths are local and API-free.

### Scientific reproduction stages

1. Build or restore Generator D under `data/production_generator_d_v1/` using the retained
   builders in `data_processing/` and `data/build_subgraph_training_data.py`.
2. Train/evaluate the DEV cross-encoder with
   `experiments/cross_encoder_reranking_dev_v1/run_experiment.py` and its `PROTOCOL.md`.
   This stage requires a GPU and the pinned DistilBERT snapshot.
3. Generate frozen TEST rankings with explicit policies:

   ```powershell
   .\.venv\Scripts\python.exe evaluation/run_cross_encoder_test_retrieval.py generate `
     --cross-encoder-policy-dir outputs/development_runs/question_candidate_cross_encoder_v1_cross_encoder_adaptive_k `
     --final-sageqa-policy-dir outputs/development_runs/question_candidate_cross_encoder_v1_final_sageqa_adaptive_k `
     --output-dir outputs/final_results/question_candidate_cross_encoder_v1_adaptive_test_a40
   ```

4. Export canonical retrieval locally with
   `python evaluation/export_manuscript_retrieval_results.py`.
5. Freeze reader inputs locally with
   `python generation/run_final_manuscript_answer_generation.py preflight`.
6. Generate answers with
   `python generation/run_final_manuscript_answer_generation.py generate`. This stage
   requires provider credentials and external API access.
7. Evaluate and finalize cached predictions without an API call:

   ```powershell
   .\.venv\Scripts\python.exe generation/run_final_manuscript_answer_generation.py evaluate --source-root .
   .\.venv\Scripts\python.exe evaluation/finalize_final_manuscript_end_to_end.py
   ```

8. Reproduce comparison families separately:
   - baselines: `evaluation/run_production_test_baselines.py` followed by
     `generation/run_final_manuscript_baselines.py`;
   - graph ablation: `evaluation/run_production_test_retrieval.py` with the explicit
     hard-pair checkpoint and policy paths from `thesis_graph_ablation/index.yaml`;
   - complete ground-truth support reference condition:
     `generation/run_gold_support_complete_oracle.py` (`preflight`, `generate`, then
     `evaluate`); only `generate` requires external API access;
   - stage-wise analysis: `python evaluation/analyze_stagewise_test_errors.py`.

The indexed `884480be53c554edae70d3b2d8e781847590aa69` revision is the scientific
thesis-source baseline. Frozen artifacts and hashes, rather than the literal current HEAD
after documentation changes, define the reported Chapter 7 state.
