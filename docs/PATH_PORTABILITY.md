# Active-source path portability

Active thesis-final, thesis-baseline, hard-pair graph-ablation, and complete-oracle
orchestration resolves repository paths through `utils/paths.py`. Defaults are anchored to
the utility's source location rather than the process working directory. Existing relative
CLI path values are interpreted relative to the repository root; absolute CLI values remain
absolute.

Optional overrides are `SAGEQA_REPO_ROOT`, `SAGEQA_DATA_ROOT`,
`SAGEQA_CHECKPOINTS_ROOT`, and `SAGEQA_OUTPUTS_ROOT`. An override must name an existing
directory. `SAGEQA_REPO_ROOT` must also contain `pyproject.toml` and
`release_manifests/`; invalid values fail instead of falling back. Importing the utility
does not create directories or files.

Run the read-only resolution check from any working directory:

```text
python evaluation/check_thesis_final_paths.py
```

It reports the thesis-final logical bundle, resolved Generator D root, cross-encoder
checkpoint, adaptive policies, frozen TEST selection, retrieval export, answer bundle,
existence status, and release-manifest IDs/hashes. It does not read `.env`, invoke a model,
call an API, or mutate an artifact.

## Fresh-checkout review

The active first-party Python source requires no edit when a checkout is located at either
`D:\research\SAGE-QA` or `/home/research/SAGE-QA`. The following are expected external
dependencies, not active path bugs:

- Git-ignored or separately distributed scientific assets under `data/`, `checkpoints/`,
  and `outputs/` must be restored with their release-manifest identities and hashes.
- The pinned DistilBERT base snapshot must be available in the local Hugging Face cache or
  supplied through the existing `SAGEQA_DISTILBERT_PATH` fail-closed override. Frozen
  cross-encoder inference uses the repository output checkpoint.
- Python dependencies and, for the original one-time inference runs, the documented GPU
  environment must be installed. Baseline GNN-RAG execution also requires the preserved
  `third_party/GNN-RAG` tree and its environment.
- Hosted-reader generation requires separately configured API credentials and provider
  availability. Reproducibility verification uses the frozen, hashed answer artifacts and
  does not require an API call.
- Historical absolute paths in frozen metadata, frozen experiment records, audit documents,
  and vendor material remain intentionally byte-identical. The descriptive
  `release_manifests/path_maps/frozen_path_map.yaml` is not a runtime rewrite mechanism.

No active path bug remains in the migrated orchestration set. PAPER_ORIGINAL remains
isolated: `experiments/run_experiments.py` does not import the active path layer and its
candidate `dbdbb507` contract is unchanged.
