# Phase 2 reproducibility gate

Date: 2026-09-26 (Europe/Luxembourg)

## Gate verdict

The published-paper and leakage-free thesis versions are now independently identifiable. The paper's surviving data, checkpoints, and results are locally preserved and checksummed, but the exact source-to-results binding remains unproven. The final hard-pair-v2 thesis artifacts remain preserved. The optimized symbolic-coefficient downstream results are independently verifiable, but the original selected-coefficient/adaptive-policy bundle and raw frozen TEST selection file have not been recovered. Repository cleanup must therefore remain blocked on external archival closure, not on the scientific result values.

No training, inference, answer generation, scientific-result rewrite, release, merge, tag, push, deletion, move, reset, clean, stash, or history rewrite was performed.

## 1. Preserved starting state

- Original checkout: `C:/Users/julie/github/PhD/SAGE-QA`
- HEAD: `75010ef4dffde3fcbd89cc8669da3d63423924b0`
- Branch/upstream: `feature/symbolic-coefficients-original-v1` / `origin/feature/symbolic-coefficients-original-v1`, ahead 0, behind 0
- Staged change: mode-only `100644 -> 100755` for `experiments/symbolic_coefficients_original_v1/run_test_on_gpu.sh`
- Unstaged tracked status entries: 23 files (21 have content diffs; two are normalization-only status entries). They are retained only in the original checkout and were not copied into this snapshot.
- Untracked state includes the OWL v1.1 dataset/source/test family plus four optimized-generation/stage-wise source/test files. The exact accessible file-level state is frozen in `release_manifests/phase2/starting_git_state.txt`.
- Status could not enumerate two ACL-protected scratch directories: `.pytest-tmp-optimized-answer/` and `pytest-of-julie/`. They are ignored scratch state, not claimed absent.

Isolated preparation:

- Worktree: `C:/Users/julie/github/PhD/SAGE-QA/.worktrees/phase2-reproducibility-cleanup`
- Branch: `phase2/reproducibility-cleanup-20260926`
- Base: exact HEAD `75010ef4dffde3fcbd89cc8669da3d63423924b0`
- The original checkout and its index remain intact.

The isolated snapshot deliberately excludes `data/production_generator_d_v1_1/`, `data_processing/ontology_benchmark_schema.py`, `docs/GENERATOR_D_V1_1_BOUNDED_PREDICATE_CONNECTIVITY.md`, `evaluation/compare_generator_d_v1_1_dev.py`, `evaluation/smoke_owl_v1_1_compatibility.py`, and `tests/test_ontology_v1_1_compatibility.py`, as well as the OWL-v1.1 portions of mixed tracked files.

## 2. Optimized-thesis recovery

### Recovered and verified

The original coefficient-optimization inputs are present and passed `prepare_dev_cache.py validate-inputs` without writing derived artifacts:

| Artifact | SHA-256 |
|---|---|
| `dev_predictions_frozen.jsonl` | `7d5a6e9c0afbc9b7e13669da27c51fdff559c35b64781e8f0fdbc52f68172e34` |
| `prediction_freeze.json` | `92b1b94886408e2b510d106de795de36cf48e2dec481a99db44b4632810f69d7` |
| `preflight_manifest.json` | `624d5941d96115aadbc4d23a11f4d65d14d4cb29e0acdd1a65653a4039754bf1` |
| Cross-Encoder `model.safetensors` | `02b06c91fa422369a0b0adbab3b33bfcf80f335b7c8a5911084c3b82af117ba9` |

All ten DEV candidate hashes in the preflight manifest also validated. These inputs support a future independent re-execution of the declared DEV optimization protocol, but such a rerun would produce a new reproduction artifact and must not be mislabeled as recovery of the original derived bytes.

Completed downstream bundles are present and hash-verifiable. The answer bundle freezes 8,498 predictions (`712fce04...`) over 4,249 questions, and its reader-input bundle (`08e493b2...`) retains both optimized `k1` and adaptive selected supports. The stage-wise bundle manifest is `a8d6cf73...`. Exact hashes are in `release_manifests/thesis_optimized/index.yaml`.

### Not recovered

- `SAGEQA_GPU_handoff` is absent at its recorded path.
- `test_predictions_frozen.jsonl` with required SHA-256 `770ef934fca43279b875a5c711765d7771f600fdcfd382590070d2d166293264` is absent.
- The derived DEV reference bundle is absent, including selected coefficients `9cc3d4b8...`, `original_vs_selected_dev.json` `0c08b16f...`, and adaptive-policy manifest `45f771a3...`.
- A filename and hash scan covered the repository's 750 output files (1,636,470,925 bytes), the common user artifact locations, Google Drive `G:`, and filenames inside 65 local ZIP archives. Only the three original DEV inputs above matched; no expected derived or TEST selection artifact matched.

Affected reproducibility levels:

- **Exact archival reproduction is blocked**: the upstream frozen coefficient/policy and TEST selection bytes cannot be independently compared.
- **Protocol reproduction is available**: code, pinned input bytes, checkpoint, grids, tie-breaking, and all ten DEV candidate files survive.
- **Result verification is available**: downstream frozen reader inputs, answers, metrics, and per-example stage-wise outputs bind to the missing retrieval SHA through independent manifests. The canonical overlay recomputes Support P/R/F1 from the retained selected supports and frozen gold alternatives and matches every persisted Support F1. This verifies reported values but is not recovery of the missing ranking file or its candidate scores.

The end-to-end directory's old `artifact_manifest.json` still describes only preflight files. It is preserved as historical state. `release_manifests/thesis_optimized/index.yaml` is the non-destructive completion inventory.

## 3. Original publication provenance

`conference-submission` resolves to `dbdbb50708bdc6c686ef82518ec71c1d1bf55985` (2026-06-05). Its `experiments/run_experiments.py` names the exact six surviving processed-data, GraphSAGE-checkpoint, and result roots. The method/result layout under each result root is consistent with that runner.

This is strong path- and protocol-level association, not an authoritative byte binding. The tag does not contain the ignored processed datasets, checkpoints, result bundles, environment lock, hosted-reader snapshot, or a contemporaneous artifact manifest. Persisted results do not record a source commit. Therefore `dbdbb507` remains `candidate_not_authoritative`; no new publication tag is justified.

The historical protocol must be preserved with its limitation: the paper-era text builders constructed candidate pools for every split, including TEST, by explicitly inserting gold support, adding gold-plus-distractor candidates, and prioritizing candidates by gold overlap. The model feature path disabled gold features, but the TEST candidate universe itself was gold-informed. These results reproduce the published historical protocol; they must not be described as leakage-free or silently compared as if generated under the thesis gold firewall.

Canonical tree digests below hash UTF-8/LF lines of the form `file_sha256  relative/posix/path`, sorted by relative path. They are directory identity digests, not hashes of an invented archive.

| Setting | Processed data tree SHA-256 | Checkpoint tree SHA-256 | Result tree SHA-256 |
|---|---|---|---|
| HotpotQA | `0f1d7cc513db78655c4fd2fc77d412372e331cede025b1410383fa60b0967f3c` | `0784bc103dff3d29210b4980cfb7b6c50f871cbcbed1db014550f05ec013a2b2` | `03ee5ba3b2ae976e0664e166adb428b4f091056c9dfa41708dfca13ab52fd611` |
| 2WikiMultiHopQA | `ac016eea85a323682a4dbe717b962ae17f0e4f088a51c7c19f55dd6d13acd781` | `d523d3135f4f28e7eb4ed3e6588bf7fa5b3ab7f69c0d7254233fd9e073c9fc56` | `32513815c631a5e7d89b3bf7969991d676d3d8a9827177ccc9d70ca0274cd1bf` |
| FamilyOWL 1-hop | `f9c20511957407014c91ed0427cc7e334648769bc41d48846c564e04789082a9` | `52c42f1d21d7a974ca67da92fc0d960d56d7c76b174c050446a8d7c83e2cdc92` | `c66547b3e89378fa80de592053a2c4b3e538d3f6a75b47b11154dcf12bc1ce71` |
| FamilyOWL 2-hop | `eab70f60dbfefcb5f17c418ba9cc1984d449919bc1bdaabc5dcf5a58d2f1bb24` | `d71e8814ae9f421482f105e69675d8224a29fa3ac5228a9ca17d5ca8e068677d` | `a08f3b1ce225c9dd0bec57ab6ee3f4228b5e8f8a6cb0d141a39bf3944ffd253a` |
| OWL2Bench 1-hop | `6c334b498cefc968c87e678133037b2e3e08a599dc10f4e4f8d864a3809c5a0e` | `2db9157849e89e4dcb1f0753a6f54b3488ab6b3aada90adeef9f6a971e7b4a58` | `e97ae93f78b28034876bdd22bc5a5a9a2c6cd02289c4ba9b804372f6b4921588` |
| OWL2Bench 2-hop | `f4e9cd5cf70b1750f20d4c0baf6f0349c765ef8ab8b79430d365ab398b81c832` | `d6471f6b541f0d3d3cc978fa5b211defb33d77cd4fd013ac6a8da726e883c38d` | `733d9174c470f09881b52f06581a7e859a9ad682a1ea4e9edbfc497beb830d16` |

## 4. Proposed final-thesis snapshot

Git-tracked source and metadata:

1. Base source revision `75010ef4...`, including `experiments/symbolic_coefficients_original_v1/`.
2. The verified optimized answer-generation and stage-wise analysis scripts and their focused tests.
3. Existing `release_manifests/thesis_final`, `thesis_graph_ablation`, `thesis_baselines`, and `oracle` indexes.
4. `release_manifests/thesis_optimized/index.yaml`, the separate optimized canonical retrieval overlay, this gate report, and the exact starting-state record.
5. The executable-bit change for `run_test_on_gpu.sh`, preserved as a snapshot change but not taken from the OWL v1.1 work.

External checksummed archives:

- Paper: six processed dataset roots, six paper GraphSAGE checkpoints, six result roots plus aggregate paper result files, required raw inputs, and hosted-reader/environment metadata if later recovered.
- Thesis: Generator D v1 (66-entry aggregate `27a4c149...`), cross-encoder checkpoint and policies, hard-pair-v2 checkpoint/policies/retrieval/end-to-end bundles, baselines, oracle, final manuscript outputs, optimized DEV reference/TEST selection/end-to-end/stage-wise bundles.

The earlier hard-pair-v2 retrieval export remains unchanged at `outputs/final_results/manuscript_retrieval_results_hard_pair_v2/`; the older V1 export remains separately preserved. The new `canonical_retrieval_overlay.json` adds the verified optimized values by reference and does not overwrite either historical export.

Large datasets, checkpoints, JSONL predictions, model weights, and result bundles should remain outside normal Git history and enter an immutable external package with per-file SHA-256 plus package SHA-256. Small source, tests, protocols, release indexes, and derived tabular/JSON summaries belong in Git.

## 5. Remaining blockers

1. Recover the original derived DEV reference bundle and `770ef934...` TEST selection file, ideally from the GPU host, shell history, transfer destination, backup, or cloud trash/version history.
2. Bind the paper tree digests to a contemporaneous archive/source/environment record. Until then, paper rerun compatibility and source identity are separate claims.
3. Archive exact dependency/model/provider identity for the historical hosted-reader calls; persisted answers survive, but a fresh hosted rerun is not independently guaranteed.
4. Package and verify ignored thesis artifacts externally. A fresh clone still cannot perform full reproduction.

## 6. Minimal next sequence

1. **Documentation:** review this gate, explicitly label paper candidate-source status and gold-informed TEST candidate construction, and add the optimized index to reviewer-facing artifact documentation.
2. **CI repair:** run only source/unit/reproducibility contracts from the isolated snapshot; keep archival failures classified rather than changing historical code or outputs.
3. **Archive packaging:** build two immutable packages, `paper_original_artifacts` and `thesis_ch7_artifacts`, with per-file manifests and package hashes. Mark the optimized upstream sub-bundle incomplete unless the exact missing hashes are recovered.
4. **Release preflight:** restore packages into a fresh clone/worktree, run hash/path contracts and non-billable reproduction checks, and compare reported tables. Do not publish yet.
5. **Eventual cleanup:** only after both package manifests pass, propose a literal deletion/move allowlist. Preserve both research versions; never rewrite or retag the historical publication state.
