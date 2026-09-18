# Scientific artifacts

This reviewer-facing map is derived from `release_manifests/`. It describes artifacts at
their current physical paths; the machine-readable indexes remain authoritative for exact
paths, hashes, lineage, and status. These documents do not move, rewrite, publish, or grant
permission to regenerate an artifact.

## Public artifact map

| Logical artifact | Protocol | Status | Purpose | Location | Manifest | Frozen? |
|---|---|---|---|---|---|---|
| Final thesis end-to-end | Final thesis protocol | `complete_frozen` | Canonical thesis answer, support, joint, and aggregate results | `outputs/final_results/final_manuscript_test_end_to_end/` | `release_manifests/thesis_final/index.yaml` and root `artifact_manifest.json` | Yes |
| Final thesis canonical retrieval | Final thesis, hard-pair-v2 lineage | `complete_frozen_artifact_only_export` | Canonical matched-cohort retrieval report | `outputs/final_results/manuscript_retrieval_results_hard_pair_v2/` | `release_manifests/thesis_final/index.yaml` and root `artifact_manifest.json` | Yes |
| Published-paper results | Published/original-paper protocol | `candidate_source_revision_unresolved` | Preserve reported historical results and execution layout | `outputs/full_results/` | `release_manifests/paper_original/index.yaml`; no artifact-bound top-level manifest | Yes, with unresolved source binding |
| Thesis baselines | Thesis baseline | `complete_frozen` | Lexical, clean GNN-RAG, and full-context comparisons | `outputs/final_results/final_manuscript_baselines_test_end_to_end/` | `release_manifests/thesis_baselines/index.yaml` and root `artifact_manifest.json` | Yes |
| Hard-pair GraphSAGE | Graph ablation | `complete_frozen_test_evaluation` | Separate graph-model lineage and TEST ablation | `outputs/final_results/production_generator_d_v2_hard_pair_test_retrieval/` and associated end-to-end/audit roots | `release_manifests/thesis_graph_ablation/index.yaml` | Yes |
| Earlier Generator-D-v1 thesis stages | Superseded thesis experiment | `externally_backed_up_and_removed` | Preserve earlier retrieval, policies, and answers for provenance | Restore mappings in `release_manifests/thesis_superseded/index.yaml` | `archive_manifests/removed/phase_8_final_cleanup.yaml` | External preservation only |
| Complete Gold Support oracle | Oracle | `complete_frozen_post_publication_oracle` | Complete 3,509-row support-bearing oracle analysis | `outputs/final_results/gold_support_complete_oracle/` | `release_manifests/oracle/index.yaml` and root `artifact_manifest.json` | Yes |
| Development archive | Development experiment | `externally_backed_up_and_removed` | Retain cleanup provenance without presenting rejected payloads as release artifacts | Restore mappings are recorded in Phase 7A, 7B, and 8 removal records | `release_manifests/development_archive/index.yaml` and records under `archive_manifests/removed/` | External preservation only |

## External restoration status

The availability labels below describe the current checkout and documented distribution
state. `external_location: pending` is deliberate: no DOI, archive URL, or model-hosting URL
has been verified for these frozen bundles.

| Requirement | Availability | Restoration/use | External location |
|---|---|---|---|
| Tracked source, release indexes, and reviewer documentation | `AVAILABLE_IN_REPOSITORY` | Present in a fresh clone | Not applicable |
| Raw public text benchmark inputs | `EXTERNAL_DOWNLOAD` | Obtain the exact dataset/version required by the relevant protocol; verify split semantics before use | `pending` |
| Frozen Generator D corpus under `data/production_generator_d_v1/` | `EXTERNAL_ARCHIVE`, `CURRENTLY_NOT_PUBLICLY_BOUND` | Restore all 66 manifest-bound members at the indexed path | `pending` |
| Original processed datasets under the six reported paper roots | `EXTERNAL_ARCHIVE`, `CURRENTLY_NOT_PUBLICLY_BOUND` | Restore without substituting thesis-final data | `pending` |
| Final cross-encoder, GraphSAGE, and clean GNN-RAG checkpoints | `EXTERNAL_ARCHIVE`, `CURRENTLY_NOT_PUBLICLY_BOUND` | Restore to the exact indexed checkpoint roots and verify hashes | `pending` |
| Pinned DistilBERT base snapshot | `EXTERNAL_ARCHIVE`, `CURRENTLY_NOT_PUBLICLY_BOUND` | Supply through the local cache or `SAGEQA_DISTILBERT_PATH`; fail closed on mismatch | `pending` |
| Frozen final, baseline, ablation, paper, and oracle output roots | `EXTERNAL_ARCHIVE`, `CURRENTLY_NOT_PUBLICLY_BOUND` | Restore to exact paths; validate embedded and release-manifest identities | `pending` |
| Rebuilt candidate datasets or new model outputs | `GENERATED` | Scientific workflow, not required for verification and never written over frozen roots | Not applicable |
| Hosted-reader answers or LLM-based KG construction | `PROVIDER_DEPENDENT` | Requires explicit credentials and provider availability; cached frozen answers are preferred for exact verification | Not applicable |
| GNN-RAG-specific models/data/configuration beyond the vendored source | `EXTERNAL_ARCHIVE`, `CURRENTLY_NOT_PUBLICLY_BOUND` | Restore according to the baseline index and upstream requirements | `pending` |

A fresh clone is therefore sufficient for source/document inspection but not for complete
artifact verification or end-to-end scientific reproduction.

## Published paper

The reported historical pipeline is `experiments/run_experiments.py` plus its paper-era builders, readers, evaluators, exporters, six processed dataset roots, six GraphSAGE checkpoint roots, raw inputs, and the six reported children and aggregate files under `outputs/full_results/`. Paper-era Pizza branches were not reported and are not retained. Commit `dbdbb507` is the strongest source candidate because it is tagged `conference-submission`, but the exact publication revision is not formally proven. See `docs/ORIGINAL_PAPER_SOURCE_REVISIONS.md`.

## Final thesis

The final thesis path is the frozen Generator D candidate corpus, frozen DistilBERT question-candidate cross-encoder, Text-Chain for text or Proof reranking for ontology, frozen adaptive support policies, and support-grounded answer generation. The canonical hard-pair retrieval and final end-to-end results are distinct from both the paper pipeline and earlier Generator-D-v1 results.

The final thesis answer bundle has 4,249 answer rows. Support/Joint metrics use the 3,509 rows with defined non-empty gold support; the 740 excluded rows are not scored as zero. Equal-dataset macro is the primary aggregate.

## Thesis baselines

Lexical, clean GNN-RAG, and full-context comparisons use the same frozen TEST cohort. Their complete frozen end-to-end bundle is `outputs/final_results/final_manuscript_baselines_test_end_to_end/`; clean retrieval inputs and models remain under `outputs/final_results/production_generator_d_v1_test_baselines/` and `checkpoints/production_generator_d_v1_gnn_rag_clean/`.

## Graph ablation

The hard-pair GraphSAGE architecture is indexed separately under `release_manifests/thesis_graph_ablation/`. Its checkpoint, DEV policies, TEST retrieval, end-to-end result, and audit remain physically separate and hash-bound.

## Superseded and development results

Valid earlier thesis stages remain scientific provenance in the Phase 8 preservation backup and `thesis_superseded` restore index. Phase 8 backed up, hash-verified, fully restore-tested, and removed all remaining outside-closure scientific and development families. Planning and removal provenance remains under `archive_manifests/`. Preservation backups are not scientific release artifacts.

## Complete Gold Support oracle

`outputs/final_results/gold_support_complete_oracle/` is a post-publication, post-final-run oracle analysis built from the order-preserving union of all persisted gold support alternatives. It is complete for 3,509 support-bearing rows and is frozen separately. It must never replace the historical `gold_support/oracle` condition in the final thesis bundle.

## Provenance correction

`outputs/final_results/provenance_correction_v1/provenance_manifest.json` records the corrected complete-oracle lineage and the hard-pair-v2 retrieval lineage while explicitly preserving the historical Gold Support and earlier manuscript retrieval exports. The correction is additive, not a rewrite.
