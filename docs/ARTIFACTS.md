# Scientific artifacts

This document describes the artifacts at their current physical paths. Logical bundle metadata is under `release_manifests/`; those files do not move or rewrite artifacts.

## Published paper

The historical pipeline is `experiments/run_experiments.py` plus its paper-era builders, readers, evaluators, exporters, ten processed dataset roots, ten `checkpoints/gnn_subgraph_ranker_*_full/` roots, raw inputs, and `outputs/full_results/`. Commit `dbdbb507` is the strongest source candidate because it is tagged `conference-submission`, but the exact publication revision is not formally proven. See `docs/ORIGINAL_PAPER_SOURCE_REVISIONS.md`.

## Final thesis

The final thesis path is the frozen Generator D candidate corpus, frozen DistilBERT question-candidate cross-encoder, Text-Chain for text or Proof reranking for ontology, frozen adaptive support policies, and support-grounded answer generation. The canonical hard-pair retrieval and final end-to-end results are distinct from both the paper pipeline and earlier Generator-D-v1 results.

The final thesis answer bundle has 4,249 answer rows. Support/Joint metrics use the 3,509 rows with defined non-empty gold support; the 740 excluded rows are not scored as zero. Equal-dataset macro is the primary aggregate.

## Thesis baselines

Lexical, clean GNN-RAG, and full-context comparisons use the same frozen TEST cohort. Their complete frozen end-to-end bundle is `outputs/final_results/final_manuscript_baselines_test_end_to_end/`; clean retrieval inputs and models remain under `outputs/final_results/production_generator_d_v1_test_baselines/` and `checkpoints/production_generator_d_v1_gnn_rag_clean/`.

## Graph ablation

The hard-pair GraphSAGE architecture is indexed separately under `release_manifests/thesis_graph_ablation/`. Its checkpoint, DEV policies, TEST retrieval, end-to-end result, and audit remain physically separate and hash-bound.

## Superseded and development results

Valid earlier thesis stages remain scientific provenance and are indexed as `thesis_superseded`; rejected, diagnostic, exploratory, aborted, or incomplete work is indexed as `development_archive`. Neither label means disposable. No physical archive operation is authorized by these indexes.

## Complete Gold Support oracle

`outputs/final_results/gold_support_complete_oracle/` is a post-publication, post-final-run oracle analysis built from the order-preserving union of all persisted gold support alternatives. It is complete for 3,509 support-bearing rows and is frozen separately. It must never replace the historical `gold_support/oracle` condition in the final thesis bundle.

## Provenance correction

`outputs/final_results/provenance_correction_v1/provenance_manifest.json` records the corrected complete-oracle lineage and the hard-pair-v2 retrieval lineage while explicitly preserving the historical Gold Support and earlier manuscript retrieval exports. The correction is additive, not a rewrite.
