# Duplicate and superseded code audit

## Decision rule

“Superseded” means “not the canonical final thesis path.” It does not mean safe to delete. Every group below retains either a compatibility dependency, historical reproducibility role, or negative-result record.

## Workflow groups

| Group | Canonical file(s) | Superseded/related files | Meaningful differences and live references | Recommendation |
|---|---|---|---|---|
| Adaptive support aggregation | `evaluation/adaptive_support_aggregation_v2.py`, `evaluation/evaluate_adaptive_support_aggregation_v2.py`, `evaluation/fit_production_dev_adaptive_k.py` | `evaluation/adaptive_support_aggregation.py`, `evaluation/evaluate_adaptive_support_aggregation.py` | V2 adds learned decision features/policies, but imports canonical identity and fixed-k helpers from V1. Readers/evaluators also import both. | Keep both in place until shared primitives are extracted; later archive only the V1 fitting/evaluation CLI. |
| Final cross-encoder retrieval | `experiments/cross_encoder_reranking_dev_v1/run_experiment.py`, `evaluation/prepare_cross_encoder_adaptive_dev.py`, `evaluation/preflight_cross_encoder_a40_test.py`, `evaluation/run_cross_encoder_test_retrieval.py` | `evaluation/run_production_test_retrieval.py`, `evaluation/run_production_dev_k_sensitivity.py` | The canonical path scores question-candidate pairs with the frozen DistilBERT cross-encoder. Older runners use GraphSAGE ranking and remain graph ablations. | Document the cross-encoder chain as default; retain GraphSAGE runners under historical/ablation documentation. |
| Manuscript retrieval export | `evaluation/export_manuscript_retrieval_results.py` | `evaluation/collect_final_results.py`, `evaluation/export_2wiki_predictions.py`, `evaluation/export_hotpot_predictions.py` | The canonical exporter is artifact-only, harmonizes frozen methods, and records sources in a self-excluding manifest. Older exporters serve paper-era formats. | Keep canonical exporter at a stable entry point; archive old exporters only with original-paper workflow. |
| Answer generation | `generation/run_final_manuscript_answer_generation.py` | `generation/run_production_test_answer_generation.py`, `generation/run_final_manuscript_baselines.py`, `generation/generate_hotpot_answers_with_llm.py`, `generation/generate_owl_answers_with_llm.py` | The new final runner consumes frozen cross-encoder selections plus hard-pair support identity and freezes inputs before generation. The production runner covers GNN-only/SAGE graph conditions; the baseline runner covers lexical/GNN-RAG/full-context; dataset readers are lower-level legacy CLIs. | Keep all until the new bundle is complete and audited; then make the final runner canonical and label the others ablation/baseline/original utilities. |
| GNN-RAG preparation/finalization | `data_processing/prepare_production_gnn_rag_clean.py`, `evaluation/run_production_gnn_rag_final_test.py`, `evaluation/finalize_production_gnn_rag_final_test.py` | `data_processing/prepare_familyowl_gnn_rag.py`, `data_processing/prepare_text_gnn_rag.py`, `evaluation/convert_gnn_rag_predictions.py`, `evaluation/finalize_production_gnn_rag_clean.py` | The production-clean path enforces frozen Generator D and clean relations. Dataset-specific adapters and conversion/finalization scripts preserve earlier workflows. | Keep production-clean baseline code; retain older adapters with the historical GNN-RAG bundle until a fresh-clone reproduction passes. |
| Candidate generation/builders | `data_processing/evidence_graph_candidates.py`, `data_processing/retrieval_contracts.py`, current text/ontology builders used by Generator D | `data_processing/build_hotpot_subgraph_dataset.py`, `data_processing/build_2wiki_subgraph_dataset.py`, paper-era behavior in `experiments/run_experiments.py` | Current builders implement the leakage-free candidate contract. Paper-era builders/results have different protocol semantics and are required to explain the leakage revision. | Do not merge histories. Publish final and paper-original protocol/config manifests separately. |
| Rejected ranking mechanisms | Cross-encoder final path | corrected-listwise, static-hard, RRF, disagreement-union, proof-gate, closure, beam-union, and compactness scripts listed in `ARCHIVE_CANDIDATES.md` | These implement different interventions and have paired DEV artifacts recording accepted/rejected decisions. Many hardcode output paths. | Archive source and outputs together under verdict-specific indexes; do not delete negative results. |
| Evaluation | Canonical retrieval exporter plus frozen answer evaluators used by `run_final_manuscript_answer_generation.py` | older `eval_*`, paper export, and one-off audit scripts | Metrics and cohort semantics differ. Replacing an evaluator would break the hash-locked evidence chain. | Freeze exact evaluator hashes per protocol; expose one documented verifier per bundle. |
| Test placement | `tests/` for supported tests | `evaluation/test_generator_d_production_smoke.py`, `experiments/cross_encoder_reranking_dev_v1/test_experiment.py` | Two tests live beside implementation; 23 tests in `tests/` are not tracked, and 18 missing-import statements occur in historical tests. | Keep experiment-local test with its protocol or move only with CI/import changes; build an explicit supported-test allowlist. |

## SHA-256-confirmed duplicate payload groups

The scan excluded `.git/`, `.venv/`, and `.worktrees/`. It found 108 groups with 887,816,341 bytes of theoretical redundancy. None is classified SAFE_DELETE solely because it is duplicated.

| Copies | Bytes each | Theoretical redundant bytes | SHA-256 prefix | Paths / interpretation | Action |
|---:|---:|---:|---|---|---|
| 3 | 87,304,616 | 174,609,232 | `05969966d00f` | HotpotQA GNN-RAG `test.json` copied under GNN data, LLM data, and LLM results | Review upstream layout; preserve until portable baseline verification. |
| 2 | 166,162,479 | 166,162,479 | `713661628434` | Hotpot raw `train-00001-of-00002.parquet` in distractor and fullwiki | Verify upstream dataset packaging/license before deduplication. |
| 2 | 165,624,177 | 165,624,177 | `76d3bb3048a7` | Hotpot raw `train-00000-of-00002.parquet` in distractor and fullwiki | Same as above. |
| 3 | 82,364,609 | 164,729,218 | `72aaee6aa650` | 2Wiki GNN-RAG `test.json` in GNN data, LLM data, and results | Preserve until baseline path abstraction. |
| 2 | 45,218,886 | 45,218,886 | `c77e28b51db4` | Incomplete Hotpot native-run `dev.json` duplicates third-party GNN data | Archive failed run; do not remove before chronology index. |
| 3 | 12,696,349 | 25,392,698 | `879bd54da4f8` | Clean Hotpot checkpoint in checkpoint root, successful output, and failed pre-prediction output | Keep authoritative checkpoint and frozen self-contained run; failed copy review later. |
| 2 | 20,901,465 | 20,901,465 | `b133619c60c9` | Successful and failed Hotpot native-run `test.json` | Failed-run archive decision. |
| 3 | 9,121,874 | 18,243,748 | `0b31c1f072d8` | Historical NeSyQA Hotpot test data/results copies | Retain under third-party historical bundle pending license review. |
| 3 | 8,931,974 | 17,863,948 | `6e8e0d22a754` | Historical NeSyQA 2Wiki test data/results copies | Same. |
| 3 | 5,493,374 | 10,986,748 | `4d2a039f2813` | FamilyOWL-2hop GNN-RAG test data/results copies | Preserve baseline self-containment. |
| 2 | 9,136,641 | 9,136,641 | `a5a3431d52d0` | Clean 2Wiki checkpoint duplicated into the successful frozen run | Intentional self-contained run; no deletion. |
| 3 | 3,538,470 | 7,076,940 | `aaf9cb95c99f` | Hotpot entity vocabulary in adapter, successful run, and failed run | Review only after authoritative adapter manifest. |

The remaining groups are mostly smaller GNN-RAG test JSON, `.info`, checkpoint, entity, vocabulary, and manifest copies. `cleanup_plan.json` therefore contains no duplicate-file deletion operation.

## Supersession map

1. Canonical: cross-encoder model selection and its two adaptive policies.
2. Supporting baseline: lexical, clean GNN-RAG, and full-context reference conditions.
3. Historical graph ablation: hard-pair GraphSAGE plus Text-Chain/Proof.
4. Original paper: `experiments/run_experiments.py`, original processed data/checkpoints/results.
5. Development archive: rejected, aborted, diagnostic, and superseded experiments.

Every public command and artifact index should include one of these five labels.
