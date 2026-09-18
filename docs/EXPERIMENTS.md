# Experiment generations

This is the canonical protocol map. Commands and dependency boundaries are in
[`../REPRODUCE.md`](../REPRODUCE.md); artifact locations and restoration status are in
[`ARTIFACTS.md`](ARTIFACTS.md). Directory names are preserved even where the preferred
reviewer-facing terminology differs.

| Category | Preferred term | Scientific role |
|---|---|---|
| `paper_original` | Published/original-paper protocol | Historical published workflow and outputs |
| `thesis_final` | Final thesis protocol | Canonical current thesis architecture and results |
| `thesis_baselines` | Thesis baseline | Matched lexical, clean GNN-RAG, and full-context comparisons |
| `thesis_graph_ablation` | Graph ablation | Hard-pair GraphSAGE ablation, separate from the cross-encoder |
| `thesis_superseded` | Superseded thesis experiment | Valid earlier thesis stages externally preserved for provenance |
| `development_archive` | Development experiment | External preservation records for DEV-only, rejected, exploratory, diagnostic, aborted, or incomplete work |
| `oracle` | Complete Gold Support oracle | Explicit post-publication oracle, distinct from Historical Gold Support |

## Published/original-paper protocol

The published-paper generation is the historical `experiments/run_experiments.py` workflow using original processed datasets, original GraphSAGE checkpoints, historical readers/evaluators/exporters, raw inputs, and `outputs/full_results/`. `dbdbb507` is only a candidate source revision until artifact and environment lineage prove it authoritative. Paper reproduction may require downloads, GPU training, and hosted-reader APIs; the saved outputs must remain available for artifact-only verification.

## Final thesis protocol

The final thesis is a different protocol, not regenerated paper numbers:

1. frozen Generator D candidate construction;
2. frozen DistilBERT question-candidate cross-encoder;
3. Text-Chain for text and Proof reranking for ontology;
4. frozen adaptive support aggregation;
5. support-grounded answer generation; and
6. evaluation only after predictions are frozen and gold is joined.

The final source revision indexed by this phase is `884480be53c554edae70d3b2d8e781847590aa69`. Historical V1 retrieval exports are externally preserved and remain distinct from the retained hard-pair-v2 export. Baselines and graph ablations have separate logical bundles and cannot be substituted for the main method.

## Thesis baselines and graph ablation

Thesis baselines and the graph ablation use separate frozen bundles. They are comparisons,
not aliases for the final thesis method. Their authoritative roots and manifests are in
`release_manifests/thesis_baselines/` and `release_manifests/thesis_graph_ablation/`.

## Superseded thesis and development experiments

Superseded thesis experiments are valid historical results that no longer define the
canonical thesis pipeline. Development experiments include rejected or diagnostic work.
Both categories were preserved in verified external backups before removal from the final
repository and must not be presented as retained release artifacts or final TEST evidence.

## Post-publication oracle analysis

The complete Gold Support oracle is a separate post-publication analysis. It uses all persisted gold support alternatives for the 3,509 support-bearing rows. The historical Gold Support result used one historical condition and remains frozen in the 16,256-row final thesis prediction bundle. The complete oracle is stored under its own root and provenance correction; it does not revise the historical result in place.

## Gold firewall

Final-thesis retrieval, reranking, support selection, and reader inputs are frozen before gold joins. Undefined or empty support is excluded from Support/Joint denominators. Development experiments must not be presented as TEST confirmation, and no release index authorizes a rerun.
