# Experiment generations

## PUBLISHED PAPER

The published-paper generation is the historical `experiments/run_experiments.py` workflow using original processed datasets, original GraphSAGE checkpoints, historical readers/evaluators/exporters, raw inputs, and `outputs/full_results/`. `dbdbb507` is only a candidate source revision until artifact and environment lineage prove it authoritative. Paper reproduction may require downloads, GPU training, and hosted-reader APIs; the saved outputs must remain available for artifact-only verification.

## FINAL THESIS

The final thesis is a different protocol, not regenerated paper numbers:

1. frozen Generator D candidate construction;
2. frozen DistilBERT question-candidate cross-encoder;
3. Text-Chain for text and Proof reranking for ontology;
4. frozen adaptive support aggregation;
5. support-grounded answer generation; and
6. evaluation only after predictions are frozen and gold is joined.

The final source revision indexed by this phase is `884480be53c554edae70d3b2d8e781847590aa69`. Historical V1 retrieval exports and hard-pair-v2 exports remain distinct. Baselines and graph ablations have separate logical bundles and cannot be substituted for the main method.

## POST-PUBLICATION ORACLE ANALYSIS

The complete Gold Support oracle is a separate post-publication analysis. It uses all persisted gold support alternatives for the 3,509 support-bearing rows. The historical Gold Support result used one historical condition and remains frozen in the 16,256-row final thesis prediction bundle. The complete oracle is stored under its own root and provenance correction; it does not revise the historical result in place.

## Gold firewall

Final-thesis retrieval, reranking, support selection, and reader inputs are frozen before gold joins. Undefined or empty support is excluded from Support/Joint denominators. Development experiments must not be presented as TEST confirmation, and no release index authorizes a rerun.
