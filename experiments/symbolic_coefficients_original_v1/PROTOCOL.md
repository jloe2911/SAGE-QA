# Original-v1 symbolic coefficient optimization protocol

## Scope and immutable inputs

This experiment changes only the numerical coefficients of the original
SAGE-QA Text-Chain and Proof-aware rerankers. It consumes the frozen original
CORE-LLM-Bench v1.0.0 DEV candidate files and frozen Cross-Encoder DEV logits.
It does not train a model, generate candidates, run Cross-Encoder inference,
change feature definitions, use v1.1 data/schema policy, or access TEST.

Text and ontology selection are two independent experiments. Text uses only
HotpotQA and 2WikiMultiHopQA DEV. Ontology uses only the eight original
FamilyOWL, Pizza and OWL2Bench DEV conditions. No dataset-specific coefficient
set is permitted.

The original input identities are pinned as follows:

- Cross-Encoder DEV predictions SHA-256:
  `7d5a6e9c0afbc9b7e13669da27c51fdff559c35b64781e8f0fdbc52f68172e34`
- prediction freeze SHA-256:
  `92b1b94886408e2b510d106de795de36cf48e2dec481a99db44b4632810f69d7`
- original preflight manifest SHA-256:
  `624d5941d96115aadbc4d23a11f4d65d14d4cb29e0acdd1a65653a4039754bf1`
- Cross-Encoder checkpoint SHA-256 recorded by that freeze:
  `02b06c91fa422369a0b0adbab3b33bfcf80f335b7c8a5911084c3b82af117ba9`
- committed original symbolic implementation blob:
  `0da6fd142196ddfed2d2a392e5a27440d55f6918`

The pinned preflight manifest supplies and validates the ten individual DEV
candidate-file hashes. A changed manifest cannot authorize changed inputs.

## Frozen formulas

The cache stores the original feature values. The adjusted score is the frozen
Cross-Encoder logit plus the symbolic adjustment. Ranking ties are resolved by
the SHA-256 identity of the canonical JSON ordered `subgraph_units` list.

Text adjustment:

`wT*T + wQ*Q + wD*D + wB*B + wK*K + wC*C - p3*I[n=3] - p4*I[n=4] - p5*max(0,n-4) - pdup*I[n>=2 and unique_titles<=1]`

Original text vector, in formula order:
`0.030, 0.020, 0.020, 0.025, 0.030, 0.020, 0.002, 0.004, 0.010, 0.010`.

Proof adjustment:

`wP*P + wQ*Q + wF*F + wc*(wp*Rp + we*Re + wf*F - ps*max(0,n-2) - px*max(0,schema_count-1)) - pu*max(0,n-2)`

Original proof vector, in formula order:
`0.180, 0.030, 0.020, 0.350, 0.010, 0.025, 0.015, 0.020, 0.006, 0.004`.

The feature implementations and defaults are locked in
`original_symbolic_features.py`. Text searches the six existing reward weights,
one coefficient preserving the original piecewise size shape
(`.2*w` at size 3, `.4*w` at size 4, and `w*(n-4)` above 4), and the existing
duplicate-page penalty. Proof searches its five outer coefficients. The five
nested Compact coefficients remain fixed because jointly changing them and the
Compact multiplier is non-identifiable.

## Predeclared grids

Each grid is ordered low to high and contains the historical value.

| Text coefficient | Grid |
|---|---|
| question title | 0 to .060, step .005 |
| question overlap | 0 to .040, step .005 |
| cross page | 0 to .040, step .005 |
| bridge overlap | 0 to .050, step .005 |
| KG connectivity | 0 to .060, step .005 |
| comparison | 0 to .040, step .005 |
| piecewise size coefficient | 0 to .020, step .002 |
| duplicate-page penalty | 0 to .020, step .002 |

| Proof coefficient | Grid |
|---|---|
| proof bonus | 0 to .300, step .010 |
| query coverage | 0 to .060, step .005 |
| schema mix | 0 to .040, step .005 |
| compact multiplier | 0 to .700, step .050 |
| outer extra-unit penalty | 0 to .020, step .002 |

## Objective and search

For each coefficient vector, select the rank-1 candidate for every example,
compute Support F1 against the best original DEV gold explanation, average F1
within each dataset, then take the unweighted mean of those dataset means.
This equal-dataset macro DEV Support F1 at k=1 is the sole objective.

Seed 42 is recorded; search uses no stochastic sampling. Three starts are used:
the historical vector, zero symbolic adjustment, and twice the historical
vector clipped to each grid. Each start uses the fixed table order above. For
each coordinate, evaluate every grid value while holding the other coordinates
fixed. Stop at convergence or after ten complete passes per start.

Tie-breaking is, in order: higher objective; fewer datasets below their own
historical baseline; larger worst-dataset delta; smaller normalized L1 distance
from the historical vector; then lexicographically smaller coefficient tuple.
Candidate-score ties use candidate identity as stated above. If the winning
macro F1 does not exceed the historical objective by more than `1e-12`, freeze
the entire historical vector.

## Reproducibility and execution boundary

Every required input is hash checked before feature computation. Prediction and
candidate identity sets must match exactly for every example. Cache and output
files use canonical/sorted JSON and contain no wall-clock timestamps. Every
reader rejects a path identified as TEST and every artifact records DEV scope
and zero TEST access.

Run the commands in `RUN_EXPERIMENT.md` in order. Optimization, ranking
materialization and adaptive-k fitting are intentionally not performed during
code validation.
