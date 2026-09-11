# Question-candidate cross-encoder reranking: frozen DEV protocol

Status: originally predeclared configuration, unchanged for the A40 rerun.

The local `distilbert-base-uncased` feasibility attempt was interrupted after
250/4,908 TRAIN-only optimizer steps because its measured runtime projected to
approximately seven hours on the available GTX 1650 Ti. It produced no saved
checkpoint and no DEV predictions or metrics. The abort is recorded in
`ABORTED_DISTILBERT_FEASIBILITY.md`. It is not an experimental result. The exact
original configuration below is to be rerun on the A40; no replacement model is
authorized and no DEV outcome informed this handoff.

## Scope and immutable inputs

- Development-only experiment over the existing `data/production_generator_d_v1`
  TRAIN and DEV candidate artifacts for all ten production datasets.
- Generator D, its 512-candidate pool, the gold-free inference admission cap of
  320, Text-Chain, Proof, and all reader/answer-generation code remain unchanged.
- TEST files, TEST-derived mining, OpenAI APIs, and answer generation are excluded.
- One shared cross-encoder is trained across the ten datasets. There is no model,
  architecture, loss, epoch, learning-rate, or threshold selection from DEV.

## Model and deterministic serialization

The pretrained encoder is `distilbert-base-uncased`, loaded
with `local_files_only=True`, with its single-logit sequence-classification head.
Each candidate is encoded as one sequence:

```text
Question:
<question>

Candidate evidence:
[1] Title: <title>
Sentence index: <index>
Text: <sentence>
...
```

For ontology candidates each ordered evidence unit is instead emitted faithfully
as `[i] <original EvidenceUnit text>`. Candidate unit order is never changed.

## Target and objective

The target is exactly `ranking_target` from
`training/train_gnn_subgraph_retriever.py`: exact complete support = 1.0,
complete-support supersets = 0.9, partial support =
`min(0.3, 0.5 * best_set_f1_to_gold)`, and irrelevant support = 0.0.

For a question `q`, a higher-target candidate `C+`, and a lower-target candidate
`C-`, the sole training objective is pairwise margin ranking loss:

`L(q,C+,C-) = max(0, 0.2 - s(q,C+) + s(q,C-))`.

All comparisons are within one question. Up to eight pairs per TRAIN question are
selected deterministically across distinct target strata, prioritizing adjacent
strata and the highest-versus-lowest stratum. This directly includes complete >
partial and partial > irrelevant comparisons whenever those strata exist.

## Fixed training configuration

- seed: 42
- epochs: 1
- optimizer: AdamW
- learning rate: 2e-5
- weight decay: 0.01
- pair batch size: 16 (32 encoded sequences)
- maximum sequence length: 256 wordpieces
- maximum pairs per question: 8
- gradient clipping: 1.0
- CUDA mixed precision: enabled when CUDA is available
- checkpoint selection: none; evaluate the final one-epoch model once

## DEV comparisons and decision rule

The four fixed-k comparisons are existing GNN k1, cross-encoder k1, existing
SAGE-QA-final k1, and cross-encoder plus the existing unchanged Text-Chain/Proof
adjustment at k1. Existing GNN/SAGE results come from the frozen
`production_generator_d_v2_hard_pair_k_sensitivity_merged` artifact. Cross-encoder predictions
and complete candidate orderings are frozen before DEV gold fields are joined.

The architecture passes only if both matched comparisons satisfy all conditions:

1. cross-encoder minus existing GNN macro support F1 is at least +0.03, and
   cross-encoder+symbolic minus existing SAGE-final macro support F1 is at least
   +0.03;
2. each new method improves F1 on at least 6 of 10 datasets; and
3. each new method has no material regression (F1 decrease greater than 0.02)
   on at least 8 of 10 datasets.

Here the criterion's macro means the unweighted mean of per-example support
precision, recall, and F1 over the full pooled DEV cohort. It is therefore
dataset-size weighted; it is not an equal-weight mean of the ten dataset-level
means or a count-level micro F1. Dataset breadth is assessed on each dataset's
own per-example macro F1. No second variation is permitted after DEV.

## Adaptive aggregation exclusion

The existing frozen adaptive policies are ranking-method-specific and standardize
continuous features that include raw/adjusted GNN scores and their gaps. A
cross-encoder logit has a different uncalibrated scale. Reusing those policies is
therefore not a scale-valid application, while refitting is forbidden here. No
cross-encoder adaptive result will be produced in this experiment.
