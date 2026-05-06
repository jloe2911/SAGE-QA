import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

import re

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.gnn_subgraph_retriever import (
    GNNSubgraphRetriever,
    build_graph_inputs_for_example,
    compute_subgraph_symbolic_features,
)
from utils.tokenizer import load_tokenizer
from utils.eval_splits import infer_dataset_name, infer_hop, infer_answer_type


# =========================================================
# Reproducibility
# =========================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# =========================================================
# IO
# =========================================================


def row_matches_source(row: Dict, source_name: str | None) -> bool:
    if not source_name:
        return True

    candidates = [
        row.get("source_name"),
        row.get("dataset"),
        row.get("Dataset"),
        row.get("source_dataset"),
    ]
    example_id = str(row.get("example_id", ""))

    return source_name in candidates or example_id.startswith(f"{source_name}__")


def load_jsonl(path: str, source_name: str | None = None) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row_matches_source(row, source_name):
                rows.append(row)
    return rows


def group_rows_by_example(rows: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["example_id"]].append(row)
    return dict(grouped)


# =========================================================
# Candidate graph reconstruction
# =========================================================


def reconstruct_candidate_axioms(example_rows: List[Dict]) -> List[str]:
    """
    Reconstruct the local candidate axiom set for one example.

    The subgraph retrieval file stores candidate subgraphs as rows.
    We recover the full local graph as the union of all units appearing
    in candidate subgraphs.
    """
    axioms = []
    seen = set()

    for row in example_rows:
        for unit in row.get("subgraph_units", []):
            if unit not in seen:
                axioms.append(unit)
                seen.add(unit)

    return axioms


def subgraph_units_to_node_ids(
    subgraph_units: List[str],
    axiom_to_idx: Dict[str, int],
) -> List[int]:
    node_ids = []

    for unit in subgraph_units:
        if unit in axiom_to_idx:
            node_ids.append(axiom_to_idx[unit])

    return node_ids


# =========================================================
# Ranking target
# =========================================================


def ranking_target(row: Dict) -> float:
    """
    Strict support-ranking target.

    Exact support should be best.
    Supersets that contain a full explanation are sufficient but less compact.
    Partial overlap is useful but must not compete with complete support.
    """
    if row.get("exact_match_any_gold", False):
        return 1.0

    if row.get("contains_any_gold_explanation", False):
        return 0.9

    partial_f1 = float(row.get("best_set_f1_to_gold", 0.0))

    # Important:
    # Cap partial support so single facts do not compete with full explanations.
    if partial_f1 > 0:
        return min(0.3, 0.5 * partial_f1)

    return 0.0


def binary_label_from_target(target: float) -> int:
    """
    Auxiliary binary target.
    Positive means the subgraph is sufficient support.
    """
    return 1 if target >= 0.9 else 0


# =========================================================
# Dataset preparation
# =========================================================


def prepare_examples(rows: List[Dict]) -> List[Dict]:
    """
    Converts flat candidate-subgraph rows into example-level graph objects.

    Each prepared example contains:
      - candidate_axioms: full local evidence graph nodes
      - candidate_rows: candidate support subgraphs to rank
    """
    by_example = group_rows_by_example(rows)
    examples = []

    for example_id, ex_rows in by_example.items():
        first = ex_rows[0]
        question = (
            first.get("question")
            or first.get("abs_question")
            or first.get("task_id")
            or first.get("sparql_query")
            or example_id
        )
        question = str(question)
        sparql_query = str(first.get("sparql_query") or "")
        dataset = infer_dataset_name(example_id, first)
        hop = infer_hop(example_id, first)
        answer_type = infer_answer_type(example_id, first)

        candidate_axioms = reconstruct_candidate_axioms(ex_rows)
        axiom_to_idx = {ax: i for i, ax in enumerate(candidate_axioms)}

        candidate_rows = []

        for row in ex_rows:
            subgraph_units = row.get("subgraph_units", [])
            node_ids = subgraph_units_to_node_ids(subgraph_units, axiom_to_idx)

            if not node_ids:
                continue

            target = ranking_target(row)
            binary_label = binary_label_from_target(target)

            symbolic_features = compute_subgraph_symbolic_features(
                question=str(row.get("question") or question),
                sparql_query=str(row.get("sparql_query") or sparql_query),
                subgraph_units=subgraph_units,
                use_gold_features=False,
                exact_match_any_gold=row.get("exact_match_any_gold", False),
                contains_any_gold_explanation=row.get(
                    "contains_any_gold_explanation", False
                ),
            )

            candidate_rows.append(
                {
                    "example_id": example_id,
                    "question": str(row.get("question") or question),
                    "sparql_query": str(row.get("sparql_query") or sparql_query),
                    "subgraph_units": subgraph_units,
                    "subgraph_node_ids": node_ids,
                    "subgraph_size": len(subgraph_units),
                    "symbolic_features": symbolic_features,
                    # Ranking supervision
                    "rank_target": target,
                    "label": binary_label,
                    # Original/debug fields
                    "gold_units": row.get("gold_units", []),
                    "gold_explanations": row.get("gold_explanations", []),
                    "best_matching_gold_explanation": row.get(
                        "best_matching_gold_explanation", []
                    ),
                    "best_matching_gold_index": row.get("best_matching_gold_index", -1),
                    "best_jaccard_to_gold": float(row.get("best_jaccard_to_gold", 0.0)),
                    "best_set_f1_to_gold": float(row.get("best_set_f1_to_gold", 0.0)),
                    "best_set_precision_to_gold": float(
                        row.get("best_set_precision_to_gold", 0.0)
                    ),
                    "best_set_recall_to_gold": float(
                        row.get("best_set_recall_to_gold", 0.0)
                    ),
                    "exact_match_any_gold": bool(
                        row.get("exact_match_any_gold", False)
                    ),
                    "contains_any_gold_explanation": bool(
                        row.get("contains_any_gold_explanation", False)
                    ),
                    "dataset": dataset,
                    "hop": hop,
                    "answer_type": answer_type,
                    "task_type": row.get("task_type", row.get("Task Type", "")),
                }
            )

        candidate_rows = subsample_candidate_rows(
            candidate_rows,
            max_pos=64,
            max_hard_neg=128,
            max_easy_neg=128,
        )

        if not candidate_rows:
            continue

        # We need at least two different target values for ranking.
        unique_targets = {round(r["rank_target"], 6) for r in candidate_rows}

        examples.append(
            {
                "example_id": example_id,
                "dataset": dataset,
                "hop": hop,
                "answer_type": answer_type,
                "question": question,
                "sparql_query": sparql_query,
                "candidate_axioms": candidate_axioms,
                "candidate_rows": candidate_rows,
                "has_rankable_pairs": len(unique_targets) >= 2,
            }
        )

    return examples


# =========================================================
# Model scoring helpers
# =========================================================


def adjusted_score(
    row: Dict,
    score_mode: str = "neural",
    size_penalty: float = 0.01,
) -> float:
    """
    Ranking-time score.

    For GNN, default should be neural while the ranker is learning completeness.
    Minimality adjustment can be tested after the GNN starts retrieving complete supports.
    """
    if score_mode == "minimality_adjusted":
        return row["score"] - size_penalty * row["subgraph_size"]

    if score_mode == "completeness_adjusted":
        feats = row.get("symbolic_features", [])
        fact_rule_mix = feats[4] if len(feats) > 4 else 0.0
        has_query_property_rule = feats[2] if len(feats) > 2 else 0.0
        oversize_penalty = max(0, row["subgraph_size"] - 2)

        return (
            row["score"]
            + 0.02 * fact_rule_mix
            + 0.02 * has_query_property_rule
            - 0.005 * oversize_penalty
        )

    return row["score"]


def encode_example_graph(
    model: GNNSubgraphRetriever,
    tokenizer,
    example: Dict,
    device,
    max_length: int = 128,
):
    """
    Encodes one local evidence graph.
    """
    graph_inputs = build_graph_inputs_for_example(
        candidate_axioms=example["candidate_axioms"],
        question=example["question"],
        sparql_query=example.get("sparql_query", ""),
        tokenizer=tokenizer,
        model_device=device,
        max_length=max_length,
    )

    query_embedding = model.encode_texts(
        input_ids=graph_inputs["query_input_ids"],
        attention_mask=graph_inputs["query_attention_mask"],
        token_type_ids=graph_inputs["query_token_type_ids"],
    ).squeeze(0)

    node_text_embeddings = model.encode_texts(
        input_ids=graph_inputs["node_input_ids"],
        attention_mask=graph_inputs["node_attention_mask"],
        token_type_ids=graph_inputs["node_token_type_ids"],
    )

    return {
        "query_embedding": query_embedding,
        "node_text_embeddings": node_text_embeddings,
        "node_symbolic_features": graph_inputs["node_symbolic_features"],
        "edge_index": graph_inputs["edge_index"],
    }


def score_candidate_rows(
    model: GNNSubgraphRetriever,
    encoded_graph: Dict,
    candidate_rows: List[Dict],
    device,
):
    query_embeddings = []
    node_text_embeddings = []
    node_symbolic_features = []
    edge_indices = []
    subgraph_node_ids = []
    subgraph_symbolic_features = []

    for row in candidate_rows:
        query_embeddings.append(encoded_graph["query_embedding"])
        node_text_embeddings.append(encoded_graph["node_text_embeddings"])
        node_symbolic_features.append(encoded_graph["node_symbolic_features"])
        edge_indices.append(encoded_graph["edge_index"])

        subgraph_node_ids.append(
            torch.tensor(row["subgraph_node_ids"], dtype=torch.long, device=device)
        )

        subgraph_symbolic_features.append(
            torch.tensor(row["symbolic_features"], dtype=torch.float, device=device)
        )

    return model.forward_batch_graphs(
        query_embeddings=query_embeddings,
        node_text_embeddings=node_text_embeddings,
        node_symbolic_features=node_symbolic_features,
        edge_indices=edge_indices,
        subgraph_node_ids=subgraph_node_ids,
        subgraph_symbolic_features=subgraph_symbolic_features,
    )


def subsample_candidate_rows(
    rows,
    max_pos: int = 64,
    max_hard_neg: int = 128,
    max_easy_neg: int = 128,
):
    positives = [r for r in rows if int(r.get("label", 0)) == 1]
    hard_negatives = [
        r
        for r in rows
        if int(r.get("label", 0)) == 0
        and float(r.get("best_set_f1_to_gold", 0.0)) > 0.0
    ]
    easy_negatives = [
        r
        for r in rows
        if int(r.get("label", 0)) == 0
        and float(r.get("best_set_f1_to_gold", 0.0)) == 0.0
    ]

    positives = sorted(
        positives,
        key=lambda r: float(r.get("rank_target", 0.0)),
        reverse=True,
    )

    hard_negatives = sorted(
        hard_negatives,
        key=lambda r: float(r.get("best_set_f1_to_gold", 0.0)),
        reverse=True,
    )

    if len(positives) > max_pos:
        positives = positives[:max_pos]

    if len(hard_negatives) > max_hard_neg:
        hard_negatives = hard_negatives[:max_hard_neg]

    if len(easy_negatives) > max_easy_neg:
        easy_negatives = random.sample(easy_negatives, max_easy_neg)

    sampled = positives + hard_negatives + easy_negatives
    random.shuffle(sampled)
    return sampled


# =========================================================
# Ranking loss
# =========================================================


def pairwise_ranking_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.2,
    max_pairs: int = 512,
) -> torch.Tensor:
    """
    Focused within-question ranking loss.

    Prioritizes:
      complete supports > partial supports
      complete supports > irrelevant supports
      exact supports > sufficient supersets
    """
    device = scores.device

    exact = []
    sufficient = []
    partial = []
    irrelevant = []

    for i, t in enumerate(targets.detach().cpu().tolist()):
        if t >= 0.999:
            exact.append(i)
        elif t >= 0.899:
            sufficient.append(i)
        elif t > 0.0:
            partial.append(i)
        else:
            irrelevant.append(i)

    weighted_pairs = []

    # Exact should outrank sufficient supersets.
    for i in exact:
        for j in sufficient:
            weighted_pairs.append((i, j, 2.0))

    # Exact and sufficient supports should strongly outrank partial supports.
    for i in exact + sufficient:
        for j in partial:
            weighted_pairs.append((i, j, 3.0))

    # Exact and sufficient supports should outrank irrelevant subgraphs.
    for i in exact + sufficient:
        for j in irrelevant:
            weighted_pairs.append((i, j, 2.0))

    # Partial supports should weakly outrank irrelevant subgraphs.
    for i in partial:
        for j in irrelevant:
            weighted_pairs.append((i, j, 0.5))

    if not weighted_pairs:
        return torch.tensor(0.0, device=device)

    if len(weighted_pairs) > max_pairs:
        weighted_pairs = random.sample(weighted_pairs, max_pairs)

    better = torch.tensor(
        [p[0] for p in weighted_pairs], dtype=torch.long, device=device
    )
    worse = torch.tensor(
        [p[1] for p in weighted_pairs], dtype=torch.long, device=device
    )
    weights = torch.tensor(
        [p[2] for p in weighted_pairs], dtype=torch.float, device=device
    )

    losses = F.relu(margin - scores[better] + scores[worse])
    return (losses * weights).mean()


def listwise_soft_target_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Optional listwise loss.

    Converts graded targets into a soft distribution and encourages
    the model score distribution to match it.

    This helps when many candidates have similar pairwise relationships.
    """
    if targets.max() <= 0:
        return torch.tensor(0.0, device=scores.device)

    target_dist = F.softmax(targets / temperature, dim=0)
    score_log_dist = F.log_softmax(scores, dim=0)
    return F.kl_div(score_log_dist, target_dist, reduction="batchmean")


def compute_example_loss(
    scores: torch.Tensor,
    candidate_rows: List[Dict],
    bce_criterion,
    ranking_margin: float,
    ranking_weight: float,
    bce_weight: float,
    listwise_weight: float,
    max_pairs: int,
) -> Tuple[torch.Tensor, Dict]:
    device = scores.device

    targets = torch.tensor(
        [float(r["rank_target"]) for r in candidate_rows],
        dtype=torch.float,
        device=device,
    )

    binary_labels = torch.tensor(
        [float(r["label"]) for r in candidate_rows],
        dtype=torch.float,
        device=device,
    )

    ranking = pairwise_ranking_loss(
        scores=scores,
        targets=targets,
        margin=ranking_margin,
        max_pairs=max_pairs,
    )

    bce = bce_criterion(scores, binary_labels)

    listwise = listwise_soft_target_loss(
        scores=scores,
        targets=targets,
        temperature=0.1,
    )

    total = ranking_weight * ranking + bce_weight * bce + listwise_weight * listwise

    stats = {
        "ranking_loss": float(ranking.detach().cpu()),
        "bce_loss": float(bce.detach().cpu()),
        "listwise_loss": float(listwise.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }

    return total, stats


def _unit_text(row: Dict[str, Any]) -> str:
    return " ".join(str(u) for u in row.get("subgraph_units", []))


def _extract_uri_fragments(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"#([^>]+)>", str(text))


def extract_query_properties(sparql_query: str) -> Set[str]:
    props = set()
    for frag in _extract_uri_fragments(sparql_query):
        if frag.startswith("has") or frag.startswith("is"):
            props.add(frag)
    return props


def extract_query_entities(sparql_query: str) -> Set[str]:
    ents = set()
    for frag in _extract_uri_fragments(sparql_query):
        if not (frag.startswith("has") or frag.startswith("is")):
            ents.add(frag)
    return ents


def is_schema_unit(unit: str) -> bool:
    unit = str(unit)
    markers = [
        "SymmetricObjectProperty",
        "TransitiveObjectProperty",
        "SubObjectPropertyOf",
        "InverseObjectProperties",
        "PropertyChain",
        "FunctionalObjectProperty",
        "domain",
        "range",
        "subPropertyOf",
        "inverseOf",
    ]
    return any(m in unit for m in markers)


def is_fact_unit(unit: str) -> bool:
    return not is_schema_unit(unit)


def query_property_match(row: Dict[str, Any]) -> float:
    sparql = row.get("sparql_query", "") or row.get("SPARQL Query", "")
    props = extract_query_properties(sparql)
    text = _unit_text(row)

    if not props:
        return 0.0

    return 1.0 if any(p in text for p in props) else 0.0


def query_entity_coverage(row: Dict[str, Any]) -> float:
    sparql = row.get("sparql_query", "") or row.get("SPARQL Query", "")
    ents = extract_query_entities(sparql)
    text = _unit_text(row)

    if not ents:
        return 0.0

    return sum(1 for e in ents if e in text) / max(len(ents), 1)


def fact_rule_mix_symbolic(row: Dict[str, Any]) -> float:
    units = row.get("subgraph_units", [])
    has_fact = any(is_fact_unit(u) for u in units)
    has_rule = any(is_schema_unit(u) for u in units)
    return 1.0 if has_fact and has_rule else 0.0


def schema_count(row: Dict[str, Any]) -> int:
    return sum(1 for u in row.get("subgraph_units", []) if is_schema_unit(u))


def subgraph_size(row: Dict[str, Any]) -> int:
    return int(row.get("subgraph_size", len(row.get("subgraph_units", []))))


def nesyqa_compact_adjustment(
    row: Dict[str, Any],
    property_bonus: float = 0.025,
    entity_bonus: float = 0.015,
    fact_rule_bonus: float = 0.020,
    size_penalty: float = 0.006,
    extra_schema_penalty: float = 0.004,
) -> float:
    """
    Gold-free symbolic adjustment used by the full NeSyQA reranker.

    Rewards:
      - query-property match
      - query-entity coverage
      - fact + schema/rule mixture

    Penalizes:
      - oversized candidate supports
      - multiple schema axioms that often create noisy supersets

    Important:
    This does not use gold-derived evaluation fields.
    """
    size = subgraph_size(row)
    n_schema = schema_count(row)

    oversize = max(0, size - 2)
    extra_schema = max(0, n_schema - 1)

    return (
        property_bonus * query_property_match(row)
        + entity_bonus * query_entity_coverage(row)
        + fact_rule_bonus * fact_rule_mix_symbolic(row)
        - size_penalty * oversize
        - extra_schema_penalty * extra_schema
    )


def compute_adjusted_score(
    row: Dict[str, Any],
    neural_score: float,
    score_mode: str = "neural",
    size_penalty: float = 0.01,
) -> float:
    """
    Final score used for ranking candidate support subgraphs.
    """
    size = subgraph_size(row)

    if score_mode == "neural":
        return float(neural_score)

    if score_mode == "minimality_adjusted":
        return float(neural_score) - size_penalty * size

    if score_mode == "completeness_adjusted":
        feats = row.get("symbolic_features", [])

        # Existing symbolic features from dataset construction.
        # Keep this mode backward-compatible.
        has_query_property_rule = feats[2] if len(feats) > 2 else 0.0
        fact_rule_mix_value = feats[4] if len(feats) > 4 else 0.0
        oversize_penalty = max(0, size - 2)

        return (
            float(neural_score)
            + 0.02 * float(fact_rule_mix_value)
            + 0.02 * float(has_query_property_rule)
            - 0.005 * float(oversize_penalty)
        )

    if score_mode == "nesyqa_compact":
        return float(neural_score) + nesyqa_compact_adjustment(row)

    raise ValueError(f"Unknown score_mode: {score_mode}")


# =========================================================
# Evaluation
# =========================================================


def evaluate(
    model: GNNSubgraphRetriever,
    tokenizer,
    examples: List[Dict],
    device,
    bce_criterion,
    max_length: int = 128,
    candidate_batch_size: int = 64,
    score_mode: str = "neural",
    size_penalty: float = 0.01,
):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    hit1 = 0
    exact_hit1 = 0
    contains_hit1 = 0

    hit3 = 0
    exact_hit3 = 0
    contains_hit3 = 0

    hit5 = 0
    exact_hit5 = 0
    contains_hit5 = 0

    total_examples = 0

    top1_jaccard = 0.0
    top1_f1 = 0.0
    top1_precision = 0.0
    top1_recall = 0.0

    best_jaccard3 = 0.0
    best_f13 = 0.0
    best_precision3 = 0.0
    best_recall3 = 0.0

    best_jaccard5 = 0.0
    best_f15 = 0.0
    best_precision5 = 0.0
    best_recall5 = 0.0

    details = []

    with torch.no_grad():
        for example in examples:
            encoded_graph = encode_example_graph(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                max_length=max_length,
            )

            scored_rows = []
            candidate_rows = example["candidate_rows"]

            for start in range(0, len(candidate_rows), candidate_batch_size):
                batch_rows = candidate_rows[start : start + candidate_batch_size]

                out = score_candidate_rows(
                    model=model,
                    encoded_graph=encoded_graph,
                    candidate_rows=batch_rows,
                    device=device,
                )

                labels = torch.tensor(
                    [float(r["label"]) for r in batch_rows],
                    dtype=torch.float,
                    device=device,
                )

                loss = bce_criterion(out["logits"], labels)
                total_loss += float(loss.item())
                total_batches += 1

                probs = out["probs"].detach().cpu().tolist()

                for row, prob in zip(batch_rows, probs):
                    row = dict(row)
                    neural_score = float(prob)
                    row["score"] = neural_score
                    row["adjusted_score"] = compute_adjusted_score(
                        row=row,
                        neural_score=neural_score,
                        score_mode=score_mode,
                        size_penalty=size_penalty,
                    )
                    scored_rows.append(row)

            if not scored_rows:
                continue

            scored_rows = sorted(
                scored_rows,
                key=lambda r: r["adjusted_score"],
                reverse=True,
            )

            top1 = scored_rows[0]
            top3 = scored_rows[:3]
            top5 = scored_rows[:5]

            total_examples += 1

            hit1 += int(top1["label"] == 1)
            exact_hit1 += int(top1["exact_match_any_gold"])
            contains_hit1 += int(top1["contains_any_gold_explanation"])

            hit3 += int(any(r["label"] == 1 for r in top3))
            exact_hit3 += int(any(r["exact_match_any_gold"] for r in top3))
            contains_hit3 += int(any(r["contains_any_gold_explanation"] for r in top3))

            hit5 += int(any(r["label"] == 1 for r in top5))
            exact_hit5 += int(any(r["exact_match_any_gold"] for r in top5))
            contains_hit5 += int(any(r["contains_any_gold_explanation"] for r in top5))

            top1_jaccard += top1["best_jaccard_to_gold"]
            top1_f1 += top1["best_set_f1_to_gold"]
            top1_precision += top1["best_set_precision_to_gold"]
            top1_recall += top1["best_set_recall_to_gold"]

            best3 = max(top3, key=lambda r: r["best_set_f1_to_gold"])
            best5 = max(top5, key=lambda r: r["best_set_f1_to_gold"])

            best_jaccard3 += best3["best_jaccard_to_gold"]
            best_f13 += best3["best_set_f1_to_gold"]
            best_precision3 += best3["best_set_precision_to_gold"]
            best_recall3 += best3["best_set_recall_to_gold"]

            best_jaccard5 += best5["best_jaccard_to_gold"]
            best_f15 += best5["best_set_f1_to_gold"]
            best_precision5 += best5["best_set_precision_to_gold"]
            best_recall5 += best5["best_set_recall_to_gold"]

            details.append(
                {
                    "example_id": example["example_id"],
                    "question": example["question"],
                    "top1_subgraph_units": top1["subgraph_units"],
                    "top1_score": top1["score"],
                    "top1_adjusted_score": top1["adjusted_score"],
                    "top1_rank_target": top1["rank_target"],
                    "top1_label": top1["label"],
                    "top1_best_jaccard_to_gold": top1["best_jaccard_to_gold"],
                    "top1_best_set_f1_to_gold": top1["best_set_f1_to_gold"],
                    "top1_best_precision_to_gold": top1["best_set_precision_to_gold"],
                    "top1_best_recall_to_gold": top1["best_set_recall_to_gold"],
                    "top1_exact_match_any_gold": top1["exact_match_any_gold"],
                    "top1_contains_any_gold_explanation": top1[
                        "contains_any_gold_explanation"
                    ],
                    "top1_best_matching_gold_explanation": top1[
                        "best_matching_gold_explanation"
                    ],
                    "gold_explanations": top1["gold_explanations"],
                    "top5": [
                        {
                            "rank": i + 1,
                            "score": r["score"],
                            "adjusted_score": r["adjusted_score"],
                            "rank_target": r["rank_target"],
                            "label": r["label"],
                            "subgraph_size": r["subgraph_size"],
                            "subgraph_units": r["subgraph_units"],
                            "best_set_f1_to_gold": r["best_set_f1_to_gold"],
                            "best_jaccard_to_gold": r["best_jaccard_to_gold"],
                            "contains_any_gold_explanation": r[
                                "contains_any_gold_explanation"
                            ],
                            "exact_match_any_gold": r["exact_match_any_gold"],
                        }
                        for i, r in enumerate(top5)
                    ],
                    "dataset": example.get("dataset", ""),
                    "hop": example.get("hop", ""),
                    "answer_type": example.get("answer_type", ""),
                }
            )

    n = max(total_examples, 1)

    metrics = {
        "examples": total_examples,
        "loss": total_loss / max(total_batches, 1),
        "score_mode": score_mode,
        "size_penalty": size_penalty,
        "hit@1": hit1 / n,
        "exact_hit@1": exact_hit1 / n,
        "contains_gold_hit@1": contains_hit1 / n,
        "best_jaccard@1": top1_jaccard / n,
        "best_set_f1@1": top1_f1 / n,
        "best_precision@1": top1_precision / n,
        "best_recall@1": top1_recall / n,
        "hit@3": hit3 / n,
        "exact_hit@3": exact_hit3 / n,
        "contains_gold_hit@3": contains_hit3 / n,
        "best_jaccard@3": best_jaccard3 / n,
        "best_set_f1@3": best_f13 / n,
        "best_precision@3": best_precision3 / n,
        "best_recall@3": best_recall3 / n,
        "hit@5": hit5 / n,
        "exact_hit@5": exact_hit5 / n,
        "contains_gold_hit@5": contains_hit5 / n,
        "best_jaccard@5": best_jaccard5 / n,
        "best_set_f1@5": best_f15 / n,
        "best_precision@5": best_precision5 / n,
        "best_recall@5": best_recall5 / n,
    }

    return metrics, details


# =========================================================
# Training
# =========================================================


def train(
    train_path: str,
    dev_path: str,
    save_dir: str,
    model_name: str = "google/bert_uncased_L-2_H-128_A-2",
    lr: float = 2e-5,
    epochs: int = 5,
    max_length: int = 128,
    candidate_batch_size: int = 256,
    max_train_examples: int = 0,
    max_dev_examples: int = 0,
    freeze_encoder: bool = False,
    ranking_margin: float = 0.1,
    ranking_weight: float = 1.0,
    bce_weight: float = 0.2,
    listwise_weight: float = 0.2,
    max_pairs: int = 512,
    score_mode: str = "neural",
    size_penalty: float = 0.01,
    source_name: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_rows = load_jsonl(train_path, source_name=source_name)
    dev_rows = load_jsonl(dev_path, source_name=source_name)

    train_examples = prepare_examples(train_rows)
    dev_examples = prepare_examples(dev_rows)

    # Only rankable examples can contribute ranking loss.
    # Keep all examples for BCE/listwise, but report rankable count.
    train_rankable = sum(int(ex["has_rankable_pairs"]) for ex in train_examples)
    dev_rankable = sum(int(ex["has_rankable_pairs"]) for ex in dev_examples)

    if max_train_examples and max_train_examples > 0:
        train_examples = train_examples[:max_train_examples]

    if max_dev_examples and max_dev_examples > 0:
        dev_examples = dev_examples[:max_dev_examples]

    print(f"Loaded train subgraph rows: {len(train_rows)}")
    print(f"Loaded dev subgraph rows:   {len(dev_rows)}")
    print(f"Prepared train examples:    {len(train_examples)}")
    print(f"Prepared dev examples:      {len(dev_examples)}")
    print(f"Rankable train examples:    {train_rankable}")
    print(f"Rankable dev examples:      {dev_rankable}")

    if not train_examples:
        raise ValueError("No train examples available.")
    if not dev_examples:
        raise ValueError("No dev examples available.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = GNNSubgraphRetriever(
        model_name=model_name,
        node_symbolic_dim=8,
        subgraph_symbolic_dim=8,
        gnn_hidden_dim=128,
        gnn_layers=2,
        classifier_hidden_dim=128,
        dropout=0.1,
        freeze_encoder=freeze_encoder,
    ).to(device)

    # Auxiliary BCE pos_weight
    all_labels = []
    for ex in train_examples:
        for row in ex["candidate_rows"]:
            all_labels.append(int(row["label"]))

    pos = sum(all_labels)
    neg = len(all_labels) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float, device=device)

    print(f"Auxiliary positive rows: {pos}")
    print(f"Auxiliary negative rows: {neg}")
    print(f"Using auxiliary BCE pos_weight={pos_weight.item():.4f}")
    print(f"Ranking margin={ranking_margin}")
    print(
        f"Loss weights: ranking={ranking_weight}, bce={bce_weight}, listwise={listwise_weight}"
    )

    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    total_steps = max(1, len(train_examples) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_dev_metric = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_examples)

        running_total_loss = 0.0
        running_ranking_loss = 0.0
        running_bce_loss = 0.0
        running_listwise_loss = 0.0
        update_count = 0

        pbar = tqdm(train_examples, desc=f"Epoch {epoch}")

        for example in pbar:
            encoded_graph = encode_example_graph(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                max_length=max_length,
            )

            candidate_rows = example["candidate_rows"]

            optimizer.zero_grad()

            all_logits = []
            all_batch_rows = []

            for start in range(0, len(candidate_rows), candidate_batch_size):
                batch_rows = candidate_rows[start : start + candidate_batch_size]

                out = score_candidate_rows(
                    model=model,
                    encoded_graph=encoded_graph,
                    candidate_rows=batch_rows,
                    device=device,
                )

                all_logits.append(out["logits"])
                all_batch_rows.extend(batch_rows)

            scores = torch.cat(all_logits, dim=0)

            loss, loss_stats = compute_example_loss(
                scores=scores,
                candidate_rows=all_batch_rows,
                bce_criterion=bce_criterion,
                ranking_margin=ranking_margin,
                ranking_weight=ranking_weight,
                bce_weight=bce_weight,
                listwise_weight=listwise_weight,
                max_pairs=max_pairs,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            running_total_loss += loss_stats["total_loss"]
            running_ranking_loss += loss_stats["ranking_loss"]
            running_bce_loss += loss_stats["bce_loss"]
            running_listwise_loss += loss_stats["listwise_loss"]
            update_count += 1

            pbar.set_postfix(
                total=loss_stats["total_loss"],
                rank=loss_stats["ranking_loss"],
                bce=loss_stats["bce_loss"],
                listwise=loss_stats["listwise_loss"],
            )

        train_total_loss = running_total_loss / max(update_count, 1)
        train_ranking_loss = running_ranking_loss / max(update_count, 1)
        train_bce_loss = running_bce_loss / max(update_count, 1)
        train_listwise_loss = running_listwise_loss / max(update_count, 1)

        dev_metrics, dev_details = evaluate(
            model=model,
            tokenizer=tokenizer,
            examples=dev_examples,
            device=device,
            bce_criterion=bce_criterion,
            max_length=max_length,
            candidate_batch_size=candidate_batch_size,
            score_mode=score_mode,
            size_penalty=size_penalty,
        )

        print(
            f"Epoch {epoch} | "
            f"train_total={train_total_loss:.4f} | "
            f"train_rank={train_ranking_loss:.4f} | "
            f"train_bce={train_bce_loss:.4f} | "
            f"train_listwise={train_listwise_loss:.4f} | "
            f"dev_loss={dev_metrics['loss']:.4f} | "
            f"dev_hit@1={dev_metrics['hit@1']:.4f} | "
            f"dev_exact@1={dev_metrics['exact_hit@1']:.4f} | "
            f"dev_contains@1={dev_metrics['contains_gold_hit@1']:.4f} | "
            f"dev_f1@1={dev_metrics['best_set_f1@1']:.4f}"
        )

        # Choose by dev top-1 F1, because this measures support quality.
        selection_metric = dev_metrics["best_set_f1@1"]

        if selection_metric > best_dev_metric:
            best_dev_metric = selection_metric

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "node_symbolic_dim": 8,
                "subgraph_symbolic_dim": 8,
                "gnn_hidden_dim": 128,
                "gnn_layers": 2,
                "classifier_hidden_dim": 128,
                "freeze_encoder": freeze_encoder,
                "training_objective": "within_question_pairwise_ranking",
                "ranking_margin": ranking_margin,
                "ranking_weight": ranking_weight,
                "bce_weight": bce_weight,
                "listwise_weight": listwise_weight,
                "max_pairs": max_pairs,
                "score_mode": score_mode,
                "size_penalty": size_penalty,
                "best_dev_metrics": dev_metrics,
            }

            torch.save(checkpoint, save_dir / "best_model.pt")
            print("Saved best GNN subgraph ranker.")

            with open(save_dir / "best_dev_metrics.json", "w", encoding="utf-8") as f:
                json.dump(dev_metrics, f, indent=2)

            with open(save_dir / "best_dev_samples.json", "w", encoding="utf-8") as f:
                json.dump(dev_details[:10], f, indent=2, ensure_ascii=False)


# =========================================================
# CLI
# =========================================================


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-path", type=str, default="data/train_subgraph_retrieval.jsonl"
    )
    parser.add_argument(
        "--dev-path", type=str, default="data/dev_subgraph_retrieval.jsonl"
    )
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints/gnn_subgraph_retriever"
    )

    parser.add_argument(
        "--model-name", type=str, default="google/bert_uncased_L-2_H-128_A-2"
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--candidate-batch-size", type=int, default=256)

    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-dev-examples", type=int, default=0)

    parser.add_argument("--freeze-encoder", action="store_true")

    parser.add_argument("--ranking-margin", type=float, default=0.2)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.2)
    parser.add_argument("--listwise-weight", type=float, default=0.2)
    parser.add_argument("--max-pairs", type=int, default=512)

    # For evaluation during training.
    # For GNN, neural is safer initially.
    parser.add_argument(
        "--score-mode",
        type=str,
        default="neural",
        choices=[
            "neural",
            "minimality_adjusted",
            "completeness_adjusted",
            "nesyqa_compact",
        ],
    )
    parser.add_argument("--size-penalty", type=float, default=0.01)
    parser.add_argument("--source-name", type=str, default=None)

    parser.add_argument("--max-pos-per-example", type=int, default=64)
    parser.add_argument("--max-hard-neg-per-example", type=int, default=128)
    parser.add_argument("--max-easy-neg-per-example", type=int, default=128)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train(
        train_path=args.train_path,
        dev_path=args.dev_path,
        save_dir=args.save_dir,
        model_name=args.model_name,
        lr=args.lr,
        epochs=args.epochs,
        max_length=args.max_length,
        candidate_batch_size=args.candidate_batch_size,
        max_train_examples=args.max_train_examples,
        max_dev_examples=args.max_dev_examples,
        freeze_encoder=args.freeze_encoder,
        ranking_margin=args.ranking_margin,
        ranking_weight=args.ranking_weight,
        bce_weight=args.bce_weight,
        listwise_weight=args.listwise_weight,
        max_pairs=args.max_pairs,
        score_mode=args.score_mode,
        size_penalty=args.size_penalty,
        source_name=args.source_name,
    )
