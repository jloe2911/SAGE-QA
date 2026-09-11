import re
from collections import defaultdict
from typing import Dict, List, Tuple


def infer_hop(example_id: str, row: Dict | None = None) -> str:
    """
    Infer hop setting from explicit row fields or from example_id.
    """
    row = row or {}

    for key in ["hop", "Hop", "hop_type"]:
        if key in row and row[key]:
            return str(row[key]).lower()

    text = str(example_id).lower()

    if "2hop" in text or "2-hop" in text:
        return "2hop"

    if "1hop" in text or "1-hop" in text:
        return "1hop"

    return "unknown_hop"


def infer_answer_type(example_id: str, row: Dict | None = None) -> str:
    """
    Normalize answer type into BIN or OPEN.
    """
    row = row or {}

    for key in ["answer_type", "Answer Type", "AnswerType"]:
        if key in row and row[key]:
            raw = str(row[key]).upper()
            if raw in {"BIN", "BOOLEAN", "BOOL", "ASK"}:
                return "BIN"
            if raw in {"MC", "OPEN", "OPEN_ENDED", "SELECT"}:
                return "OPEN"

    text = str(example_id).upper()

    if "-BIN" in text or "__BIN" in text:
        return "BIN"

    if "-MC" in text or "__MC" in text or "SELECT ?X" in text:
        return "OPEN"

    return "UNKNOWN_ANSWER_TYPE"


def infer_dataset_name(example_id: str, row: Dict | None = None) -> str:
    """
    Infer dataset name. Keeps this intentionally simple.
    """
    row = row or {}

    for key in ["dataset", "Dataset", "source_dataset"]:
        if key in row and row[key]:
            return str(row[key])

    text = str(example_id)

    if "__" in text:
        return text.split("__", 1)[0]

    return "unknown_dataset"


def split_key(row: Dict) -> Tuple[str, str, str]:
    """
    Returns:
      (dataset, hop, answer_type)
    """
    example_id = row.get("example_id", "")

    dataset = infer_dataset_name(example_id, row)
    hop = infer_hop(example_id, row)
    answer_type = infer_answer_type(example_id, row)

    return dataset, hop, answer_type


def split_name(row: Dict) -> str:
    dataset, hop, answer_type = split_key(row)
    return f"{dataset} | {hop} | {answer_type}"


def group_details_by_split(details: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)

    for row in details:
        grouped[split_name(row)].append(row)

    return dict(grouped)


def average_metric(rows: List[Dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(r.get(key, 0.0)) for r in rows) / len(rows)


def binary_rate(rows: List[Dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if bool(r.get(key, False))) / len(rows)


def support_metrics_from_details(details: List[Dict]) -> Dict:
    """
    Computes support-subgraph metrics from detailed top-k rows.

    Expects fields produced by eval_gnn_subgraph_retriever.py:
      top1_label
      top1_exact_match_any_gold
      top1_contains_any_gold_explanation
      top1_best_jaccard_to_gold
      top1_best_set_f1_to_gold
      top1_best_precision_to_gold
      top1_best_recall_to_gold
      top5
    """
    n = max(len(details), 1)

    hit1 = sum(1 for d in details if int(d.get("top1_label", 0)) == 1) / n
    exact1 = sum(1 for d in details if bool(d.get("top1_exact_match_any_gold", False))) / n
    contains1 = (
        sum(1 for d in details if bool(d.get("top1_contains_any_gold_explanation", False))) / n
    )

    best_jaccard1 = average_metric(details, "top1_best_jaccard_to_gold")
    best_f11 = average_metric(details, "top1_best_set_f1_to_gold")
    best_precision1 = average_metric(details, "top1_best_precision_to_gold")
    best_recall1 = average_metric(details, "top1_best_recall_to_gold")

    hit3 = 0
    exact3 = 0
    contains3 = 0
    best_f13 = 0.0
    best_jaccard3 = 0.0

    hit5 = 0
    exact5 = 0
    contains5 = 0
    best_f15 = 0.0
    best_jaccard5 = 0.0

    for d in details:
        top5 = d.get("top5", [])

        top3 = top5[:3]

        hit3 += int(any(int(r.get("label", 0)) == 1 for r in top3))
        exact3 += int(any(bool(r.get("exact_match_any_gold", False)) for r in top3))
        contains3 += int(any(bool(r.get("contains_any_gold_explanation", False)) for r in top3))

        hit5 += int(any(int(r.get("label", 0)) == 1 for r in top5))
        exact5 += int(any(bool(r.get("exact_match_any_gold", False)) for r in top5))
        contains5 += int(any(bool(r.get("contains_any_gold_explanation", False)) for r in top5))

        if top3:
            best3 = max(top3, key=lambda r: float(r.get("best_set_f1_to_gold", 0.0)))
            best_f13 += float(best3.get("best_set_f1_to_gold", 0.0))
            best_jaccard3 += float(best3.get("best_jaccard_to_gold", 0.0))

        if top5:
            best5 = max(top5, key=lambda r: float(r.get("best_set_f1_to_gold", 0.0)))
            best_f15 += float(best5.get("best_set_f1_to_gold", 0.0))
            best_jaccard5 += float(best5.get("best_jaccard_to_gold", 0.0))

    return {
        "examples": len(details),
        "hit@1": hit1,
        "exact_hit@1": exact1,
        "contains_gold_hit@1": contains1,
        "best_jaccard@1": best_jaccard1,
        "best_set_f1@1": best_f11,
        "best_precision@1": best_precision1,
        "best_recall@1": best_recall1,
        "hit@3": hit3 / n,
        "exact_hit@3": exact3 / n,
        "contains_gold_hit@3": contains3 / n,
        "best_jaccard@3": best_jaccard3 / n,
        "best_set_f1@3": best_f13 / n,
        "hit@5": hit5 / n,
        "exact_hit@5": exact5 / n,
        "contains_gold_hit@5": contains5 / n,
        "best_jaccard@5": best_jaccard5 / n,
        "best_set_f1@5": best_f15 / n,
    }


def split_support_metrics(details: List[Dict]) -> Dict[str, Dict]:
    grouped = group_details_by_split(details)

    out = {}
    for name, rows in grouped.items():
        out[name] = support_metrics_from_details(rows)

    return out
