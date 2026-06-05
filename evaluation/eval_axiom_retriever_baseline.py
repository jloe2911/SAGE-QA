import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


RANDOM_SEED = 42
random.seed(RANDOM_SEED)


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
            if line.strip():
                row = json.loads(line)
                if row_matches_source(row, source_name):
                    rows.append(row)
    return rows


def normalize_unit(unit: str) -> str:
    return str(unit).strip()


def strip_tag_units(units: List[str]) -> List[str]:
    return [normalize_unit(u) for u in units if not str(u).strip().startswith("TAG:")]


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    return text


def tokenize(text: str) -> Set[str]:
    return set(t for t in normalize_text(text).split() if t)


def get_question_text(row: Dict, input_format: str = "hybrid") -> str:
    question = (
        row.get("question") or row.get("NL Question") or row.get("nl_question") or ""
    )
    abs_question = row.get("abs_question") or row.get("ABS Question") or ""
    sparql = row.get("sparql_query") or row.get("SPARQL Query") or ""

    if input_format == "nl":
        return question

    if input_format == "abs":
        return abs_question or question

    if input_format == "sparql":
        return sparql or question

    if input_format == "hybrid":
        return f"{question} {sparql}"

    return question


def infer_answer_type(example_id: str, row: Dict) -> str:
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

    return "UNKNOWN"


def infer_hop(example_id: str, row: Dict) -> str:
    for key in ["hop", "Hop", "hop_type"]:
        if key in row and row[key]:
            return str(row[key]).lower()

    text = str(example_id).lower()
    if "2hop" in text or "2-hop" in text:
        return "2hop"
    if "1hop" in text or "1-hop" in text:
        return "1hop"

    return "UNKNOWN"


def infer_dataset(example_id: str, row: Dict) -> str:
    for key in ["dataset", "Dataset", "source_dataset"]:
        if key in row and row[key]:
            return str(row[key])

    if "__" in example_id:
        return example_id.split("__", 1)[0]

    return "UNKNOWN"


def group_by_example(rows: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["example_id"]].append(row)
    return dict(grouped)


def lexical_score(question_text: str, unit: str) -> float:
    q = tokenize(question_text)
    u = tokenize(unit)

    if not q or not u:
        return 0.0

    return len(q & u) / max(len(q | u), 1)


def random_score(_: str, __: str) -> float:
    return random.random()


def get_gold_explanations(row: Dict) -> List[List[str]]:
    """
    Expects gold_explanations if available.
    Falls back to gold_support_units/gold_units.
    """
    gold_explanations = row.get("gold_explanations", [])

    if gold_explanations:
        return [strip_tag_units(g) for g in gold_explanations if strip_tag_units(g)]

    gold_units = strip_tag_units(
        row.get("gold_support_units", []) or row.get("gold_units", [])
    )
    if gold_units:
        return [gold_units]

    return []


def get_candidate_units(rows: List[Dict]) -> List[str]:
    """
    Recover individual candidate support units from candidate subgraph rows.
    """
    seen = set()
    units = []

    for row in rows:
        for unit in row.get("subgraph_units", []):
            unit = normalize_unit(unit)
            if not unit or unit.startswith("TAG:"):
                continue
            if unit not in seen:
                seen.add(unit)
                units.append(unit)

    return units


def set_scores(pred: List[str], gold_explanations: List[List[str]]) -> Dict:
    pred_set = set(strip_tag_units(pred))

    if not gold_explanations:
        return {
            "best_jaccard": 0.0,
            "best_f1": 0.0,
            "best_precision": 0.0,
            "best_recall": 0.0,
            "exact": False,
            "contains": False,
            "best_gold": [],
        }

    best = {
        "best_jaccard": 0.0,
        "best_f1": 0.0,
        "best_precision": 0.0,
        "best_recall": 0.0,
        "exact": False,
        "contains": False,
        "best_gold": [],
    }

    for gold in gold_explanations:
        gold_set = set(strip_tag_units(gold))

        if not gold_set:
            continue

        inter = len(pred_set & gold_set)
        union = len(pred_set | gold_set)

        precision = inter / max(len(pred_set), 1)
        recall = inter / max(len(gold_set), 1)
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        jaccard = inter / max(union, 1)

        exact = pred_set == gold_set
        contains = gold_set.issubset(pred_set)

        if f1 > best["best_f1"]:
            best = {
                "best_jaccard": jaccard,
                "best_f1": f1,
                "best_precision": precision,
                "best_recall": recall,
                "exact": exact,
                "contains": contains,
                "best_gold": list(gold_set),
            }

        # Preserve exact/contains even if another gold has same/higher F1
        best["exact"] = bool(best["exact"] or exact)
        best["contains"] = bool(best["contains"] or contains)

    return best


def min_gold_size(gold_explanations: List[List[str]]) -> int:
    sizes = [len(strip_tag_units(g)) for g in gold_explanations if strip_tag_units(g)]
    return min(sizes) if sizes else 1


def evaluate_example(
    example_id: str,
    rows: List[Dict],
    baseline: str,
    input_format: str,
    support_size_mode: str,
    fixed_k: int,
) -> Dict:
    first = rows[0]
    question_text = get_question_text(first, input_format=input_format)
    gold_explanations = get_gold_explanations(first)

    candidate_units = get_candidate_units(rows)

    scored_units = []

    for unit in candidate_units:
        if baseline == "lexical":
            score = lexical_score(question_text, unit)
        elif baseline == "random":
            score = random_score(question_text, unit)
        else:
            raise ValueError(baseline)

        scored_units.append((unit, score))

    scored_units = sorted(scored_units, key=lambda x: x[1], reverse=True)

    if support_size_mode == "min_gold":
        k = min_gold_size(gold_explanations)
    elif support_size_mode == "fixed":
        k = fixed_k
    else:
        raise ValueError(support_size_mode)

    pred_top1 = [u for u, _ in scored_units[:k]]
    pred_top3 = [u for u, _ in scored_units[: max(k, 3)]]
    pred_top5 = [u for u, _ in scored_units[: max(k, 5)]]

    s1 = set_scores(pred_top1, gold_explanations)
    s3 = set_scores(pred_top3, gold_explanations)
    s5 = set_scores(pred_top5, gold_explanations)

    return {
        "example_id": example_id,
        "dataset": infer_dataset(example_id, first),
        "hop": infer_hop(example_id, first),
        "answer_type": infer_answer_type(example_id, first),
        "predicted_support@1": pred_top1,
        "predicted_support@3": pred_top3,
        "predicted_support@5": pred_top5,
        "gold_explanations": gold_explanations,
        "top1_exact_match_any_gold": s1["exact"],
        "top1_contains_any_gold_explanation": s1["contains"],
        "top1_best_jaccard_to_gold": s1["best_jaccard"],
        "top1_best_set_f1_to_gold": s1["best_f1"],
        "top1_best_precision_to_gold": s1["best_precision"],
        "top1_best_recall_to_gold": s1["best_recall"],
        "top3_exact_match_any_gold": s3["exact"],
        "top3_contains_any_gold_explanation": s3["contains"],
        "top3_best_jaccard_to_gold": s3["best_jaccard"],
        "top3_best_set_f1_to_gold": s3["best_f1"],
        "top3_best_precision_to_gold": s3["best_precision"],
        "top3_best_recall_to_gold": s3["best_recall"],
        "top5_exact_match_any_gold": s5["exact"],
        "top5_contains_any_gold_explanation": s5["contains"],
        "top5_best_jaccard_to_gold": s5["best_jaccard"],
        "top5_best_set_f1_to_gold": s5["best_f1"],
        "top5_best_precision_to_gold": s5["best_precision"],
        "top5_best_recall_to_gold": s5["best_recall"],
    }


def compute_metrics(details: List[Dict]) -> Dict:
    n = max(len(details), 1)

    return {
        "examples": len(details),
        "em@1": sum(1 for d in details if d["top1_exact_match_any_gold"]) / n,
        "gold_contained@1": sum(
            1 for d in details if d["top1_contains_any_gold_explanation"]
        )
        / n,
        "support_jaccard@1": sum(d["top1_best_jaccard_to_gold"] for d in details) / n,
        "support_f1@1": sum(d["top1_best_set_f1_to_gold"] for d in details) / n,
        "support_precision@1": sum(d["top1_best_precision_to_gold"] for d in details)
        / n,
        "support_recall@1": sum(d["top1_best_recall_to_gold"] for d in details) / n,
        "em@3": sum(1 for d in details if d["top3_exact_match_any_gold"]) / n,
        "gold_contained@3": sum(
            1 for d in details if d["top3_contains_any_gold_explanation"]
        )
        / n,
        "support_jaccard@3": sum(d["top3_best_jaccard_to_gold"] for d in details) / n,
        "support_f1@3": sum(d["top3_best_set_f1_to_gold"] for d in details) / n,
        "support_precision@3": sum(d["top3_best_precision_to_gold"] for d in details)
        / n,
        "support_recall@3": sum(d["top3_best_recall_to_gold"] for d in details) / n,
        "em@5": sum(1 for d in details if d["top5_exact_match_any_gold"]) / n,
        "gold_contained@5": sum(
            1 for d in details if d["top5_contains_any_gold_explanation"]
        )
        / n,
        "support_jaccard@5": sum(d["top5_best_jaccard_to_gold"] for d in details) / n,
        "support_f1@5": sum(d["top5_best_set_f1_to_gold"] for d in details) / n,
        "support_precision@5": sum(d["top5_best_precision_to_gold"] for d in details)
        / n,
        "support_recall@5": sum(d["top5_best_recall_to_gold"] for d in details) / n,
    }


def compute_split_metrics(details: List[Dict]) -> Dict:
    splits = defaultdict(list)

    for d in details:
        splits["ALL"].append(d)
        splits[d["answer_type"]].append(d)

        hop = d["hop"]
        answer_type = d["answer_type"]
        splits[f"{hop}|{answer_type}"].append(d)

    return {name: compute_metrics(rows) for name, rows in splits.items()}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-path", required=True)
    parser.add_argument("--baseline", choices=["random", "lexical"], default="lexical")
    parser.add_argument(
        "--input-format", choices=["nl", "abs", "sparql", "hybrid"], default="hybrid"
    )
    parser.add_argument(
        "--support-size-mode", choices=["min_gold", "fixed"], default="min_gold"
    )
    parser.add_argument("--fixed-k", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-name", default=None)

    args = parser.parse_args()

    rows = load_jsonl(args.test_path, source_name=args.source_name)
    grouped = group_by_example(rows)

    details = []

    for example_id, ex_rows in grouped.items():
        details.append(
            evaluate_example(
                example_id=example_id,
                rows=ex_rows,
                baseline=args.baseline,
                input_format=args.input_format,
                support_size_mode=args.support_size_mode,
                fixed_k=args.fixed_k,
            )
        )

    metrics = compute_split_metrics(details)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = (
        output_dir
        / f"axiom_{args.baseline}_{args.input_format}_{args.support_size_mode}_metrics.json"
    )
    details_path = (
        output_dir
        / f"axiom_{args.baseline}_{args.input_format}_{args.support_size_mode}_details.json"
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved details to: {details_path}")


if __name__ == "__main__":
    main()
