import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


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


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    return text


def tokenize(text: str) -> set:
    return set(t for t in normalize_text(text).split() if t)


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


def infer_dataset(example_id: str, row: Dict) -> str:
    for key in ["dataset", "Dataset", "source_dataset"]:
        if key in row and row[key]:
            return str(row[key])
    if "__" in example_id:
        return example_id.split("__", 1)[0]
    return "UNKNOWN"


def infer_hop(example_id: str, row: Dict) -> str:
    for key in ["hop", "Hop", "hop_type"]:
        if key in row and row[key]:
            return str(row[key]).lower()

    text = example_id.lower()
    if "2hop" in text or "2-hop" in text:
        return "2hop"
    if "1hop" in text or "1-hop" in text:
        return "1hop"
    return "UNKNOWN"


def group_by_example(rows: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["example_id"]].append(row)
    return dict(grouped)


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


def row_subgraph_text(row: Dict) -> str:
    units = row.get("subgraph_units", [])
    return " ".join(str(u) for u in units)


def lexical_score(row: Dict, input_format: str) -> float:
    q_tokens = tokenize(get_question_text(row, input_format=input_format))
    s_tokens = tokenize(row_subgraph_text(row))

    if not q_tokens or not s_tokens:
        return 0.0

    overlap = len(q_tokens & s_tokens)
    union = len(q_tokens | s_tokens)
    return overlap / max(union, 1)


def random_score(_: Dict, __: str = "hybrid") -> float:
    return random.random()


def as_bool(row: Dict, key: str) -> bool:
    return bool(row.get(key, False))


def as_float(row: Dict, key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except Exception:
        return 0.0


def eval_group(rows: List[Dict], scoring_fn, input_format: str) -> Dict:
    details = []

    grouped = group_by_example(rows)

    for example_id, candidates in grouped.items():
        scored = []
        for row in candidates:
            r = dict(row)
            r["score"] = float(scoring_fn(r, input_format))
            scored.append(r)

        scored = sorted(scored, key=lambda r: r["score"], reverse=True)
        top1 = scored[0]
        top3 = scored[:3]
        top5 = scored[:5]

        details.append(
            {
                "example_id": example_id,
                "dataset": infer_dataset(example_id, top1),
                "hop": infer_hop(example_id, top1),
                "answer_type": infer_answer_type(example_id, top1),
                "top1_label": int(top1.get("label", 0)),
                "top1_exact_match_any_gold": as_bool(top1, "exact_match_any_gold"),
                "top1_contains_any_gold_explanation": as_bool(
                    top1, "contains_any_gold_explanation"
                ),
                "top1_best_jaccard_to_gold": as_float(top1, "best_jaccard_to_gold"),
                "top1_best_set_f1_to_gold": as_float(top1, "best_set_f1_to_gold"),
                "top1_best_precision_to_gold": as_float(
                    top1, "best_set_precision_to_gold"
                ),
                "top1_best_recall_to_gold": as_float(top1, "best_set_recall_to_gold"),
                "top5": [
                    {
                        "rank": i + 1,
                        "score": r["score"],
                        "label": int(r.get("label", 0)),
                        "exact_match_any_gold": as_bool(r, "exact_match_any_gold"),
                        "contains_any_gold_explanation": as_bool(
                            r, "contains_any_gold_explanation"
                        ),
                        "best_jaccard_to_gold": as_float(r, "best_jaccard_to_gold"),
                        "best_set_f1_to_gold": as_float(r, "best_set_f1_to_gold"),
                        "subgraph_units": r.get("subgraph_units", []),
                    }
                    for i, r in enumerate(top5)
                ],
            }
        )

    return compute_split_metrics(details), details


def compute_metrics(details: List[Dict]) -> Dict:
    n = max(len(details), 1)

    hit1 = sum(1 for d in details if int(d.get("top1_label", 0)) == 1) / n
    exact1 = sum(1 for d in details if d.get("top1_exact_match_any_gold", False)) / n
    contains1 = (
        sum(1 for d in details if d.get("top1_contains_any_gold_explanation", False))
        / n
    )

    jaccard1 = sum(float(d.get("top1_best_jaccard_to_gold", 0.0)) for d in details) / n
    f11 = sum(float(d.get("top1_best_set_f1_to_gold", 0.0)) for d in details) / n
    precision1 = (
        sum(float(d.get("top1_best_precision_to_gold", 0.0)) for d in details) / n
    )
    recall1 = sum(float(d.get("top1_best_recall_to_gold", 0.0)) for d in details) / n

    hit3 = 0
    exact3 = 0
    contains3 = 0
    best_jaccard3 = 0.0
    best_f13 = 0.0

    hit5 = 0
    exact5 = 0
    contains5 = 0
    best_jaccard5 = 0.0
    best_f15 = 0.0

    for d in details:
        top5 = d.get("top5", [])
        top3 = top5[:3]

        hit3 += int(any(int(r.get("label", 0)) == 1 for r in top3))
        exact3 += int(any(r.get("exact_match_any_gold", False) for r in top3))
        contains3 += int(
            any(r.get("contains_any_gold_explanation", False) for r in top3)
        )

        hit5 += int(any(int(r.get("label", 0)) == 1 for r in top5))
        exact5 += int(any(r.get("exact_match_any_gold", False) for r in top5))
        contains5 += int(
            any(r.get("contains_any_gold_explanation", False) for r in top5)
        )

        if top3:
            best3 = max(top3, key=lambda r: float(r.get("best_set_f1_to_gold", 0.0)))
            best_jaccard3 += float(best3.get("best_jaccard_to_gold", 0.0))
            best_f13 += float(best3.get("best_set_f1_to_gold", 0.0))

        if top5:
            best5 = max(top5, key=lambda r: float(r.get("best_set_f1_to_gold", 0.0)))
            best_jaccard5 += float(best5.get("best_jaccard_to_gold", 0.0))
            best_f15 += float(best5.get("best_set_f1_to_gold", 0.0))

    return {
        "examples": len(details),
        "hit@1": hit1,
        "exact@1": exact1,
        "contains@1": contains1,
        "jaccard@1": jaccard1,
        "set_f1@1": f11,
        "precision@1": precision1,
        "recall@1": recall1,
        "hit@3": hit3 / n,
        "exact@3": exact3 / n,
        "contains@3": contains3 / n,
        "jaccard@3": best_jaccard3 / n,
        "set_f1@3": best_f13 / n,
        "hit@5": hit5 / n,
        "exact@5": exact5 / n,
        "contains@5": contains5 / n,
        "jaccard@5": best_jaccard5 / n,
        "set_f1@5": best_f15 / n,
    }


def compute_split_metrics(details: List[Dict]) -> Dict:
    splits = defaultdict(list)

    for d in details:
        splits["ALL"].append(d)
        splits[d.get("answer_type", "UNKNOWN")].append(d)

        hop = d.get("hop", "UNKNOWN")
        answer_type = d.get("answer_type", "UNKNOWN")
        splits[f"{hop}|{answer_type}"].append(d)

    return {name: compute_metrics(rows) for name, rows in splits.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--baseline", choices=["random", "lexical"], required=True)
    parser.add_argument(
        "--input-format", choices=["nl", "abs", "sparql", "hybrid"], default="hybrid"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-name", default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.test_path, source_name=args.source_name)

    if args.baseline == "random":
        scoring_fn = random_score
    elif args.baseline == "lexical":
        scoring_fn = lexical_score
    else:
        raise ValueError(args.baseline)

    metrics, details = eval_group(rows, scoring_fn, input_format=args.input_format)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{args.baseline}_{args.input_format}_metrics.json"
    details_path = output_dir / f"{args.baseline}_{args.input_format}_details.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved details to: {details_path}")


if __name__ == "__main__":
    main()
