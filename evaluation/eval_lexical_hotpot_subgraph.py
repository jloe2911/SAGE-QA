import argparse
import json
import math
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


ARTICLES = {"a", "an", "the"}


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(text: str) -> List[str]:
    return [t for t in normalize_text(text).split() if t and t not in ARTICLES]


def token_set(text: str) -> Set[str]:
    return set(tokens(text))


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def support_f1(pred_units: List[str], gold_units: List[str]) -> float:
    p = set(pred_units)
    g = set(gold_units)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    return 2 * len(p & g) / (len(p) + len(g))


def support_jaccard(pred_units: List[str], gold_units: List[str]) -> float:
    p = set(pred_units)
    g = set(gold_units)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    return len(p & g) / len(p | g)


def best_against_gold(
    pred_units: List[str], gold_explanations: List[List[str]]
) -> Dict[str, Any]:
    best_f1 = 0.0
    best_j = 0.0
    best_precision = 0.0
    best_recall = 0.0
    exact = False
    contains = False
    best_gold = []

    pred_set = set(pred_units)

    for gold in gold_explanations:
        gold_set = set(gold)

        if pred_set == gold_set:
            exact = True

        if gold_set and gold_set.issubset(pred_set):
            contains = True

        f1 = support_f1(pred_units, gold)
        jac = support_jaccard(pred_units, gold)

        if pred_set:
            prec = len(pred_set & gold_set) / len(pred_set)
        else:
            prec = 0.0

        if gold_set:
            rec = len(pred_set & gold_set) / len(gold_set)
        else:
            rec = 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_j = jac
            best_precision = prec
            best_recall = rec
            best_gold = gold

    return {
        "best_set_f1_to_gold": best_f1,
        "best_jaccard_to_gold": best_j,
        "best_precision_to_gold": best_precision,
        "best_recall_to_gold": best_recall,
        "exact_match_any_gold": exact,
        "contains_any_gold_explanation": contains,
        "best_matching_gold_explanation": best_gold,
    }


def lexical_score(row: Dict[str, Any]) -> float:
    question = row.get("question", "")
    answer = row.get("answer", "")
    units = row.get("subgraph_units", [])

    candidate_text = " ".join(str(u) for u in units)

    q = token_set(question)
    c = token_set(candidate_text)

    if not q:
        q_overlap = 0.0
    else:
        q_overlap = len(q & c) / len(q)

    # Small answer bonus if answer string appears in candidate.
    answer_norm = normalize_text(answer)
    cand_norm = normalize_text(candidate_text)
    answer_bonus = 1.0 if answer_norm and answer_norm in cand_norm else 0.0

    # Prefer compact candidates slightly.
    size = int(row.get("subgraph_size", len(units)))
    compact_bonus = 1.0 / max(size, 1)

    return q_overlap + 0.25 * answer_bonus + 0.02 * compact_bonus


def group_by_example(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["example_id"]].append(r)
    return dict(grouped)


def evaluate_group(
    rows: List[Dict[str, Any]], top_k_values=(1, 3, 5)
) -> Dict[str, float]:
    details = build_details(rows)

    metrics = {
        "examples": len(details),
    }

    for k in top_k_values:
        exact = []
        contains = []
        jac = []
        f1 = []
        prec = []
        rec = []

        for item in details:
            top = item["top5"][:k]
            best = {
                "exact": 0.0,
                "contains": 0.0,
                "jaccard": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

            for cand in top:
                best["exact"] = max(best["exact"], float(cand["exact_match_any_gold"]))
                best["contains"] = max(
                    best["contains"], float(cand["contains_any_gold_explanation"])
                )

                if cand["best_set_f1_to_gold"] > best["f1"]:
                    best["f1"] = cand["best_set_f1_to_gold"]
                    best["jaccard"] = cand["best_jaccard_to_gold"]
                    best["precision"] = cand["best_precision_to_gold"]
                    best["recall"] = cand["best_recall_to_gold"]

            exact.append(best["exact"])
            contains.append(best["contains"])
            jac.append(best["jaccard"])
            f1.append(best["f1"])
            prec.append(best["precision"])
            rec.append(best["recall"])

        metrics[f"exact_hit@{k}"] = sum(exact) / len(exact) if exact else 0.0
        metrics[f"contains_gold_hit@{k}"] = (
            sum(contains) / len(contains) if contains else 0.0
        )
        metrics[f"best_jaccard@{k}"] = sum(jac) / len(jac) if jac else 0.0
        metrics[f"best_set_f1@{k}"] = sum(f1) / len(f1) if f1 else 0.0
        metrics[f"best_precision@{k}"] = sum(prec) / len(prec) if prec else 0.0
        metrics[f"best_recall@{k}"] = sum(rec) / len(rec) if rec else 0.0

    return metrics


def build_details(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = group_by_example(rows)
    details = []

    for example_id, candidates in grouped.items():
        ranked = sorted(candidates, key=lexical_score, reverse=True)

        first = ranked[0]
        gold_explanations = first.get("gold_explanations", [])

        top5 = []
        for rank, row in enumerate(ranked[:5], start=1):
            units = row.get("subgraph_units", [])
            gold_stats = best_against_gold(units, gold_explanations)
            score = lexical_score(row)

            top5.append(
                {
                    "rank": rank,
                    "score": score,
                    "adjusted_score": score,
                    "label": int(row.get("label", 0)),
                    "rank_target": float(row.get("rank_target", 0.0)),
                    "subgraph_size": int(row.get("subgraph_size", len(units))),
                    "subgraph_units": units,
                    **gold_stats,
                }
            )

        top1 = top5[0]
        top1_units = top1["subgraph_units"]

        details.append(
            {
                "example_id": example_id,
                "dataset": first.get("dataset", "HotpotQA"),
                "hop": first.get("hop", "2hop"),
                "answer_type": first.get("answer_type", "OPEN"),
                "question": first.get("question", ""),
                "answer": first.get("answer", ""),
                "gold_explanations": gold_explanations,
                "top1_subgraph_units": top1_units,
                "top1_score": top1["score"],
                "top1_adjusted_score": top1["adjusted_score"],
                "top1_label": top1["label"],
                "top1_best_jaccard_to_gold": top1["best_jaccard_to_gold"],
                "top1_best_set_f1_to_gold": top1["best_set_f1_to_gold"],
                "top1_best_precision_to_gold": top1["best_precision_to_gold"],
                "top1_best_recall_to_gold": top1["best_recall_to_gold"],
                "top1_exact_match_any_gold": top1["exact_match_any_gold"],
                "top1_contains_any_gold_explanation": top1[
                    "contains_any_gold_explanation"
                ],
                "top1_best_matching_gold_explanation": top1[
                    "best_matching_gold_explanation"
                ],
                "top5": top5,
            }
        )

    return details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.test_path)
    details = build_details(rows)
    metrics = evaluate_group(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "test_details.json").open("w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    with (out_dir / "test_split_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"HotpotQA | 2hop | OPEN": metrics}, f, indent=2, ensure_ascii=False)

    print(json.dumps({"HotpotQA | 2hop | OPEN": metrics}, indent=2))
    print(f"\nSaved lexical details to: {out_dir}")


if __name__ == "__main__":
    main()
