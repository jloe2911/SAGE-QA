import argparse
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def resolve_input_path(path: str) -> str:
    original = Path(path)
    if original.exists():
        return str(original)

    candidates = []
    name = original.name
    parent = original.parent

    replacements = [
        ("gpt-4_1-mini", "gpt_4_1_mini"),
        ("gpt-4.1-mini", "gpt_4_1_mini"),
        ("gpt-4-1-mini", "gpt_4_1_mini"),
    ]

    for old, new in replacements:
        if old in name:
            candidates.append(parent / name.replace(old, new))

    if "-" in name:
        candidates.append(parent / name.replace("-", "_"))

    for candidate in candidates:
        if candidate.exists():
            print(f"[path] Using existing file for {path}: {candidate}")
            return str(candidate)

    return path


def load_json(path: str):
    path = resolve_input_path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    path = resolve_input_path(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def answer_scores(prediction: str, gold: str) -> Tuple[float, float, float, float]:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    em = float(pred_norm == gold_norm)

    if pred_norm in {"yes", "no", "unknown", "noanswer"} and pred_norm != gold_norm:
        return em, 0.0, 0.0, 0.0
    if gold_norm in {"yes", "no", "unknown", "noanswer"} and pred_norm != gold_norm:
        return em, 0.0, 0.0, 0.0

    pred_toks = pred_norm.split()
    gold_toks = gold_norm.split()

    if not pred_toks or not gold_toks:
        return em, 0.0, 0.0, 0.0

    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())

    if num_same == 0:
        return em, 0.0, 0.0, 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = 2 * precision * recall / (precision + recall)

    return em, f1, precision, recall


def support_scores(pred_units: List[Any], gold_units: List[Any]) -> Dict[str, float]:
    pred = {tuple(x) if isinstance(x, list) else x for x in pred_units}
    gold = {tuple(x) if isinstance(x, list) else x for x in gold_units}

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    em = float(pred == gold)

    return {
        "support_em": em,
        "support_f1": f1,
        "support_precision": precision,
        "support_recall": recall,
        "pred_support_size": len(pred),
        "gold_support_size": len(gold),
        "extra_support": fp,
        "missing_support": fn,
    }


def get_topk_candidate_diagnostics(detail: Dict[str, Any], k: int) -> Dict[str, Any]:
    top = detail.get("top5", [])[:k]

    exact_at_k = any(c.get("exact_match_any_gold", False) for c in top)
    contained_at_k = any(c.get("contains_any_gold_explanation", False) for c in top)
    best_f1 = max([float(c.get("best_set_f1_to_gold", 0.0)) for c in top] or [0.0])

    top1 = top[0] if top else {}
    top1_units = top1.get("subgraph_units", detail.get("top1_subgraph_units", []))

    return {
        "candidate_exact_at_k": float(exact_at_k),
        "candidate_contained_at_k": float(contained_at_k),
        "candidate_best_support_set_f1_at_k": best_f1,
        "top1_size": len(top1_units),
        "topk_union_size": len(
            set(u for c in top for u in c.get("subgraph_units", []))
        ),
    }


def classify_error(
    ans_em: float,
    ans_f1: float,
    sp_em: float,
    sp_f1: float,
    sp_recall: float,
    contained_at_k: float,
    pred_support_size: int,
    gold_support_size: int,
) -> str:
    answer_ok = ans_em == 1.0 or ans_f1 >= 0.8
    support_exact = sp_em == 1.0
    support_good = sp_f1 >= 0.8
    support_recall_high = sp_recall >= 0.8

    if answer_ok and support_exact:
        return "correct_answer_correct_support"

    if (
        answer_ok
        and not support_exact
        and support_recall_high
        and pred_support_size > gold_support_size
    ):
        return "correct_answer_noisy_support"

    if answer_ok and sp_recall < 0.8:
        return "correct_answer_incomplete_support"

    if not answer_ok and contained_at_k == 1.0:
        return "reader_failed_despite_gold_in_topk"

    if not answer_ok and support_good:
        return "reader_failed_despite_good_support_overlap"

    if not answer_ok and sp_recall >= 0.8 and pred_support_size > gold_support_size:
        return "answer_failed_noisy_support"

    if not answer_ok and sp_recall < 0.5:
        return "retrieval_missed_gold_support"

    return "mixed_or_partial_error"


def gold_by_id_from_hotpot_style(gold_path: str) -> Dict[str, Dict[str, Any]]:
    gold = load_json(gold_path)
    return {
        row["_id"]: {
            "answer": row.get("answer", ""),
            "supporting_facts": row.get("supporting_facts", []),
        }
        for row in gold
    }


def strip_prefix_id(example_id: str) -> str:
    # Handles IDs like HotpotQA__test__abc or 2WikiMultiHopQA__test__abc.
    if "__" in example_id:
        return example_id.split("__")[-1]
    return example_id


def analyze_text(
    details_path: str,
    llm_answers_path: str,
    gold_path: str,
    prediction_path: str,
    output_csv: str,
    output_summary: str,
    top_k: int,
):
    details = load_json(details_path)
    answers = load_jsonl(llm_answers_path)
    gold_by_id = gold_by_id_from_hotpot_style(gold_path)
    predictions = load_json(prediction_path)

    answer_by_id = {strip_prefix_id(r.get("example_id", "")): r for r in answers}
    detail_by_id = {strip_prefix_id(d.get("example_id", "")): d for d in details}

    rows = []

    for ex_id, pred_answer in predictions.get("answer", {}).items():
        gold = gold_by_id.get(ex_id, {})
        detail = detail_by_id.get(ex_id, {})
        llm_row = answer_by_id.get(ex_id, {})

        gold_answer = gold.get("answer", "")
        predicted_answer = pred_answer
        pred_support = predictions.get("sp", {}).get(ex_id, [])
        gold_support = gold.get("supporting_facts", [])

        ans_em, ans_f1, ans_prec, ans_rec = answer_scores(predicted_answer, gold_answer)
        sp = support_scores(pred_support, gold_support)
        cand = get_topk_candidate_diagnostics(detail, top_k)

        category = classify_error(
            ans_em=ans_em,
            ans_f1=ans_f1,
            sp_em=sp["support_em"],
            sp_f1=sp["support_f1"],
            sp_recall=sp["support_recall"],
            contained_at_k=cand["candidate_contained_at_k"],
            pred_support_size=sp["pred_support_size"],
            gold_support_size=sp["gold_support_size"],
        )

        rows.append(
            {
                "example_id": ex_id,
                "question": detail.get("question", llm_row.get("question", "")),
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "answer_em": ans_em,
                "answer_f1": ans_f1,
                "support_em": sp["support_em"],
                "support_f1": sp["support_f1"],
                "support_precision": sp["support_precision"],
                "support_recall": sp["support_recall"],
                "pred_support_size": sp["pred_support_size"],
                "gold_support_size": sp["gold_support_size"],
                "extra_support": sp["extra_support"],
                "missing_support": sp["missing_support"],
                **cand,
                "error_category": category,
                "gold_support": json.dumps(gold_support, ensure_ascii=False),
                "pred_support": json.dumps(pred_support, ensure_ascii=False),
            }
        )

    write_outputs(rows, output_csv, output_summary)


def analyze_owl(
    details_path: str,
    llm_answers_path: str,
    output_csv: str,
    output_summary: str,
    top_k: int,
):
    details = load_json(details_path)
    answers = load_jsonl(llm_answers_path)
    answer_by_id = {r.get("example_id"): r for r in answers}

    rows = []

    for detail in details:
        ex_id = detail.get("example_id")
        llm_row = answer_by_id.get(ex_id, {})

        gold_answer = str(detail.get("answer", llm_row.get("gold_answer", "")))
        predicted_answer = str(llm_row.get("predicted_answer", ""))

        ans_em, ans_f1, ans_prec, ans_rec = answer_scores(predicted_answer, gold_answer)

        pred_support = []
        for c in detail.get("top5", [])[:top_k]:
            pred_support.extend(c.get("subgraph_units", []))
        pred_support = list(dict.fromkeys(pred_support))

        gold_support_units = detail.get("gold_support_units", []) or []
        gold_explanations = detail.get("gold_explanations", []) or (
            [gold_support_units] if gold_support_units else [[]]
        )
        # Use the gold explanation with best support F1.
        best_sp = None
        best_gold = None
        for gold_support in gold_explanations:
            sp = support_scores(pred_support, gold_support)
            if best_sp is None or sp["support_f1"] > best_sp["support_f1"]:
                best_sp = sp
                best_gold = gold_support

        sp = best_sp or support_scores(pred_support, [])
        cand = get_topk_candidate_diagnostics(detail, top_k)

        category = classify_error(
            ans_em=ans_em,
            ans_f1=ans_f1,
            sp_em=sp["support_em"],
            sp_f1=sp["support_f1"],
            sp_recall=sp["support_recall"],
            contained_at_k=cand["candidate_contained_at_k"],
            pred_support_size=sp["pred_support_size"],
            gold_support_size=sp["gold_support_size"],
        )

        rows.append(
            {
                "example_id": ex_id,
                "question": detail.get("question", llm_row.get("question", "")),
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "answer_em": ans_em,
                "answer_f1": ans_f1,
                "support_em": sp["support_em"],
                "support_f1": sp["support_f1"],
                "support_precision": sp["support_precision"],
                "support_recall": sp["support_recall"],
                "pred_support_size": sp["pred_support_size"],
                "gold_support_size": sp["gold_support_size"],
                "extra_support": sp["extra_support"],
                "missing_support": sp["missing_support"],
                **cand,
                "error_category": category,
                "gold_support": json.dumps(best_gold, ensure_ascii=False),
                "pred_support": json.dumps(pred_support, ensure_ascii=False),
            }
        )

    write_outputs(rows, output_csv, output_summary)


def write_outputs(rows: List[Dict[str, Any]], output_csv: str, output_summary: str):
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    counts = Counter(r["error_category"] for r in rows)
    n = len(rows)

    summary = {
        "examples": n,
        "category_counts": dict(counts),
        "category_rates": {k: v / n for k, v in counts.items()} if n else {},
        "avg_pred_support_size": sum(r["pred_support_size"] for r in rows) / n
        if n
        else 0.0,
        "avg_gold_support_size": sum(r["gold_support_size"] for r in rows) / n
        if n
        else 0.0,
        "avg_extra_support": sum(r["extra_support"] for r in rows) / n if n else 0.0,
        "avg_missing_support": sum(r["missing_support"] for r in rows) / n
        if n
        else 0.0,
    }

    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved per-example analysis to: {output_csv}")
    print(f"Saved summary to: {output_summary}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "owl"], required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--llm-answers", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)

    # Text-only.
    parser.add_argument("--gold", default=None)
    parser.add_argument("--predictions", default=None)

    args = parser.parse_args()

    if args.mode == "text":
        if not args.gold or not args.predictions:
            raise ValueError("--gold and --predictions are required in text mode")
        analyze_text(
            details_path=args.details,
            llm_answers_path=args.llm_answers,
            gold_path=args.gold,
            prediction_path=args.predictions,
            output_csv=args.output_csv,
            output_summary=args.output_summary,
            top_k=args.top_k,
        )
    else:
        analyze_owl(
            details_path=args.details,
            llm_answers_path=args.llm_answers,
            output_csv=args.output_csv,
            output_summary=args.output_summary,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
