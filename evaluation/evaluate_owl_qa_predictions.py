import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_answer_only(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        return load_jsonl(path)
    return load_json(path)


def normalize_answer(s: str) -> str:
    s = normalize_bool_answer(s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def normalize_bool_answer(s: str) -> str:
    s = str(s).strip().lower()

    yes_values = {"true", "yes", "1", "entailed", "correct"}
    no_values = {"false", "no", "0", "not entailed", "incorrect"}

    if s in yes_values:
        return "yes"
    if s in no_values:
        return "no"

    return s


def normalize_entity_name(s: str) -> str:
    s = str(s)

    # Keep URI local names.
    if "#" in s:
        s = s.split("#")[-1]
    if "/" in s and s.startswith("http"):
        s = s.rstrip("/").split("/")[-1]

    # Remove year suffixes like _1863, _1944.
    s = re.sub(r"_[0-9]{3,4}$", "", s)

    # Convert snake/camel case to words.
    s = s.replace("_", " ")
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)

    return normalize_answer(s)


def answer_f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_prediction == normalized_ground_truth and normalized_prediction:
        return 1.0, 1.0, 1.0

    zero = (0.0, 0.0, 0.0)

    if (
        normalized_prediction in ["yes", "no", "noanswer", "unknown"]
        and normalized_prediction != normalized_ground_truth
    ):
        return zero
    if (
        normalized_ground_truth in ["yes", "no", "noanswer", "unknown"]
        and normalized_prediction != normalized_ground_truth
    ):
        return zero

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    if not prediction_tokens or not ground_truth_tokens:
        return zero

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return zero

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


def answer_em_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def split_answer_items(s: str) -> List[str]:
    s = str(s or "")
    parts = re.split(r"\s*;\s*|\s*,\s*|\s+\band\b\s+", s)

    items = []
    for p in parts:
        p = normalize_entity_name(p)
        if p:
            items.append(p)

    return sorted(set(items))


def answer_set_scores(prediction: str, gold: str):
    pred_set = set(split_answer_items(prediction))
    gold_set = set(split_answer_items(gold))

    if not pred_set or not gold_set:
        return answer_em_score(prediction, gold), *answer_f1_score(prediction, gold)

    tp = len(pred_set & gold_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(gold_set) if gold_set else 0.0
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    em = float(pred_set == gold_set)

    return em, f1, prec, rec


def support_scores(pred_units: List[str], gold_units: List[str]) -> Dict[str, float]:
    pred = set(pred_units)
    gold = set(gold_units)

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    em = 1.0 if fp + fn == 0 else 0.0

    return {
        "em": em,
        "f1": f1,
        "prec": precision,
        "recall": recall,
    }


def best_support_scores(
    pred_units: List[str], gold_explanations: List[List[str]]
) -> Dict[str, Any]:
    """
    If multiple gold explanations exist, evaluate against the one with highest support F1.
    """
    if not gold_explanations:
        return {
            "em": 0.0,
            "f1": 0.0,
            "prec": 0.0,
            "recall": 0.0,
            "best_gold": [],
        }

    best = None
    best_gold = []

    for gold in gold_explanations:
        cur = support_scores(pred_units, gold)
        if best is None or cur["f1"] > best["f1"]:
            best = cur
            best_gold = gold

    best = best or {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    best["best_gold"] = best_gold
    return best


def get_top_support_units(item: Dict[str, Any], top_k: int) -> List[str]:
    units = []
    seen = set()

    top_list = item.get("top5") or item.get("topk") or []

    if isinstance(top_list, list) and top_list:
        top_list = sorted(top_list, key=lambda x: int(x.get("rank", 999)))

        for cand in top_list[:top_k]:
            for u in cand.get("subgraph_units", []):
                if u not in seen:
                    seen.add(u)
                    units.append(u)

    if not units:
        for u in item.get("top1_subgraph_units", []):
            if u not in seen:
                seen.add(u)
                units.append(u)

    return units


def retrieval_at_k(item: Dict[str, Any], k: int) -> Dict[str, float]:
    gold_explanations = item.get("gold_explanations", []) or []
    top_list = item.get("top5") or item.get("topk") or []
    top_list = sorted(top_list, key=lambda x: int(x.get("rank", 999)))[:k]

    exact = 0.0
    contained = 0.0
    best_f1 = 0.0

    for cand in top_list:
        cand_units = cand.get("subgraph_units", [])
        cset = set(cand_units)

        for gold in gold_explanations:
            gset = set(gold)

            if cset == gset:
                exact = 1.0

            if gset and gset.issubset(cset):
                contained = 1.0

            cur = support_scores(cand_units, gold)
            best_f1 = max(best_f1, cur["f1"])

    return {
        f"exact@{k}": exact,
        f"contained@{k}": contained,
        f"support_set_f1@{k}": best_f1,
    }


def evaluate(
    details_path: str,
    llm_answers_path: str,
    top_k: int,
    output_path: str = None,
    answer_only_paths: List[str] = None,
) -> Dict[str, Any]:
    details = load_json(details_path)
    answer_only_paths = answer_only_paths or []
    for path in answer_only_paths:
        if Path(path).exists():
            details.extend(load_answer_only(path))

    answer_rows = load_jsonl(llm_answers_path)

    answers_by_id = {
        row.get("example_id"): row for row in answer_rows if row.get("example_id")
    }

    metrics = {
        "em": 0.0,
        "f1": 0.0,
        "prec": 0.0,
        "recall": 0.0,
        "sp_em": 0.0,
        "sp_f1": 0.0,
        "sp_prec": 0.0,
        "sp_recall": 0.0,
        "joint_em": 0.0,
        "joint_f1": 0.0,
        "joint_prec": 0.0,
        "joint_recall": 0.0,
        f"exact@{top_k}": 0.0,
        f"contained@{top_k}": 0.0,
        f"support_set_f1@{top_k}": 0.0,
        "examples": 0,
        "answer_examples": 0,
        "support_examples": 0,
        "answer_only_examples": 0,
        "missing_answers": 0,
        "empty_predictions": 0,
        "errors": 0,
    }

    per_example = []

    for item in details:
        example_id = item.get("example_id")
        if not example_id:
            continue

        gold_answer = str(item.get("answer", item.get("gold_answer", "")))
        gold_explanations = item.get("gold_explanations", []) or []
        has_gold_support = bool(gold_explanations)

        ans_row = answers_by_id.get(example_id)
        if ans_row is None:
            metrics["missing_answers"] += 1
            predicted_answer = ""
            error = "missing_answer_row"
        else:
            predicted_answer = str(ans_row.get("predicted_answer", "")).strip()
            error = ans_row.get("error")

        if not predicted_answer:
            metrics["empty_predictions"] += 1

        if error:
            metrics["errors"] += 1

        # Answer metrics.
        ans_em, ans_f1, ans_prec, ans_recall = answer_set_scores(
            predicted_answer,
            gold_answer,
        )

        pred_support = []
        sp = {"em": None, "f1": None, "prec": None, "recall": None, "best_gold": []}
        sp_em = sp_f1 = sp_prec = sp_recall = None
        joint_em = joint_f1 = joint_prec = joint_recall = None
        ret = None

        if has_gold_support:
            # Support metrics using top-k union from details.
            pred_support = get_top_support_units(item, top_k=top_k)
            if not pred_support and ans_row is not None:
                pred_support = ans_row.get("support_units", []) or []
            sp = best_support_scores(pred_support, gold_explanations)

            sp_em = sp["em"]
            sp_f1 = sp["f1"]
            sp_prec = sp["prec"]
            sp_recall = sp["recall"]

            # Joint metrics, same idea as HotpotQA.
            joint_prec = ans_prec * sp_prec
            joint_recall = ans_recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.0
            joint_em = ans_em * sp_em

            ret = retrieval_at_k(item, k=top_k)

        metrics["em"] += ans_em
        metrics["f1"] += ans_f1
        metrics["prec"] += ans_prec
        metrics["recall"] += ans_recall

        metrics["examples"] += 1
        metrics["answer_examples"] += 1

        if has_gold_support:
            metrics["support_examples"] += 1
            metrics["sp_em"] += sp_em
            metrics["sp_f1"] += sp_f1
            metrics["sp_prec"] += sp_prec
            metrics["sp_recall"] += sp_recall

            metrics["joint_em"] += joint_em
            metrics["joint_f1"] += joint_f1
            metrics["joint_prec"] += joint_prec
            metrics["joint_recall"] += joint_recall

            metrics[f"exact@{top_k}"] += ret[f"exact@{top_k}"]
            metrics[f"contained@{top_k}"] += ret[f"contained@{top_k}"]
            metrics[f"support_set_f1@{top_k}"] += ret[f"support_set_f1@{top_k}"]
        else:
            metrics["answer_only_examples"] += 1

        per_example.append(
            {
                "example_id": example_id,
                "question": item.get("question", ""),
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "answer_em": ans_em,
                "answer_f1": ans_f1,
                "has_gold_support": has_gold_support,
                "evaluation_scope": item.get("evaluation_scope", "support"),
                "support_em": sp_em,
                "support_f1": sp_f1,
                "support_precision": sp_prec,
                "support_recall": sp_recall,
                "joint_em": joint_em,
                "joint_f1": joint_f1,
                "retrieval": ret,
                "pred_support": pred_support,
                "best_gold_support": sp.get("best_gold", []),
                "error": error,
            }
        )

    n = metrics["examples"]

    if n > 0:
        for key in [
            "em",
            "f1",
            "prec",
            "recall",
        ]:
            metrics[key] /= n

    support_n = metrics["support_examples"]

    if support_n > 0:
        for key in [
            "sp_em",
            "sp_f1",
            "sp_prec",
            "sp_recall",
            "joint_em",
            "joint_f1",
            "joint_prec",
            "joint_recall",
            f"exact@{top_k}",
            f"contained@{top_k}",
            f"support_set_f1@{top_k}",
        ]:
            metrics[key] /= support_n

    result = {
        "metrics": metrics,
        "details_path": details_path,
        "llm_answers_path": llm_answers_path,
        "top_k": top_k,
        "per_example": per_example,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--details", type=str, required=True)
    parser.add_argument("--llm-answers", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--answer-only", type=str, nargs="*", default=[])

    args = parser.parse_args()

    evaluate(
        details_path=args.details,
        llm_answers_path=args.llm_answers,
        top_k=args.top_k,
        output_path=args.output,
        answer_only_paths=args.answer_only,
    )


if __name__ == "__main__":
    main()
