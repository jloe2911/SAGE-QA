import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_sent_unit(unit: str) -> Tuple[str, int]:
    """
    SENT::Page Title::3::sentence text
    -> ("Page Title", 3)
    """
    parts = str(unit).split("::", 3)
    if len(parts) >= 3 and parts[0] == "SENT":
        title = parts[1]
        try:
            idx = int(parts[2])
        except Exception:
            idx = -1
        return title, idx

    if len(parts) >= 3 and parts[0] == "EVIDENCE":
        title = parts[1]
        try:
            idx = int(parts[2])
        except Exception:
            idx = -1
        return title, idx

    loose_parts = str(unit).split("::", 2)
    if len(loose_parts) == 3:
        title = loose_parts[0]
        try:
            idx = int(loose_parts[1])
        except Exception:
            idx = -1
        return title, idx

    return str(unit), -1


def get_hotpot_id(example_id: str) -> str:
    """
    Converts:
      HotpotQA__test__5a8b57f25542995d1e6f1371
    into:
      5a8b57f25542995d1e6f1371

    If no prefix is found, returns original.
    """
    prefix = "HotpotQA__test__"
    if example_id.startswith(prefix):
        return example_id[len(prefix) :]

    # More general fallback:
    parts = example_id.split("__")
    if len(parts) >= 3 and parts[0] == "HotpotQA":
        return parts[-1]

    return example_id


def get_top_support_units(item: Dict[str, Any], top_k: int) -> List[str]:
    """
    Gets union of support units from top-k candidates.
    """
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

    if not units and item.get("top1_subgraph_units"):
        for u in item["top1_subgraph_units"]:
            if u not in seen:
                seen.add(u)
                units.append(u)

    return units


def load_llm_answers(path: str) -> Dict[str, str]:
    """
    Expects JSONL rows like:
      {
        "example_id": "HotpotQA__test__...",
        "predicted_answer": "...",
        ...
      }
    """
    rows = load_jsonl(path)
    answers = {}

    for row in rows:
        ex_id = row.get("example_id")
        if not ex_id:
            continue

        hp_id = get_hotpot_id(ex_id)

        answer = row.get("predicted_answer") or row.get("answer") or row.get("prediction") or ""

        answers[hp_id] = str(answer)

    return answers


def load_llm_rows(path: str) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for row in load_jsonl(path):
        ex_id = row.get("example_id")
        if ex_id:
            rows[get_hotpot_id(ex_id)] = row
    return rows


def export_predictions(
    details_path: str,
    output_path: str,
    top_k: int,
    llm_answers_path: str = None,
    fallback_answer_from_details: bool = True,
):
    details = load_json(details_path)

    llm_answers = {}
    llm_rows = {}
    if llm_answers_path:
        llm_answers = load_llm_answers(llm_answers_path)
        llm_rows = load_llm_rows(llm_answers_path)

    pred = {
        "answer": {},
        "sp": {},
    }

    for item in details:
        example_id = item.get("example_id")
        if not example_id:
            continue

        hp_id = get_hotpot_id(example_id)

        # Answer prediction
        if hp_id in llm_answers:
            answer = llm_answers[hp_id]
        elif fallback_answer_from_details:
            # This is not a real generated prediction.
            # Useful only before LLM generation is implemented.
            answer = str(item.get("answer", ""))
        else:
            answer = ""

        pred["answer"][hp_id] = answer

        # Supporting-fact prediction
        units = get_top_support_units(item, top_k=top_k)
        if not units and hp_id in llm_rows:
            units = llm_rows[hp_id].get("support_units", []) or []

        sp_facts: List[List[Any]] = []
        seen_sp: Set[Tuple[str, int]] = set()

        for u in units:
            title, idx = parse_sent_unit(u)
            if idx < 0:
                continue
            key = (title, idx)
            if key not in seen_sp:
                seen_sp.add(key)
                sp_facts.append([title, idx])

        pred["sp"][hp_id] = sp_facts

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(pred, f, indent=2, ensure_ascii=False)

    print(f"Saved HotpotQA predictions to: {out}")
    print(f"Examples: {len(pred['answer'])}")
    print(f"Top-k support union: {top_k}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--details", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=3)

    parser.add_argument(
        "--llm-answers",
        type=str,
        default=None,
        help="Optional JSONL with generated answers.",
    )

    parser.add_argument(
        "--no-fallback-answer",
        action="store_true",
        help="Do not use gold answer from details as temporary fallback.",
    )

    args = parser.parse_args()

    export_predictions(
        details_path=args.details,
        output_path=args.output,
        top_k=args.top_k,
        llm_answers_path=args.llm_answers,
        fallback_answer_from_details=not args.no_fallback_answer,
    )


if __name__ == "__main__":
    main()
