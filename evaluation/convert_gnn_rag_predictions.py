import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_prediction(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value or "").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]

    return [text]


def normalize_prediction(value: Any) -> str:
    items = []
    for item in flatten_prediction(value):
        item = item.replace("ANSWER::", "").replace("Answer::", "").strip()
        if item:
            items.append(item)

    for item in items:
        normalized = item.strip().lower()
        if normalized in {"true", "yes", "1"} or normalized.startswith("yes,"):
            return "TRUE"
        if normalized in {"false", "no", "0"} or normalized.startswith("no,"):
            return "FALSE"

    if len(items) == 1:
        return items[0]
    return "; ".join(items)


def extract_support_units(prompt: str) -> List[str]:
    units = []
    seen = set()
    for match in re.finditer(r"EVIDENCE::(.+?)(?:\s*->|\n|$)", str(prompt or "")):
        unit = match.group(1).strip()
        if unit and unit not in seen:
            seen.add(unit)
            units.append(unit)
    return units


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    out_rows = []
    for row in load_jsonl(args.input):
        prediction = row.get("prediction", "")
        predicted_answer = normalize_prediction(prediction)
        support_units = extract_support_units(row.get("input", ""))

        out_rows.append(
            {
                "example_id": row.get("id", ""),
                "question": row.get("question", ""),
                "gold_answer": row.get("ground_truth", ""),
                "predicted_answer": predicted_answer,
                "explanation": "",
                "support_units": support_units,
                "raw_response": prediction,
                "gnn_rag_input": row.get("input", ""),
                "method": "gnn_rag",
            }
        )

    write_jsonl(args.output, out_rows)
    print(f"Converted {len(out_rows)} GNN-RAG predictions to: {args.output}")


if __name__ == "__main__":
    main()
