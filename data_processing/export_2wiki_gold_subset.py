import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def to_python_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if hasattr(x, "tolist"):
        y = x.tolist()
        if isinstance(y, list):
            return y
        return [y]
    return [x]


def get_first_present(mapping: Dict[str, Any], keys: List[str], default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def normalize_2wiki_record(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)

    if "_id" not in out:
        out["_id"] = str(out.get("id", ""))

    sf = out.get("supporting_facts", {})

    if isinstance(sf, dict):
        titles = to_python_list(get_first_present(sf, ["title", "titles"], []))
        sent_ids = to_python_list(
            get_first_present(
                sf, ["sent_id", "sent_ids", "sentence_id", "sent_idx"], []
            )
        )

        supporting_facts = []
        for title, idx in zip(titles, sent_ids):
            try:
                supporting_facts.append([str(title), int(idx)])
            except Exception:
                continue

        out["supporting_facts"] = supporting_facts

    elif isinstance(sf, list):
        supporting_facts = []
        for item in sf:
            if isinstance(item, dict):
                title = item.get("title", "")
                idx = item.get(
                    "sent_id", item.get("sentence_id", item.get("sent_idx", 0))
                )
                try:
                    supporting_facts.append([str(title), int(idx)])
                except Exception:
                    continue
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                title, idx = item
                try:
                    supporting_facts.append([str(title), int(idx)])
                except Exception:
                    continue
        out["supporting_facts"] = supporting_facts

    else:
        out["supporting_facts"] = []

    return out


def load_2wiki_file(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
        return [normalize_2wiki_record(r) for r in df.to_dict(orient="records")]

    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        return [normalize_2wiki_record(r) for r in rows]

    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(normalize_2wiki_record(json.loads(line)))
        return rows

    raise ValueError(f"Unsupported file type: {p}")


def get_2wiki_id(example_id: str) -> str:
    """
    Converts:
      2WikiMultiHopQA__test__13f5...
    into:
      13f5...
    """
    prefix = "2WikiMultiHopQA__test__"
    if example_id.startswith(prefix):
        return example_id[len(prefix) :]

    parts = example_id.split("__")
    if len(parts) >= 3 and parts[0] == "2WikiMultiHopQA":
        return parts[-1]

    return example_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--details", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with open(args.details, "r", encoding="utf-8") as f:
        details = json.load(f)

    wanted_ids = {get_2wiki_id(item["example_id"]) for item in details}

    records = load_2wiki_file(args.parquet)

    subset = []
    for r in records:
        rid = str(r.get("_id") or r.get("id"))
        if rid in wanted_ids:
            subset.append(
                {
                    "_id": rid,
                    "answer": str(r.get("answer", "")),
                    "supporting_facts": r.get("supporting_facts", []),
                }
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False)

    print(f"Wanted ids: {len(wanted_ids)}")
    print(f"Saved subset examples: {len(subset)}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
