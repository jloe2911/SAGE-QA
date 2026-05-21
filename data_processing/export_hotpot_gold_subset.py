import argparse
import json
from pathlib import Path

import pandas as pd


def first_present(mapping, keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def normalize_hotpot_record(row):
    out = dict(row)

    if "_id" not in out and "id" in out:
        out["_id"] = out["id"]

    context = out.get("context")
    if isinstance(context, dict):
        titles = first_present(context, ["title", "titles"], [])
        sentences = first_present(context, ["sentences", "sentence"], [])
        out["context"] = [[t, list(s)] for t, s in zip(titles, sentences)]

    sf = out.get("supporting_facts")
    if isinstance(sf, dict):
        titles = first_present(sf, ["title", "titles"], [])
        sent_ids = first_present(
            sf, ["sent_id", "sent_ids", "sentence_id", "sent_idx"], []
        )
        out["supporting_facts"] = [[t, int(i)] for t, i in zip(titles, sent_ids)]

    return out


def get_hotpot_id(example_id):
    prefix = "HotpotQA__test__"
    if example_id.startswith(prefix):
        return example_id[len(prefix) :]
    parts = example_id.split("__")
    if len(parts) >= 3 and parts[0] == "HotpotQA":
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

    wanted_ids = {get_hotpot_id(item["example_id"]) for item in details}

    df = pd.read_parquet(args.parquet)
    records = [normalize_hotpot_record(r) for r in df.to_dict(orient="records")]

    subset = [r for r in records if str(r.get("_id")) in wanted_ids]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False)

    print(f"Wanted ids: {len(wanted_ids)}")
    print(f"Saved subset examples: {len(subset)}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
