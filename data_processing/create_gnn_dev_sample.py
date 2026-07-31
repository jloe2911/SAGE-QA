"""Materialize a small, reproducible GNN retrieval benchmark.

The retrieval JSONL files contain one row per candidate, not one row per
question. Sampling rows directly would therefore corrupt the ranking task.
This utility selects question/example IDs and copies *all* candidate rows for
each selected example into a compact, persistent dataset directory.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_COUNTS = {"train": 64, "dev": 32, "test": 32}
DEFAULT_STRATIFY_FIELDS = ("hop", "answer_type", "task_type")


def stable_hash(seed: int, split: str, example_id: str) -> str:
    value = f"{seed}:{split}:{example_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def leakage_key(example_id: str) -> str:
    """Remove a builder-added split marker to expose reused source records."""
    return re.sub(r"__(?:train|dev|test)__", "__", example_id, count=1)


def normalized_stratum(
    row: Dict[str, Any], fields: Sequence[str] = DEFAULT_STRATIFY_FIELDS
) -> Tuple[str, ...]:
    values = []
    for field in fields:
        value = row.get(field)
        values.append(str(value).strip() if value not in (None, "") else "unknown")
    return tuple(values)


def discover_examples(
    path: Path, stratify_fields: Sequence[str]
) -> Dict[str, Tuple[str, ...]]:
    examples: Dict[str, Tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row.get("example_id", "")).strip()
            if not example_id:
                raise ValueError(f"Missing example_id in {path}:{line_number}")
            stratum = normalized_stratum(row, stratify_fields)
            previous = examples.setdefault(example_id, stratum)
            if previous != stratum:
                raise ValueError(
                    f"Inconsistent stratum for {example_id!r} in {path}: "
                    f"{previous!r} versus {stratum!r}"
                )
    return examples


def proportional_quotas(
    stratum_sizes: Dict[Tuple[str, ...], int], requested: int
) -> Dict[Tuple[str, ...], int]:
    """Allocate a representative sample with deterministic largest remainders."""
    total = sum(stratum_sizes.values())
    target = min(max(requested, 0), total)
    if not target or not total:
        return {key: 0 for key in stratum_sizes}

    exact = {key: target * size / total for key, size in stratum_sizes.items()}
    quotas = {
        key: min(size, int(math.floor(exact[key])))
        for key, size in stratum_sizes.items()
    }

    # When possible, retain at least one example from every observed stratum.
    if target >= len(stratum_sizes):
        for key, size in stratum_sizes.items():
            if size and quotas[key] == 0:
                quotas[key] = 1

    while sum(quotas.values()) > target:
        removable = [
            key
            for key in quotas
            if quotas[key] > (1 if target >= len(stratum_sizes) else 0)
        ]
        key = min(
            removable,
            key=lambda item: (exact[item] - quotas[item], str(item)),
        )
        quotas[key] -= 1

    while sum(quotas.values()) < target:
        available = [key for key, size in stratum_sizes.items() if quotas[key] < size]
        key = max(
            available,
            key=lambda item: (exact[item] - quotas[item], str(item)),
        )
        quotas[key] += 1

    return quotas


def select_examples(
    examples: Dict[str, Tuple[str, ...]],
    requested: int,
    seed: int,
    split: str,
) -> Tuple[List[str], Dict[Tuple[str, ...], int]]:
    by_stratum: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for example_id, stratum in examples.items():
        by_stratum[stratum].append(example_id)

    quotas = proportional_quotas(
        {key: len(ids) for key, ids in by_stratum.items()}, requested
    )
    selected = []
    for stratum, ids in sorted(by_stratum.items()):
        ordered = sorted(ids, key=lambda value: stable_hash(seed, split, value))
        selected.extend(ordered[: quotas[stratum]])
    selected.sort(key=lambda value: stable_hash(seed, split, value))
    return selected, quotas


def materialize_rows(
    source: Path, destination: Path, selected_ids: Iterable[str]
) -> int:
    selected = set(selected_ids)
    row_count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with (
        source.open("r", encoding="utf-8") as input_handle,
        temporary.open("w", encoding="utf-8", newline="\n") as output_handle,
    ):
        for line in input_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("example_id", "")) in selected:
                output_handle.write(line if line.endswith("\n") else line + "\n")
                row_count += 1
    temporary.replace(destination)
    return row_count


def stratum_label(fields: Sequence[str], values: Tuple[str, ...]) -> str:
    return " | ".join(f"{field}={value}" for field, value in zip(fields, values))


def build_sample(
    input_dir: Path,
    output_dir: Path,
    counts: Dict[str, int],
    seed: int,
    stratify_fields: Sequence[str],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifests: Dict[str, Any] = {}
    all_selected: Dict[str, str] = {}

    for split in ("train", "dev", "test"):
        source = input_dir / f"{split}_subgraph_retrieval.jsonl"
        if not source.exists():
            raise FileNotFoundError(f"Missing retrieval split: {source}")

        examples = discover_examples(source, stratify_fields)
        eligible_examples = {
            example_id: stratum
            for example_id, stratum in examples.items()
            if leakage_key(example_id) not in all_selected
        }
        selected, quotas = select_examples(
            examples=eligible_examples,
            requested=counts[split],
            seed=seed,
            split=split,
        )
        for example_id in selected:
            key = leakage_key(example_id)
            previous_split = all_selected.setdefault(key, split)
            if previous_split != split:
                raise ValueError(
                    f"Source-record leakage: {example_id!r} occurs in both "
                    f"{previous_split} and {split}"
                )

        destination = output_dir / source.name
        row_count = materialize_rows(source, destination, selected)
        selected_counts = Counter(examples[example_id] for example_id in selected)
        split_manifests[split] = {
            "source": str(source),
            "source_bytes": source.stat().st_size,
            "available_examples": len(examples),
            "eligible_after_leakage_filter": len(eligible_examples),
            "requested_examples": counts[split],
            "selected_examples": len(selected),
            "selected_rows": row_count,
            "selected_example_ids": selected,
            "available_strata": {
                stratum_label(stratify_fields, key): value
                for key, value in sorted(Counter(examples.values()).items())
            },
            "selected_strata": {
                stratum_label(stratify_fields, key): value
                for key, value in sorted(selected_counts.items())
            },
            "stratum_quotas": {
                stratum_label(stratify_fields, key): value
                for key, value in sorted(quotas.items())
            },
        }
        print(
            f"{split}: selected {len(selected)}/{len(examples)} examples "
            f"and {row_count} candidate rows"
        )

    manifest = {
        "sample_schema_version": "gnn_dev_sample_v1",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "sampling_unit": "example_id",
        "leakage_key_policy": "strip_first_train_dev_test_marker",
        "candidate_policy": "all_rows_for_selected_examples",
        "stratify_fields": list(stratify_fields),
        "splits": split_manifests,
    }
    with (output_dir / "sample_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    source_metadata = input_dir / "metadata.json"
    metadata: Dict[str, Any] = {}
    if source_metadata.exists():
        with source_metadata.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    metadata["development_sample"] = {
        "schema_version": manifest["sample_schema_version"],
        "manifest": "sample_manifest.json",
        "seed": seed,
        "stratify_fields": list(stratify_fields),
        "example_counts": {
            split: details["selected_examples"]
            for split, details in split_manifests.items()
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select persistent example-level train/dev/test slices while retaining "
            "the complete candidate pool for every selected question."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-examples", type=int, default=DEFAULT_COUNTS["train"])
    parser.add_argument("--dev-examples", type=int, default=DEFAULT_COUNTS["dev"])
    parser.add_argument("--test-examples", type=int, default=DEFAULT_COUNTS["test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-by",
        default=",".join(DEFAULT_STRATIFY_FIELDS),
        help="Comma-separated row fields used for proportional stratification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fields = tuple(
        field.strip() for field in args.stratify_by.split(",") if field.strip()
    )
    if not fields:
        raise ValueError("--stratify-by must contain at least one field")
    counts = {
        "train": args.train_examples,
        "dev": args.dev_examples,
        "test": args.test_examples,
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError("Sample counts must be non-negative")
    build_sample(args.input_dir, args.output_dir, counts, args.seed, fields)


if __name__ == "__main__":
    main()
