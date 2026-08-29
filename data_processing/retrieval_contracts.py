"""Shared leakage guards and provenance helpers for retrieval dataset builders."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple


BUILDER_VERSION = "generator_d_production_v1"
CONTAMINATED_KG_MARKERS = ("provided", "provided_plus_llm", "llm_with_provided")
GOLD_DERIVED_FIELDS = frozenset(
    {
        "answer",
        "supporting_facts",
        "evidences",
        "raw_evidences",
        "gold_explanations",
        "gold_units",
        "gold_support_units",
        "raw_supporting_facts",
        "label",
        "rank_target",
        "best_set_f1_to_gold",
        "best_jaccard_to_gold",
        "exact_match_any_gold",
        "contains_any_gold_explanation",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(repo_root: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "working_tree_clean": not bool(status),
        "working_tree_status": status.splitlines(),
    }


def clean_text_retrieval_input(example: Mapping[str, Any]) -> Dict[str, Any]:
    """Project a text benchmark row onto the only fields generation may read."""
    raw_id = str(example.get("_id") or example.get("id") or "")
    return {
        "id": raw_id,
        "question": str(example.get("question") or ""),
        "context": example.get("context") or [],
    }


def assert_gold_free_mapping(mapping: Mapping[str, Any], *, stage: str) -> None:
    forbidden = sorted(GOLD_DERIVED_FIELDS.intersection(mapping))
    if forbidden:
        raise ValueError(f"{stage} received forbidden gold-derived fields: {forbidden}")


def validate_clean_kg_backend(backend: str) -> None:
    lowered = str(backend or "").lower()
    if any(marker in lowered for marker in CONTAMINATED_KG_MARKERS):
        raise ValueError(
            f"Contaminated KG backend {backend!r} is forbidden in gold-free retrieval."
        )


def validate_clean_cache_row(
    row: Mapping[str, Any], *, expected_signature: str | None = None
) -> None:
    method = str(
        row.get("construction_method")
        or row.get("kg_construction_backend")
        or row.get("backend")
        or ""
    ).lower()
    if any(marker in method for marker in CONTAMINATED_KG_MARKERS):
        raise ValueError(
            f"Contaminated KG cache provenance {method!r} cannot be reused."
        )
    if row.get("gold_available_during_candidate_generation") is True:
        raise ValueError("KG cache reports gold access during candidate generation.")
    if expected_signature is not None:
        actual = row.get("cache_signature")
        if actual != expected_signature:
            raise ValueError(
                "KG cache signature mismatch: "
                f"expected {expected_signature!r}, found {actual!r}."
            )


def cap_inference_candidate_rows(
    rows: Sequence[Mapping[str, Any]], max_candidates: int = 320
) -> List[Mapping[str, Any]]:
    """Deterministically cap inference candidates without reading supervision."""
    if max_candidates <= 0 or len(rows) <= max_candidates:
        return list(rows)

    def key(row: Mapping[str, Any]):
        pre_rank = float(row.get("candidate_pre_rank_score", 0.0))
        generation_rank = int(row.get("generation_rank", 2**31 - 1))
        units = tuple(str(unit) for unit in row.get("subgraph_units", []) or [])
        return (-pre_rank, generation_rank, len(units), units)

    return sorted(rows, key=key)[:max_candidates]


def stable_example_id(example: Mapping[str, Any]) -> str:
    return str(example.get("_id") or example.get("id") or "")


def assert_disjoint_split_ids(
    split_examples: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    id_fn: Callable[[Mapping[str, Any]], str] = stable_example_id,
) -> None:
    """Assert pairwise-disjoint train/dev/test IDs before generation starts."""
    ids = {
        split: {id_fn(example) for example in examples}
        for split, examples in split_examples.items()
    }
    if any("" in split_ids for split_ids in ids.values()):
        raise ValueError("Every split example must have a non-empty ID.")
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = sorted(ids.get(left, set()).intersection(ids.get(right, set())))
        if overlap:
            raise AssertionError(f"{left}/{right} ID overlap: {overlap[:5]}")


def select_disjoint_cohorts(
    dev_examples: Sequence[Dict[str, Any]],
    test_examples: Sequence[Dict[str, Any]],
    *,
    max_dev_examples: int,
    max_test_examples: int,
    seed: int,
    id_fn: Callable[[Mapping[str, Any]], str] = stable_example_id,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select deterministic dev/test cohorts and exclude every overlapping ID."""
    dev_unique = {id_fn(example): example for example in dev_examples}
    test_unique = {id_fn(example): example for example in test_examples}
    if "" in dev_unique or "" in test_unique:
        raise ValueError("Every text example must have a non-empty ID.")

    dev_ids = sorted(dev_unique)
    test_ids = sorted(test_unique)
    random.Random(seed + 100_000).shuffle(dev_ids)
    random.Random(seed + 200_000).shuffle(test_ids)

    if max_dev_examples > 0:
        dev_ids = dev_ids[:max_dev_examples]
    selected_dev = set(dev_ids)
    test_ids = [example_id for example_id in test_ids if example_id not in selected_dev]
    if max_test_examples > 0:
        test_ids = test_ids[:max_test_examples]

    overlap = selected_dev.intersection(test_ids)
    if overlap:
        raise AssertionError(f"dev/test ID overlap: {sorted(overlap)[:5]}")
    return (
        [dev_unique[example_id] for example_id in dev_ids],
        [test_unique[example_id] for example_id in test_ids],
    )


def split_manifest(
    *,
    dataset: str,
    source_paths: Mapping[str, Iterable[Path]],
    split_examples: Mapping[str, Sequence[Mapping[str, Any]]],
    seed: int,
    id_fn: Callable[[Mapping[str, Any]], str] = stable_example_id,
) -> Dict[str, Any]:
    ids = {
        split: [id_fn(example) for example in examples]
        for split, examples in split_examples.items()
    }
    assert_disjoint_split_ids(split_examples, id_fn=id_fn)
    overlaps = {"train_dev": [], "train_test": [], "dev_test": []}
    return {
        "builder_version": BUILDER_VERSION,
        "dataset": dataset,
        "seed": seed,
        "source_files": {
            split: [
                {"path": str(path), "sha256": sha256_file(path)} for path in paths
            ]
            for split, paths in source_paths.items()
        },
        "splits": {
            split: {"example_count": len(split_ids), "example_ids": split_ids}
            for split, split_ids in ids.items()
        },
        "split_overlaps": overlaps,
        "dev_test_overlap": overlaps["dev_test"],
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
