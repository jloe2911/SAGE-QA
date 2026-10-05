"""Verify the optimized retrieval overlay from retained frozen reader supports.

This checker never reconstructs the missing optimized ranking artifact.  It joins
the already-frozen reader inputs to the canonical annotated support alternatives
and applies the repository's official ``best_support_scores`` evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_owl_qa_predictions import best_support_scores
from utils.paths import repo_root


TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _require_close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > TOLERANCE:
        raise ValueError(f"{label}: expected {expected}, got {actual}")


def verify_overlay(asset_root: Path, overlay_path: Path) -> dict[str, Any]:
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    sources = overlay["source_artifacts"]
    inputs_record = sources["frozen_reader_inputs"]
    gold_record = sources["support_gold"]
    metrics_record = sources["optimized_end_to_end_metrics"]

    def source_path(record: Mapping[str, str]) -> Path:
        path = asset_root / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Required external artifact is absent: {path}")
        actual = sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual}")
        return path

    inputs_path = source_path(inputs_record)
    gold_path = source_path(gold_record)
    metrics_path = source_path(metrics_record)
    base_path = asset_root / overlay["base_export"]["path"]
    if not base_path.is_file():
        raise FileNotFoundError(f"Required external artifact is absent: {base_path}")
    if sha256(base_path) != overlay["base_export"]["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {base_path}")

    inputs = load_jsonl(inputs_path)
    gold_rows = load_jsonl(gold_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    gold_by_id = {row["example_id"]: row for row in gold_rows}
    if len(gold_by_id) != overlay["cohort"]["prediction_population"]:
        raise ValueError("Unexpected support-gold population")

    values: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    seen: set[tuple[str, str]] = set()
    for row in inputs:
        key = (str(row["example_id"]), str(row["setting"]))
        if key in seen:
            raise ValueError(f"Duplicate frozen input: {key}")
        seen.add(key)
        alternatives = gold_by_id[key[0]].get("gold_explanations") or []
        if not alternatives:
            continue
        score = best_support_scores(row["selected_support"], alternatives)
        dataset, setting = str(row["dataset"]), key[1]
        values[setting][dataset]["precision"].append(float(score["prec"]))
        values[setting][dataset]["recall"].append(float(score["recall"]))
        values[setting][dataset]["f1"].append(float(score["f1"]))

    expected_rows = overlay["cohort"]["prediction_population"] * 2
    if len(inputs) != expected_rows or len(seen) != expected_rows:
        raise ValueError("Frozen reader-input population is incomplete")

    methods = overlay["methods"]["sageqa_optimized"]
    checked = 0
    for setting, dataset_values in values.items():
        recorded = methods[setting]
        for dataset, fields in dataset_values.items():
            by_dataset = recorded["by_dataset"][dataset]
            if len(fields["f1"]) != by_dataset["n"]:
                raise ValueError(f"Eligible-count mismatch for {setting}/{dataset}")
            for field, samples in fields.items():
                _require_close(fmean(samples), by_dataset[field], f"{setting}/{dataset}/{field}")
            checked += len(fields["f1"])
        for field in ("precision", "recall", "f1"):
            dataset_means = [fmean(fields[field]) for fields in dataset_values.values()]
            _require_close(
                fmean(dataset_means),
                recorded["equal_dataset_macro"][field],
                f"{setting}/equal_dataset_macro/{field}",
            )
        _require_close(
            recorded["equal_dataset_macro"]["f1"],
            metrics["primary_equal_dataset_macro"][setting]["support_f1"],
            f"{setting}/end_to_end_support_f1",
        )
        for dataset, cells in recorded["by_dataset"].items():
            _require_close(
                cells["f1"],
                metrics["by_dataset"][dataset][setting]["support_f1"],
                f"{setting}/{dataset}/end_to_end_support_f1",
            )

    eligible = overlay["cohort"]["included"]
    if checked != eligible * 2:
        raise ValueError(f"Expected {eligible * 2} eligible setting rows, checked {checked}")
    return {
        "status": "verified_derived_overlay",
        "eligible_examples": eligible,
        "settings": sorted(values),
        "checked_setting_rows": checked,
        "raw_selection_artifact_recovered": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=repo_root())
    parser.add_argument(
        "--overlay",
        type=Path,
        default=repo_root()
        / "release_manifests/thesis_optimized/canonical_retrieval_overlay.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(verify_overlay(args.asset_root.resolve(), args.overlay.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
