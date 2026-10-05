"""Reproduce reviewer-facing paper and thesis tables from restored frozen artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


COMPARE_FILES = ("canonical_metrics.json", "per_dataset.csv", "main_table.tex", "ablation_table.tex")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_identical(generated: Path, frozen: Path) -> None:
    if sha256(generated) != sha256(frozen):
        raise ValueError(f"Generated table product differs from frozen result: {generated}")


def reproduce_paper(root: Path, output: Path) -> dict[str, Any]:
    source = root / "outputs/full_results"
    if not source.is_dir():
        raise FileNotFoundError("Restore paper_table_artifacts.zip before reproducing tables")
    output.mkdir(parents=True, exist_ok=False)
    copied = []
    for name in (
        "full_pipeline_results.csv",
        "full_pipeline_results.json",
        "sageqa_retrieval_prf_deduplicated_union_k1_k2_k3_k5.csv",
    ):
        shutil.copyfile(source / name, output / name)
        copied.append(name)
    return {
        "status": "paper_reported_tables_materialized_from_verified_frozen_outputs",
        "files": copied,
        "historical_protocol": "gold_informed_candidate_generation",
    }


def reproduce_thesis(root: Path, output: Path) -> dict[str, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    output.mkdir(parents=True, exist_ok=False)
    retrieval_output = output / "retrieval_hard_pair_v2"
    subprocess.run(
        [
            sys.executable,
            str(root / "evaluation/export_manuscript_retrieval_results.py"),
            "--output-dir",
            str(retrieval_output),
        ],
        cwd=root,
        check=True,
    )
    frozen_retrieval = root / "outputs/final_results/manuscript_retrieval_results_hard_pair_v2"
    for name in COMPARE_FILES:
        require_identical(retrieval_output / name, frozen_retrieval / name)

    from evaluation import finalize_final_manuscript_end_to_end as finalizer

    frozen_end = root / "outputs/final_results/final_manuscript_test_end_to_end"
    end_output = output / "end_to_end_hard_pair_v2"
    end_output.mkdir()
    shutil.copyfile(frozen_end / "metrics.json", end_output / "metrics.json")
    shutil.copyfile(
        frozen_end / "per_example_end_to_end.jsonl", end_output / "per_example_end_to_end.jsonl"
    )
    finalizer.OUTPUT = end_output
    finalizer.NEW_METRICS = end_output / "metrics.json"
    finalizer.NEW_PER_EXAMPLE = end_output / "per_example_end_to_end.jsonl"
    finalizer.OLD_METRICS = (
        root / "outputs/final_results/production_generator_d_v2_hard_pair_test_end_to_end/metrics.json"
    )
    finalizer.BASELINE_METRICS = (
        root / "outputs/final_results/final_manuscript_baselines_test_end_to_end/metrics.json"
    )
    finalizer.PREDICTIONS = frozen_end / "predictions.jsonl"
    finalizer.READER_INPUTS = frozen_end / "frozen_reader_inputs.jsonl"
    finalizer.GENERATION_FREEZE = frozen_end / "generation_freeze.json"
    finalizer.PREFLIGHT = frozen_end / "preflight_report.json"
    if finalizer.main() != 0:
        raise RuntimeError("Thesis end-to-end finalizer failed")
    for name in COMPARE_FILES:
        require_identical(end_output / name, frozen_end / name)

    from evaluation.verify_canonical_retrieval_overlay import verify_overlay

    overlay_path = root / "release_manifests/thesis_optimized/canonical_retrieval_overlay.json"
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    overlay_report = verify_overlay(root, overlay_path)
    optimized = output / "optimized_overlay"
    optimized.mkdir()
    with (optimized / "retrieval_per_dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("dataset", "setting", "n", "precision", "recall", "f1")
        )
        writer.writeheader()
        methods = overlay["methods"]["sageqa_optimized"]
        for setting in ("k1", "adaptive"):
            for dataset, cells in methods[setting]["by_dataset"].items():
                writer.writerow({"dataset": dataset, "setting": setting, **cells})
    with (optimized / "retrieval_overall.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("setting", "aggregation", "precision", "recall", "f1")
        )
        writer.writeheader()
        for setting in ("k1", "adaptive"):
            for aggregation in ("equal_dataset_macro", "pooled_per_example_mean"):
                writer.writerow(
                    {"setting": setting, "aggregation": aggregation, **methods[setting][aggregation]}
                )
    source_metrics = root / overlay["source_artifacts"]["optimized_end_to_end_metrics"]["path"]
    shutil.copyfile(source_metrics, optimized / "end_to_end_metrics.json")
    (optimized / "verification.json").write_text(
        json.dumps(overlay_report, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "status": "thesis_tables_reproduced_and_matched_frozen_outputs",
        "hard_pair_v2_products_byte_identical": True,
        "optimized_overlay": overlay_report,
    }


def output_manifest(output: Path, reports: dict[str, Any]) -> None:
    entries = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "REPRODUCTION_MANIFEST.json":
            continue
        entries.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": "sageqa_reproduced_tables_v1",
        "reports": reports,
        "files": entries,
    }
    (output / "REPRODUCTION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("reproduced_tables"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.repo_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing reproduction output: {output}")
    output.mkdir(parents=True)
    reports = {
        "paper": reproduce_paper(root, output / "paper"),
        "thesis": reproduce_thesis(root, output / "thesis"),
    }
    output_manifest(output, reports)
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
