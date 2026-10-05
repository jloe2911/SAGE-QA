"""Build compact, checksummed reviewer packages for frozen table reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence


PAPER_PATHS = (
    "outputs/full_results",
)

THESIS_PATHS = (
    "outputs/final_results/production_generator_d_v2_hard_pair_test_retrieval",
    "outputs/final_results/production_generator_d_v1_test_baselines/lexical_subgraph",
    "outputs/final_results/production_generator_d_v1_test_baselines/gnn_rag/predictions_frozen.jsonl",
    "outputs/final_results/production_generator_d_v1_test_baselines/gnn_rag/prediction_freeze_manifest.json",
    "outputs/final_results/question_candidate_cross_encoder_v1_adaptive_test_a40",
    "outputs/development_diagnostics/final_old_vs_cross_encoder_analysis/comparison.json",
    "outputs/final_results/manuscript_retrieval_results_hard_pair_v2",
    "outputs/final_results/final_manuscript_test_end_to_end",
    "outputs/final_results/production_generator_d_v2_hard_pair_test_end_to_end",
    "outputs/final_results/final_manuscript_baselines_test_end_to_end",
    "outputs/final_results/final_manuscript_stagewise_error_analysis",
    "outputs/final_results/symbolic_coefficients_original_v1_test_end_to_end",
    "outputs/final_results/symbolic_coefficients_original_v1_stagewise_error_analysis",
)

PACKAGES = {
    "paper": {
        "filename": "paper_table_artifacts.zip",
        "lineage": "original_published_paper_historical_protocol",
        "paths": PAPER_PATHS,
        "limitations": [
            "Exact historical source, environment, and hosted-reader binding remains unverified.",
            "The historical candidate-generation protocol was gold-informed and is not the thesis protocol.",
        ],
    },
    "thesis": {
        "filename": "thesis_table_artifacts.zip",
        "lineage": "final_leakage_free_thesis_with_hard_pair_v2_and_optimized_overlay",
        "paths": THESIS_PATHS,
        "limitations": [
            "The optimized raw TEST ranking and selected DEV coefficient/policy bundle are missing.",
            "Optimized official metrics are replayed from retained frozen selected supports, not reconstructed rankings.",
        ],
    },
}

FIXED_ZIP_TIME = (2026, 10, 5, 0, 0, 0)
MANIFEST_NAME = "REVIEWER_ARTIFACT_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_for(source_root: Path, declared: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for relative in declared:
        path = source_root / relative
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(item for item in path.rglob("*") if item.is_file())
        else:
            raise FileNotFoundError(f"Required frozen artifact is absent: {path}")
    return sorted(files, key=lambda item: item.relative_to(source_root).as_posix())


def manifest_for(kind: str, source_root: Path, files: Sequence[Path]) -> dict[str, Any]:
    spec = PACKAGES[kind]
    entries = []
    for path in files:
        entries.append(
            {
                "path": path.relative_to(source_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "schema_version": "sageqa_reviewer_table_artifacts_v1",
        "package": spec["filename"],
        "lineage": spec["lineage"],
        "purpose": "frozen_artifacts_required_to_reproduce_reported_result_tables",
        "file_count": len(entries),
        "uncompressed_bytes": sum(entry["size_bytes"] for entry in entries),
        "files": entries,
        "limitations": spec["limitations"],
    }


def write_member(archive: zipfile.ZipFile, relative: str, source: Path) -> None:
    info = zipfile.ZipInfo(relative, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    with source.open("rb") as input_handle, archive.open(info, "w", force_zip64=True) as output:
        for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
            output.write(chunk)


def build(kind: str, source_root: Path, output_dir: Path, manifest_dir: Path, overwrite: bool) -> dict[str, Any]:
    spec = PACKAGES[kind]
    files = files_for(source_root, spec["paths"])
    manifest = manifest_for(kind, source_root, files)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / spec["filename"]
    if archive_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing package: {archive_path}")
    with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
        for path in files:
            write_member(archive, path.relative_to(source_root).as_posix(), path)
        info = zipfile.ZipInfo(MANIFEST_NAME, FIXED_ZIP_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    sidecar = {
        **manifest,
        "archive_path": f"release/reviewer_artifacts/{archive_path.name}",
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "publication_status": "local_supplied_artifact_not_uploaded",
        "download_url": None,
    }
    sidecar_path = manifest_dir / f"{archive_path.stem}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return sidecar


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("release/reviewer_artifacts"))
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("release_manifests/reviewer_artifacts")
    )
    parser.add_argument("--package", choices=("paper", "thesis", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    kinds = tuple(PACKAGES) if args.package == "all" else (args.package,)
    for kind in kinds:
        result = build(
            kind,
            args.source_root.resolve(),
            args.output_dir.resolve(),
            args.manifest_dir.resolve(),
            args.overwrite,
        )
        print(json.dumps({key: result[key] for key in ("package", "file_count", "archive_size_bytes", "archive_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
