"""Report canonical thesis-final path resolution without mutating the checkout."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import (  # noqa: E402
    cross_encoder_checkpoint,
    cross_encoder_policy_root,
    cross_encoder_test_root,
    final_answer_root,
    final_sageqa_policy_root,
    generator_d_root,
    manuscript_retrieval_root,
    repo_root,
)


PROTOCOL = "thesis_final"
LOGICAL_BUNDLE = "thesis_final_generator_d_cross_encoder_adaptive_answers"


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "kind": "file" if path.is_file() else "directory" if path.is_dir() else "missing"}


def resolution_report() -> dict[str, Any]:
    root = repo_root()
    index_path = root / "release_manifests/thesis_final/index.yaml"
    release = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    bundle = next(item for item in release["bundles"] if item["logical_name"] == LOGICAL_BUNDLE)
    return {
        "protocol": PROTOCOL,
        "logical_bundle": LOGICAL_BUNDLE,
        "repository_root": _record(root),
        "generator_d_root": _record(generator_d_root()),
        "cross_encoder_checkpoint": _record(cross_encoder_checkpoint()),
        "adaptive_policy": {
            "cross_encoder": _record(cross_encoder_policy_root()),
            "final_sageqa": _record(final_sageqa_policy_root()),
        },
        "frozen_test_selection": _record(cross_encoder_test_root() / "test_predictions_frozen.jsonl"),
        "retrieval_export": _record(manuscript_retrieval_root()),
        "answer_bundle": _record(final_answer_root()),
        "release_manifest": {
            "path": str(index_path),
            "source_revision": bundle["source_revision"],
            "status": bundle["status"],
            "known_sha256": bundle["known_sha256"],
        },
        "mutated_files": 0,
    }


def main() -> int:
    print(json.dumps(resolution_report(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
