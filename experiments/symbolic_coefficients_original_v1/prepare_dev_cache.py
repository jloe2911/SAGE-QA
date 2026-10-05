"""Join frozen original DEV logits to original candidates and cache features."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.symbolic_coefficients_original_v1.original_symbolic_features import (
    ONTOLOGY_DATASETS,
    TEXT_DATASETS,
    candidate_identity,
    proof_features,
    text_features,
)


SCHEMA_VERSION = "sageqa_symbolic_coefficients_original_v1_cache"
DATASETS = tuple(sorted(TEXT_DATASETS)) + tuple(sorted(ONTOLOGY_DATASETS))
MAX_CANDIDATES = 320
EXPECTED_SOURCE_MANIFEST_SHA256 = "624d5941d96115aadbc4d23a11f4d65d14d4cb29e0acdd1a65653a4039754bf1"
EXPECTED_PREDICTION_FREEZE_SHA256 = "92b1b94886408e2b510d106de795de36cf48e2dec481a99db44b4632810f69d7"
EXPECTED_PREDICTIONS_SHA256 = "7d5a6e9c0afbc9b7e13669da27c51fdff559c35b64781e8f0fdbc52f68172e34"
EXPECTED_CHECKPOINT_SHA256 = "02b06c91fa422369a0b0adbab3b33bfcf80f335b7c8a5911084c3b82af117ba9"


def assert_dev_only_path(path: Path) -> None:
    tokens: list[str] = []
    for part in path.parts:
        tokens.extend(token for token in re.split(r"[^a-z0-9]+", part.lower()) if token)
    if "test" in tokens:
        raise ValueError(f"TEST path is forbidden: {path}")


def sha256(path: Path) -> str:
    assert_dev_only_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def read_json(path: Path) -> Any:
    assert_dev_only_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    assert_dev_only_path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def cap_candidates(rows: Sequence[Mapping[str, Any]], limit: int = MAX_CANDIDATES) -> list[Mapping[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("candidate_pre_rank_score", 0.0)),
            int(row.get("generation_rank", 2**31 - 1)),
            len(row.get("subgraph_units", []) or []),
            tuple(str(unit) for unit in row.get("subgraph_units", []) or []),
        ),
    )[:limit]


def grouped_rows(path: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    seen: set[str] = set()
    current_id: str | None = None
    current_rows: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        example_id = str(row.get("example_id") or "")
        if not example_id:
            raise ValueError(f"{path}: candidate missing example_id")
        if current_id is None:
            current_id = example_id
        if example_id != current_id:
            if example_id in seen:
                raise ValueError(f"{path}: non-contiguous example {example_id!r}")
            seen.add(current_id)
            yield current_id, current_rows
            current_id, current_rows = example_id, []
        current_rows.append(row)
    if current_id is not None:
        yield current_id, current_rows


def gold_explanations(row: Mapping[str, Any], domain: str) -> list[list[str]]:
    if domain == "text":
        values = list(row.get("gold_support_units", []) or [])
        return [values] if values else []
    explicit = [list(value) for value in row.get("gold_explanations", []) or [] if value]
    if explicit:
        return explicit
    values = list(row.get("gold_units", []) or [])
    return [values] if values else []


def expected_inputs(data_root: Path) -> dict[str, Path]:
    return {dataset: data_root / dataset / "dev_subgraph_retrieval.jsonl" for dataset in DATASETS}


def validate_inputs(predictions: Path, freeze_path: Path, data_root: Path, source_manifest_path: Path) -> dict[str, Any]:
    for path in (predictions, freeze_path, data_root, source_manifest_path):
        assert_dev_only_path(path)
    freeze_hash = sha256(freeze_path)
    source_manifest_hash = sha256(source_manifest_path)
    if freeze_hash != EXPECTED_PREDICTION_FREEZE_SHA256:
        raise ValueError("Original prediction-freeze manifest hash mismatch")
    if source_manifest_hash != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("Original Cross-Encoder preflight manifest hash mismatch")
    freeze = read_json(freeze_path)
    source = read_json(source_manifest_path)
    if freeze.get("split") != "dev" or freeze.get("test_accessed") is not False:
        raise ValueError("Prediction freeze is not an explicit DEV-only freeze")
    prediction_hash = sha256(predictions)
    if prediction_hash != EXPECTED_PREDICTIONS_SHA256 or prediction_hash != freeze.get("predictions_sha256"):
        raise ValueError("Frozen Cross-Encoder prediction hash mismatch")
    if freeze.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Frozen Cross-Encoder checkpoint identity mismatch")
    expected_hashes = source.get("input_sha256") or {}
    actual: dict[str, str] = {}
    for dataset, path in expected_inputs(data_root).items():
        if not path.is_file():
            raise FileNotFoundError(path)
        matching = [value for key, value in expected_hashes.items() if key.replace("\\", "/").endswith(f"/{dataset}/dev_subgraph_retrieval.jsonl")]
        if len(matching) != 1:
            raise ValueError(f"Source manifest must contain exactly one DEV hash for {dataset}")
        actual[str(path)] = sha256(path)
        if actual[str(path)] != matching[0]:
            raise ValueError(f"Frozen DEV candidate hash mismatch for {dataset}")
    return {
        "predictions_sha256": prediction_hash,
        "prediction_freeze_sha256": freeze_hash,
        "source_manifest_sha256": source_manifest_hash,
        "candidate_sha256": actual,
    }


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        example_id = str(row.get("example_id") or "")
        if not example_id or example_id in result:
            raise ValueError(f"Missing or duplicate frozen prediction identity: {example_id!r}")
        if row.get("domain") not in {"text", "ontology"}:
            raise ValueError(f"{example_id}: invalid prediction domain")
        result[example_id] = row
    return result


def build_cached_example(
    dataset: str,
    domain: str,
    example_id: str,
    source_rows: Sequence[Mapping[str, Any]],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    if prediction.get("dataset") != dataset or prediction.get("domain") != domain:
        raise ValueError(f"{example_id}: prediction/source dataset or domain mismatch")
    admitted = cap_candidates(source_rows)
    by_identity = {candidate_identity(row): row for row in admitted}
    if len(by_identity) != len(admitted):
        raise ValueError(f"{example_id}: duplicate candidate identities")
    order = list(prediction.get("cross_encoder_order") or [])
    logits = list(prediction.get("cross_encoder_scores") or [])
    if len(order) != len(logits) or len(order) != len(admitted) or set(order) != set(by_identity):
        raise ValueError(f"{example_id}: frozen prediction/source candidate identity mismatch")
    candidates = []
    for source_rank, (identity, logit) in enumerate(zip(order, logits), 1):
        row = by_identity[identity]
        features = text_features(row) if domain == "text" else proof_features(row)
        candidates.append({
            "candidate_identity": identity,
            "cross_encoder_logit": float(logit),
            "cross_encoder_rank": source_rank,
            "subgraph_units": [str(value) for value in row.get("subgraph_units", []) or []],
            "features": features,
        })
    gold = gold_explanations(admitted[0], domain)
    if not gold:
        raise ValueError(f"{example_id}: required DEV gold explanations are empty")
    return {
        "schema_version": SCHEMA_VERSION,
        "split": "dev",
        "dataset": dataset,
        "domain": domain,
        "example_id": example_id,
        "gold_explanations": gold,
        "candidates": candidates,
    }


def prepare_cache(predictions: Path, freeze_path: Path, data_root: Path, source_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    hashes = validate_inputs(predictions, freeze_path, data_root, source_manifest_path)
    frozen = load_predictions(predictions)
    counts: dict[str, int] = {}

    def records() -> Iterable[dict[str, Any]]:
        for dataset, path in expected_inputs(data_root).items():
            domain = "text" if dataset in TEXT_DATASETS else "ontology"
            counts[dataset] = 0
            for example_id, source_rows in grouped_rows(path):
                prediction = frozen.pop(example_id, None)
                if prediction is None:
                    raise ValueError(f"{example_id}: missing frozen Cross-Encoder prediction")
                yield build_cached_example(dataset, domain, example_id, source_rows, prediction)
                counts[dataset] += 1

    cache_path = output_dir / "dev_feature_cache.jsonl"
    write_jsonl(cache_path, records())
    if frozen:
        raise ValueError(f"{len(frozen)} frozen prediction examples were not joined")
    manifest = {
        "schema_version": "sageqa_symbolic_coefficients_original_v1_cache_manifest",
        "status": "complete_dev_feature_cache",
        "split": "dev",
        "test_accessed": False,
        "candidate_limit": MAX_CANDIDATES,
        "datasets": counts,
        "examples": sum(counts.values()),
        "inputs": hashes,
        "cache_sha256": sha256(cache_path),
        "code_sha256": {
            "prepare_dev_cache.py": sha256(Path(__file__)),
            "original_symbolic_features.py": sha256(Path(__file__).with_name("original_symbolic_features.py")),
            "PROTOCOL.md": sha256(Path(__file__).with_name("PROTOCOL.md")),
        },
    }
    write_json(output_dir / "cache_manifest.json", manifest)
    return manifest


def validate_cache(cache_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("split") != "dev" or manifest.get("test_accessed") is not False:
        raise ValueError("Cache manifest is not DEV-only")
    if sha256(cache_path) != manifest.get("cache_sha256"):
        raise ValueError("DEV cache hash mismatch")
    seen: set[str] = set()
    counts: dict[str, int] = defaultdict(int)
    for row in read_jsonl(cache_path):
        if row.get("schema_version") != SCHEMA_VERSION or row.get("split") != "dev":
            raise ValueError("Unexpected cache schema or split")
        example_id = str(row.get("example_id") or "")
        if not example_id or example_id in seen:
            raise ValueError("Missing or duplicate cached example identity")
        seen.add(example_id)
        identities = [candidate["candidate_identity"] for candidate in row.get("candidates", [])]
        if not identities or len(identities) != len(set(identities)):
            raise ValueError(f"{example_id}: empty or duplicate cached candidates")
        counts[str(row["dataset"])] += 1
    if dict(counts) != manifest.get("datasets"):
        raise ValueError("Cache dataset counts do not match manifest")
    return {"valid": True, "examples": len(seen), "cache_sha256": manifest["cache_sha256"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-inputs", "prepare"):
        command = subparsers.add_parser(name)
        command.add_argument("--predictions", type=Path, required=True)
        command.add_argument("--prediction-freeze", type=Path, required=True)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--source-manifest", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--output-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate-cache")
    validate.add_argument("--cache", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate-cache":
        result = validate_cache(args.cache, args.manifest)
    elif args.command == "validate-inputs":
        result = validate_inputs(args.predictions, args.prediction_freeze, args.data_root, args.source_manifest)
    else:
        result = prepare_cache(args.predictions, args.prediction_freeze, args.data_root, args.source_manifest, args.output_dir)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
