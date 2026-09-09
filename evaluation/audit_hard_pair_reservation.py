"""Implementation/invariance gate for the precommitted SAGE-QA v2 sampler."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_gnn_subgraph_retriever import (
    binary_label_from_target,
    ranking_target,
    sample_weighted_margin_pairs,
    select_reserved_hard_pair,
    subsample_candidate_rows,
)


DATASETS = (
    "HotpotQA",
    "2WikiMultiHopQA",
    "FamilyOWL_1hop",
    "FamilyOWL_2hop",
    "pizza_100_1hop",
    "pizza_100_2hop",
    "pizza_250_1hop",
    "pizza_250_2hop",
    "OWL2Bench_1hop",
    "OWL2Bench_2hop",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_row(row: dict, order: int) -> dict:
    target = ranking_target(row)
    units = row.get("subgraph_units", []) or []
    return {
        "row_id": str(row.get("row_id") or f"order:{order}"),
        "label": binary_label_from_target(target),
        "rank_target": target,
        "best_set_f1_to_gold": float(row.get("best_set_f1_to_gold", 0.0)),
        "candidate_pre_rank_score": float(row.get("candidate_pre_rank_score", 0.0)),
        "generation_rank": int(row.get("generation_rank", order)),
        "materialization_order": order,
        "subgraph_units": units,
        "subgraph_size": int(row.get("subgraph_size", len(units))),
    }


def iter_examples(path: Path):
    current_id = None
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            example_id = str(raw["example_id"])
            if current_id is not None and example_id != current_id:
                if example_id in seen:
                    raise AssertionError(f"Non-contiguous example {example_id!r}")
                seen.add(current_id)
                yield current_id, rows
                rows = []
            current_id = example_id
            rows.append(normalized_row(raw, len(rows)))
    if current_id is not None:
        yield current_id, rows


def audit_dataset(path: Path) -> dict:
    counts = defaultdict(int)
    random.seed(42)
    examples = []
    for example_id, full_rows in iter_examples(path):
        counts["examples"] += 1
        chosen = select_reserved_hard_pair(full_rows)

        candidate_rng_before = random.getstate()
        baseline = subsample_candidate_rows(full_rows)
        candidate_rng_after_baseline = random.getstate()
        random.setstate(candidate_rng_before)
        reserved = subsample_candidate_rows(full_rows, reserve_hard_pair=True)
        candidate_rng_after_reserved = random.getstate()
        if candidate_rng_after_baseline != candidate_rng_after_reserved:
            raise AssertionError(f"Candidate RNG drift for {example_id}")
        random.setstate(candidate_rng_after_baseline)

        eligible = chosen is not None
        complete = [i for i, r in enumerate(reserved) if r.get("_hard_pair_role") == "complete"]
        competitor = [i for i, r in enumerate(reserved) if r.get("_hard_pair_role") == "competitor"]
        if eligible:
            counts["eligible_examples"] += 1
            if len(complete) != 1 or len(competitor) != 1:
                raise AssertionError(f"Reserved candidates missing/duplicated for {example_id}")
            if reserved[complete[0]] is not chosen[0] or reserved[competitor[0]] is not chosen[1]:
                raise AssertionError(f"Wrong reserved candidates for {example_id}")
            counts["candidate_reserved_pair_included"] += 1
        elif complete or competitor:
            raise AssertionError(f"Ineligible example was reserved: {example_id}")

        if len(baseline) != len(reserved):
            raise AssertionError(f"Candidate budget changed for {example_id}")
        counts["candidate_budget_unchanged"] += 1

        baseline_ids = [r["row_id"] for r in baseline]
        reserved_ids = [r["row_id"] for r in reserved]
        displaced = set(baseline_ids).symmetric_difference(reserved_ids)
        missing_reservations = sum(
            r["row_id"] not in baseline_ids for r in reserved if r.get("_hard_pair_role")
        )
        if len(displaced) != 2 * missing_reservations:
            raise AssertionError(f"Unexpected candidate displacement for {example_id}")
        counts["candidate_only_required_displacement"] += 1
        counts["candidate_displacements"] += missing_reservations

        full_targets = {r["row_id"]: r["rank_target"] for r in full_rows}
        if any(r["rank_target"] != full_targets[r["row_id"]] for r in reserved):
            raise AssertionError(f"Rank target changed for {example_id}")
        counts["rank_targets_unchanged"] += 1

        targets = [float(r["rank_target"]) for r in reserved]
        pair_rng_before = random.getstate()
        baseline_pairs = sample_weighted_margin_pairs(targets, max_pairs=512)
        pair_rng_after_baseline = random.getstate()
        random.setstate(pair_rng_before)
        reserved_pair = (complete[0], competitor[0]) if eligible else None
        reserved_pairs = sample_weighted_margin_pairs(
            targets, max_pairs=512, reserved_pair=reserved_pair
        )
        pair_rng_after_reserved = random.getstate()
        if pair_rng_after_baseline != pair_rng_after_reserved:
            raise AssertionError(f"Pair RNG drift for {example_id}")
        random.setstate(pair_rng_after_baseline)
        if len(baseline_pairs) != len(reserved_pairs):
            raise AssertionError(f"Pair budget changed for {example_id}")
        counts["pair_budget_unchanged"] += 1
        if eligible:
            hits = sum(pair[:2] == reserved_pair for pair in reserved_pairs)
            if hits != 1:
                raise AssertionError(f"Reserved pair count {hits} for {example_id}")
            counts["margin_reserved_pair_included"] += 1
            if len(set(reserved_pairs)) != len(reserved_pairs):
                raise AssertionError(f"Duplicate margin pair for {example_id}")
            counts["no_duplicate_reserved_pairs"] += 1

        if len(examples) < 5:
            examples.append({
                "example_id": example_id,
                "eligible": eligible,
                "full_candidates": len(full_rows),
                "sampled_candidates": len(reserved),
                "sampled_pairs": len(reserved_pairs),
                "candidate_displacements": missing_reservations,
            })

    result_counts = dict(counts)
    eligible_count = result_counts.get("eligible_examples", 0)
    return {
        "counts": result_counts,
        "reserved_candidate_inclusion_rate": result_counts.get("candidate_reserved_pair_included", 0) / max(eligible_count, 1),
        "reserved_margin_pair_inclusion_rate": result_counts.get("margin_reserved_pair_included", 0) / max(eligible_count, 1),
        "sample_examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/production_generator_d_v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/final_results/production_generator_d_v2_freeze/implementation_invariance_gate.json",
    )
    args = parser.parse_args()
    report = {
        "schema_version": "sageqa_v2_hard_pair_reservation_gate_v1",
        "status": "passed",
        "split": "train",
        "seed": 42,
        "candidate_budget": {"positive": 64, "hard_negative": 128, "easy_negative": 128},
        "pair_budget": 512,
        "test_data_accessed": False,
        "dev_data_accessed": False,
        "rank_target_definition_changed": False,
        "datasets": {},
        "accessed_files": [],
    }
    for dataset in DATASETS:
        path = args.data_root / dataset / "train_subgraph_retrieval.jsonl"
        if not path.is_file() or path.name != "train_subgraph_retrieval.jsonl":
            raise FileNotFoundError(path)
        print(f"audit {dataset}", flush=True)
        result = audit_dataset(path)
        result["train_path"] = str(path.relative_to(ROOT))
        result["train_sha256"] = sha256(path)
        report["datasets"][dataset] = result
        report["accessed_files"].append(str(path.relative_to(ROOT)))

    total_eligible = sum(d["counts"].get("eligible_examples", 0) for d in report["datasets"].values())
    total_candidate = sum(d["counts"].get("candidate_reserved_pair_included", 0) for d in report["datasets"].values())
    total_pair = sum(d["counts"].get("margin_reserved_pair_included", 0) for d in report["datasets"].values())
    report["global"] = {
        "eligible_examples": total_eligible,
        "reserved_candidate_inclusion_rate": total_candidate / max(total_eligible, 1),
        "reserved_margin_pair_inclusion_rate": total_pair / max(total_eligible, 1),
    }
    if total_candidate != total_eligible or total_pair != total_eligible:
        raise AssertionError("Global reservation inclusion is not 100%")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["global"], indent=2))


if __name__ == "__main__":
    main()
