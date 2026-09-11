"""Evaluate one fixed, training-free RRF replacement on production-v1 DEV.

The first pass reconstructs the frozen admitted candidate pools, scores them with
the frozen production GNNs, and freezes GNN/additive/RRF top-1 predictions.  It
projects every source row onto an explicit gold-free allowlist before candidate
preparation.  A separate second pass then joins DEV gold for evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.run_production_dev_k_sensitivity import (
    DATASETS,
    grouped_jsonl,
    load_model,
)
from training.train_gnn_subgraph_retriever import (
    compute_adjusted_score,
    encode_example_graph,
    prepare_examples,
    score_candidate_rows,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/production_generator_d_v1"
CHECKPOINT_ROOT = ROOT / "checkpoints/production_generator_d_v1"
BASELINE_RANKINGS = (
    ROOT
    / "outputs/development_runs/production_generator_d_v1_k_sensitivity/per_example_rankings.jsonl"
)
BASELINE_METADATA = (
    ROOT
    / "outputs/development_runs/production_generator_d_v1_k_sensitivity/checkpoint_metadata.json"
)
OUTPUT_DIR = ROOT / "outputs/diagnostics/production_generator_d_v1_rrf_dev"
RRF_K = 60
METHODS = ("gnn_k1", "sageqa_additive_k1", "sageqa_rrf_k1")
EXPECTED_EXAMPLES = 1624
EXPECTED_EFFECT_COUNTS = {"fixed": 113, "harmed": 21}

# Only these fields may cross into candidate admission, neural scoring, or
# symbolic scoring.  In particular, no answer, evidence/support, label, target,
# or gold-derived diagnostic is retained.
SELECTION_FIELDS = {
    "example_id",
    "dataset",
    "hop",
    "answer_type",
    "question",
    "abs_question",
    "task_id",
    "sparql_query",
    "subgraph_units",
    "subgraph_size",
    "graph_context_units",
    "kg_evidence_units",
    "symbolic_features",
    "candidate_pre_rank_score",
    "generation_rank",
    "kg_construction_method",
    "task_type",
    "Task Type",
    "gold_available_during_candidate_generation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def candidate_key(units: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(unit) for unit in units)


def safe_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projected = [{key: row[key] for key in SELECTION_FIELDS if key in row} for row in rows]
    for row in projected:
        if row.get("gold_available_during_candidate_generation") is True:
            raise ValueError("Candidate generation reports gold access")
    return projected


def read_baseline_top1(path: Path) -> dict[str, dict[str, Any]]:
    """Read persisted identities/scores while deliberately ignoring gold fields."""
    paired: dict[str, dict[str, Any]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("split") != "dev":
                raise ValueError(f"{path}:{line_number}: non-DEV row rejected")
            method = str(row.get("method"))
            if method not in {"gnn_only", "sageqa_final"}:
                raise ValueError(f"{path}:{line_number}: unexpected method {method!r}")
            top = (row.get("ranked_candidates") or [None])[0]
            if not top:
                raise ValueError(f"{path}:{line_number}: missing persisted top-1")
            paired[str(row["example_id"])][method] = {
                "dataset": str(row["dataset"]),
                "units": candidate_key(top["subgraph_units"]),
                "score": float(top["score"]),
                "adjusted_score": float(top["adjusted_score"]),
            }
    if any(set(pair) != {"gnn_only", "sageqa_final"} for pair in paired.values()):
        raise ValueError("Persisted baseline rankings are not paired")
    return dict(paired)


def freeze_predictions(
    *,
    data_root: Path,
    checkpoint_root: Path,
    baseline: Mapping[str, Mapping[str, Any]],
    frozen_path: Path,
    batch_size: int,
    max_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selections: list[dict[str, Any]] = []
    accessed: list[str] = []
    dataset_meta: dict[str, Any] = {}
    parity_errors: list[str] = []

    with frozen_path.open("w", encoding="utf-8") as frozen_handle:
        for dataset, data_dir, checkpoint_dir, domain, symbolic_mode in DATASETS:
            dev_path = data_root / data_dir / "dev_subgraph_retrieval.jsonl"
            checkpoint_path = checkpoint_root / checkpoint_dir / "best_model.pt"
            if "test" in str(dev_path).lower() or "test" in str(checkpoint_path).lower():
                raise AssertionError("TEST path rejected")
            if not dev_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(f"Missing frozen input: {dev_path} or {checkpoint_path}")
            accessed.append(str(dev_path.relative_to(ROOT)))
            checkpoint, tokenizer, model = load_model(checkpoint_path, device)
            examples = 0
            admitted_total = 0

            for example_id, raw_rows in grouped_jsonl(dev_path):
                # This projection is the hard boundary: supervision fields are
                # absent from every object passed to preparation and scoring.
                prepared = prepare_examples(safe_rows(raw_rows), candidate_selection="inference")
                if len(prepared) != 1 or prepared[0]["example_id"] != example_id:
                    raise ValueError(f"Unexpected preparation result for {example_id}")
                example = prepared[0]
                encoded = encode_example_graph(
                    model, tokenizer, example, device, max_length=max_length
                )
                scored: list[dict[str, Any]] = []
                with torch.no_grad():
                    for start in range(0, len(example["candidate_rows"]), batch_size):
                        batch = example["candidate_rows"][start : start + batch_size]
                        probabilities = (
                            score_candidate_rows(model, encoded, batch, device)["probs"]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        for row, probability in zip(batch, probabilities):
                            item = dict(row)
                            item["score"] = float(probability)
                            item["candidate_order"] = int(row["materialization_order"])
                            item["symbolic_score"] = float(
                                compute_adjusted_score(
                                    item, 0.0, score_mode=symbolic_mode, size_penalty=0.01
                                )
                            )
                            item["adjusted_score"] = float(item["score"] + item["symbolic_score"])
                            scored.append(item)
                if not scored:
                    raise ValueError(f"No admitted candidates for {example_id}")

                order = lambda row: int(row["candidate_order"])
                neural = sorted(scored, key=lambda row: (-float(row["score"]), order(row)))
                symbolic = sorted(
                    scored, key=lambda row: (-float(row["symbolic_score"]), order(row))
                )
                additive = sorted(
                    scored, key=lambda row: (-float(row["adjusted_score"]), order(row))
                )
                neural_rank = {order(row): rank for rank, row in enumerate(neural, start=1)}
                symbolic_rank = {order(row): rank for rank, row in enumerate(symbolic, start=1)}
                for row in scored:
                    row["rank_gnn"] = neural_rank[order(row)]
                    row["rank_symbolic"] = symbolic_rank[order(row)]
                    row["rrf_score"] = 1.0 / (RRF_K + row["rank_gnn"]) + 1.0 / (
                        RRF_K + row["rank_symbolic"]
                    )
                rrf = sorted(scored, key=lambda row: (-float(row["rrf_score"]), order(row)))

                persisted = baseline.get(example_id)
                if persisted is None or persisted["gnn_only"]["dataset"] != dataset:
                    parity_errors.append(f"{example_id}: missing/mismatched persisted baseline")
                else:
                    if candidate_key(neural[0]["subgraph_units"]) != persisted["gnn_only"]["units"]:
                        parity_errors.append(f"{example_id}: GNN top-1 identity mismatch")
                    if (
                        candidate_key(additive[0]["subgraph_units"])
                        != persisted["sageqa_final"]["units"]
                    ):
                        parity_errors.append(f"{example_id}: additive top-1 identity mismatch")
                    if abs(float(neural[0]["score"]) - persisted["gnn_only"]["score"]) >= 1e-7:
                        parity_errors.append(f"{example_id}: GNN score mismatch")
                    if (
                        abs(
                            float(additive[0]["adjusted_score"])
                            - persisted["sageqa_final"]["adjusted_score"]
                        )
                        >= 1e-7
                    ):
                        parity_errors.append(f"{example_id}: additive score mismatch")

                def top_record(row: Mapping[str, Any]) -> dict[str, Any]:
                    return {
                        "subgraph_units": [str(unit) for unit in row["subgraph_units"]],
                        "candidate_order": order(row),
                        "generation_rank": int(row.get("generation_rank", order(row))),
                        "neural_score": float(row["score"]),
                        "symbolic_score": float(row["symbolic_score"]),
                        "additive_score": float(row["adjusted_score"]),
                        "rank_gnn": int(row["rank_gnn"]),
                        "rank_symbolic": int(row["rank_symbolic"]),
                        "rrf_score": float(row["rrf_score"]),
                    }

                selection = {
                    "example_id": example_id,
                    "dataset": dataset,
                    "domain": domain,
                    "hop": "1hop" if "1hop" in dataset else "2hop",
                    "admitted_candidates": len(scored),
                    "predictions": {
                        "gnn_k1": top_record(neural[0]),
                        "sageqa_additive_k1": top_record(additive[0]),
                        "sageqa_rrf_k1": top_record(rrf[0]),
                    },
                }
                frozen_handle.write(json.dumps(selection, ensure_ascii=False) + "\n")
                selections.append(selection)
                admitted_total += len(scored)
                examples += 1
                if examples % 50 == 0:
                    print(f"{dataset}: froze {examples} DEV predictions", flush=True)

            dataset_meta[dataset] = {
                "examples": examples,
                "admitted_candidates": admitted_total,
                "data_path": str(dev_path.relative_to(ROOT)),
                "data_sha256": sha256(dev_path),
                "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
                "checkpoint_sha256": sha256(checkpoint_path),
                "symbolic_mode": symbolic_mode,
                "inference_candidate_cap": 320,
            }
            del model, tokenizer, checkpoint

    if parity_errors:
        raise AssertionError("Frozen production parity failed:\n" + "\n".join(parity_errors[:20]))
    if len(selections) != EXPECTED_EXAMPLES or set(baseline) != {
        row["example_id"] for row in selections
    }:
        raise AssertionError("DEV example coverage mismatch")
    return selections, {
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "accessed_data_files": accessed,
        "datasets": dataset_meta,
        "baseline_top1_parity": {"passed": True, "identity_mismatches": 0, "score_tolerance": 1e-7},
    }


def gold_pass(path: Path, expected_ids: set[str]) -> dict[str, list[list[str]]]:
    gold: dict[str, list[list[str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("split") != "dev":
                raise ValueError(f"{path}:{line_number}: non-DEV row rejected")
            example_id = str(row["example_id"])
            if example_id not in expected_ids:
                raise ValueError(f"Unexpected example {example_id}")
            alternatives = [
                [str(unit) for unit in alt] for alt in row.get("gold_explanations", []) if alt
            ]
            if not alternatives:
                raise ValueError(f"{example_id}: missing DEV gold")
            if example_id in gold and gold[example_id] != alternatives:
                raise ValueError(f"{example_id}: paired gold mismatch")
            gold[example_id] = alternatives
    if set(gold) != expected_ids:
        raise ValueError("Frozen prediction / gold ID mismatch")
    return gold


def score(units: Sequence[str], alternatives: Sequence[Sequence[str]]) -> dict[str, Any]:
    predicted = set(units)
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    complete = False
    for alternative in alternatives:
        target = set(alternative)
        overlap = len(predicted & target)
        precision = overlap / len(predicted) if predicted else 0.0
        recall = overlap / len(target) if target else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"]:
            best = {"precision": precision, "recall": recall, "f1": f1}
        complete = complete or target <= predicted
    return {**best, "complete_support_containment": complete}


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty slice")
    return {
        "examples": len(rows),
        "methods": {
            method: {
                metric: statistics.fmean(float(row["scores"][method][metric]) for row in rows)
                for metric in ("precision", "recall", "f1", "complete_support_containment")
            }
            for method in METHODS
        },
    }


def deltas(summary: Mapping[str, Any]) -> dict[str, Any]:
    rrf = summary["methods"]["sageqa_rrf_k1"]
    return {
        baseline: {
            metric: float(rrf[metric]) - float(summary["methods"][baseline][metric])
            for metric in rrf
        }
        for baseline in ("gnn_k1", "sageqa_additive_k1")
    }


def table(summary: Mapping[str, Any]) -> list[str]:
    labels = {
        "gnn_k1": "GNN",
        "sageqa_additive_k1": "Additive SAGE-QA",
        "sageqa_rrf_k1": "RRF SAGE-QA",
    }
    lines = [
        "| Method | Precision | Recall | F1 | Complete containment |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary["methods"][method]
        lines.append(
            f"| {labels[method]} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['complete_support_containment']:.6f} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--baseline-rankings", type=Path, default=BASELINE_RANKINGS)
    parser.add_argument("--baseline-metadata", type=Path, default=BASELINE_METADATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    baseline_metadata = json.loads(args.baseline_metadata.read_text(encoding="utf-8"))
    if baseline_metadata.get("split") != "dev" or baseline_metadata.get("test_rows_read") != 0:
        raise ValueError("Baseline lineage is not the frozen DEV-only run")
    baseline = read_baseline_top1(args.baseline_rankings)
    frozen_path = args.output_dir / "frozen_predictions.jsonl"
    selections, run_meta = freeze_predictions(
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        baseline=baseline,
        frozen_path=frozen_path,
        batch_size=args.candidate_batch_size,
        max_length=args.max_length,
    )
    for dataset, observed in run_meta["datasets"].items():
        expected = baseline_metadata["datasets"][dataset]
        if observed["data_sha256"] != expected["dev_sha256"]:
            raise AssertionError(f"{dataset}: DEV candidate artifact hash drift")
        if observed["checkpoint_sha256"] != expected["checkpoint_sha256"]:
            raise AssertionError(f"{dataset}: frozen checkpoint hash drift")

    # Gold is first accessed only after every prediction is persisted/frozen.
    by_id = {row["example_id"]: row for row in selections}
    gold = gold_pass(args.baseline_rankings, set(by_id))
    evaluated: list[dict[str, Any]] = []
    for selection in selections:
        evaluated.append(
            {
                **{
                    key: selection[key]
                    for key in ("example_id", "dataset", "domain", "hop", "admitted_candidates")
                },
                "scores": {
                    method: score(
                        selection["predictions"][method]["subgraph_units"],
                        gold[selection["example_id"]],
                    )
                    for method in METHODS
                },
                "top1_additive_rrf_differ": candidate_key(
                    selection["predictions"]["sageqa_additive_k1"]["subgraph_units"]
                )
                != candidate_key(selection["predictions"]["sageqa_rrf_k1"]["subgraph_units"]),
            }
        )

    datasets = [item[0] for item in DATASETS]
    per_dataset = {
        dataset: summarize([row for row in evaluated if row["dataset"] == dataset])
        for dataset in datasets
    }
    for dataset, summary in per_dataset.items():
        summary["domain"] = next(row["domain"] for row in evaluated if row["dataset"] == dataset)
        summary["hop"] = next(row["hop"] for row in evaluated if row["dataset"] == dataset)
        summary["rrf_deltas"] = deltas(summary)
    slices = {
        "macro_all_examples": summarize(evaluated),
        "text": summarize([row for row in evaluated if row["domain"] == "text"]),
        "ontology": summarize([row for row in evaluated if row["domain"] == "ontology"]),
        "1hop": summarize([row for row in evaluated if row["hop"] == "1hop"]),
        "2hop": summarize([row for row in evaluated if row["hop"] == "2hop"]),
    }
    for summary in slices.values():
        summary["rrf_deltas"] = deltas(summary)
    dataset_macro = {
        "examples": 10,
        "methods": {
            method: {
                metric: statistics.fmean(
                    per_dataset[dataset]["methods"][method][metric] for dataset in datasets
                )
                for metric in ("precision", "recall", "f1", "complete_support_containment")
            }
            for method in METHODS
        },
    }
    dataset_macro["rrf_deltas"] = deltas(dataset_macro)
    slices["macro_across_datasets"] = dataset_macro

    fixed: list[dict[str, Any]] = []
    harmed: list[dict[str, Any]] = []
    for row in evaluated:
        gnn_complete = bool(row["scores"]["gnn_k1"]["complete_support_containment"])
        additive_complete = bool(
            row["scores"]["sageqa_additive_k1"]["complete_support_containment"]
        )
        if not gnn_complete and additive_complete:
            fixed.append(row)
        elif gnn_complete and not additive_complete:
            harmed.append(row)
    if {"fixed": len(fixed), "harmed": len(harmed)} != EXPECTED_EFFECT_COUNTS:
        raise AssertionError(
            f"Known symbolic cohorts failed to reproduce: {len(fixed)=}, {len(harmed)=}"
        )

    rrf_fixes = [
        row
        for row in evaluated
        if not row["scores"]["gnn_k1"]["complete_support_containment"]
        and row["scores"]["sageqa_rrf_k1"]["complete_support_containment"]
    ]
    rrf_harms = [
        row
        for row in evaluated
        if row["scores"]["gnn_k1"]["complete_support_containment"]
        and not row["scores"]["sageqa_rrf_k1"]["complete_support_containment"]
    ]
    cohort = {
        "current_symbolic_fixes": len(fixed),
        "fixes_preserved_by_rrf": sum(
            row["scores"]["sageqa_rrf_k1"]["complete_support_containment"] for row in fixed
        ),
        "current_symbolic_harms": len(harmed),
        "harms_avoided_by_rrf": sum(
            row["scores"]["sageqa_rrf_k1"]["complete_support_containment"] for row in harmed
        ),
        "rrf_complete_top1_fixes_vs_gnn": len(rrf_fixes),
        "rrf_complete_top1_harms_vs_gnn": len(rrf_harms),
        "net_complete_top1_corrections_vs_gnn": len(rrf_fixes) - len(rrf_harms),
        "additive_rrf_top1_differences": sum(row["top1_additive_rrf_differ"] for row in evaluated),
    }

    overall = slices["macro_all_examples"]
    f1_gain = overall["rrf_deltas"]["sageqa_additive_k1"]["f1"]
    symbolic_positive = cohort["net_complete_top1_corrections_vs_gnn"] > 0
    domain_deltas = {
        domain: slices[domain]["rrf_deltas"]["sageqa_additive_k1"]["f1"]
        for domain in ("text", "ontology")
    }
    # Predeclared operational reading of "major systematic": an absolute domain
    # F1 loss greater than 0.01, or every dataset in a domain declining.
    systematic = {}
    for domain in ("text", "ontology"):
        domain_datasets = [
            dataset for dataset in datasets if per_dataset[dataset]["domain"] == domain
        ]
        declining = [
            dataset
            for dataset in domain_datasets
            if per_dataset[dataset]["rrf_deltas"]["sageqa_additive_k1"]["f1"] < 0.0
        ]
        systematic[domain] = {
            "f1_delta": domain_deltas[domain],
            "declining_datasets": declining,
            "dataset_count": len(domain_datasets),
            "major_systematic_regression": domain_deltas[domain] < -0.01
            or len(declining) == len(domain_datasets),
        }
    no_major_domain_regression = not any(
        item["major_systematic_regression"] for item in systematic.values()
    )
    accepted = f1_gain > 0.0 and symbolic_positive and no_major_domain_regression
    decision = "ACCEPT_RRF" if accepted else "RETAIN_ADDITIVE"

    constraints = {
        "split": "dev",
        "rrf_k": RRF_K,
        "fusion_methods_evaluated": 1,
        "training_run": False,
        "gnn_inference_run": True,
        "gnn_inference_necessity": "Required because the frozen production artifact persisted only five of up to 320 admitted candidates.",
        "candidate_generation_run": False,
        "parameter_search_run": False,
        "symbolic_weight_change": False,
        "weighted_rrf_run": False,
        "alternative_fusion_run": False,
        "test_accessed": False,
        "answer_generation_run": False,
    }
    lineage = {
        **run_meta,
        "baseline_rankings": str(args.baseline_rankings.relative_to(ROOT)),
        "baseline_rankings_sha256": sha256(args.baseline_rankings),
        "baseline_metadata": str(args.baseline_metadata.relative_to(ROOT)),
        "baseline_metadata_sha256": sha256(args.baseline_metadata),
        "selection_boundary": "All 1,624 predictions were written to frozen_predictions.jsonl before gold_pass accessed DEV gold.",
        "gold_free_projection_allowlist": sorted(SELECTION_FIELDS),
        "ranking": "1-based ranks; descending neural score and unchanged symbolic-only contribution; stable ties by admitted candidate order; RRF=1/(60+rank_gnn)+1/(60+rank_symbolic).",
    }
    metrics = {
        "schema_version": "production_generator_d_v1_rrf_dev_v1",
        "decision": decision,
        "slices": slices,
        "cohort_analysis": cohort,
        "domain_regression_assessment": systematic,
        "criteria": {
            "macro_dev_f1_exceeds_additive": f1_gain > 0.0,
            "symbolic_benefit_positive": symbolic_positive,
            "no_major_systematic_domain_regression": no_major_domain_regression,
        },
        "constraints": constraints,
        "lineage": lineage,
    }
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(
        args.output_dir / "per_dataset.json",
        {"schema_version": metrics["schema_version"], "split": "dev", "datasets": per_dataset},
    )
    write_json(args.output_dir / "cohort_analysis.json", cohort)

    summary_lines = [
        "# Fixed RRF reranker replacement - DEV",
        "",
        f"**Decision: {decision}.** Standard unweighted RRF with fixed k={RRF_K} was evaluated once on all {len(evaluated):,} examples in all ten DEV datasets.",
        "",
        "## Macro over all DEV examples",
        "",
        *table(overall),
        "",
        "## Macro across the ten dataset metrics",
        "",
        *table(dataset_macro),
        "",
        "## Requested slices",
        "",
    ]
    for name in ("text", "ontology", "1hop", "2hop"):
        summary_lines.extend([f"### {name}", "", *table(slices[name]), ""])
    summary_lines.extend(
        [
            "## Per dataset",
            "",
            "| Dataset | Method | Precision | Recall | F1 | Complete containment |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in datasets:
        item = per_dataset[dataset]
        for method, label in (
            ("gnn_k1", "GNN"),
            ("sageqa_additive_k1", "Additive SAGE-QA"),
            ("sageqa_rrf_k1", "RRF SAGE-QA"),
        ):
            values = item["methods"][method]
            summary_lines.append(
                f"| {dataset} | {label} | {values['precision']:.6f} | "
                f"{values['recall']:.6f} | {values['f1']:.6f} | "
                f"{values['complete_support_containment']:.6f} |"
            )
    summary_lines.extend(
        [
            "",
            "## Known symbolic-change cohorts",
            "",
            f"RRF preserves {cohort['fixes_preserved_by_rrf']}/{cohort['current_symbolic_fixes']} current symbolic fixes and avoids {cohort['harms_avoided_by_rrf']}/{cohort['current_symbolic_harms']} current symbolic harms.",
            f"Relative to GNN, RRF makes {cohort['rrf_complete_top1_fixes_vs_gnn']} complete-top1 fixes and {cohort['rrf_complete_top1_harms_vs_gnn']} harms, for {cohort['net_complete_top1_corrections_vs_gnn']:+d} net corrections.",
            f"Additive SAGE-QA and RRF select different top-1 candidates on {cohort['additive_rrf_top1_differences']}/{len(evaluated)} examples.",
            "",
            "## Protocol",
            "",
            lineage["selection_boundary"],
            "No training, candidate generation, parameter search, weight tuning, TEST access, or answer generation was performed. Frozen GNN inference was necessary because only the top five candidates had been persisted previously.",
            "",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    decision_lines = [
        "# RRF decision",
        "",
        f"## {decision}",
        "",
        f"1. {'PASS' if f1_gain > 0 else 'FAIL'} - macro DEV F1: RRF {overall['methods']['sageqa_rrf_k1']['f1']:.6f}, additive {overall['methods']['sageqa_additive_k1']['f1']:.6f}, delta {f1_gain:+.6f}.",
        f"2. {'PASS' if symbolic_positive else 'FAIL'} - positive symbolic benefit: net complete-top1 corrections relative to GNN are {cohort['net_complete_top1_corrections_vs_gnn']:+d}.",
        f"3. {'PASS' if no_major_domain_regression else 'FAIL'} - no major systematic domain regression: text delta {domain_deltas['text']:+.6f}; ontology delta {domain_deltas['ontology']:+.6f}.",
        "",
        "The three acceptance conditions are conjunctive. No other k, weighting, or fusion method was evaluated.",
        "",
    ]
    (args.output_dir / "mechanism_decision.md").write_text(
        "\n".join(decision_lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "examples": len(evaluated),
                "f1_delta_vs_additive": f1_gain,
                **cohort,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
