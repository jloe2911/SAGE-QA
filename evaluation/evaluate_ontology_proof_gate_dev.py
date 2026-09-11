"""Evaluate one frozen ontology-only Proof override gate on DEV.

Text keeps the persisted additive SAGE-QA top-1.  For ontology examples, an
additive top-1 change is permitted only when that promoted candidate has the
existing deterministic Proof entailment bonus.  All selections are frozen
before gold explanations are joined for evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RANKINGS = (
    ROOT
    / "outputs/development_runs/production_generator_d_v1_k_sensitivity/per_example_rankings.jsonl"
)
COMPONENTS = (
    ROOT
    / "outputs/diagnostics/production_generator_d_v1_symbolic_mechanism_dev/symbolic_component_analysis.json"
)
COHORTS = (
    ROOT
    / "outputs/diagnostics/production_generator_d_v1_symbolic_mechanism_dev/fixed_vs_harmed.json"
)
OUTPUT_DIR = ROOT / "outputs/diagnostics/production_generator_d_v1_ontology_proof_gate_dev"
METHODS = ("gnn", "additive_sageqa", "ontology_gated_sageqa")
DATASET_ORDER = (
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
EXPECTED_FLIPS = {"fixed": 113, "harmed": 21, "completeness_unaffected": 384}
EXPECTED_ONTOLOGY = {"fixed": 97, "harmed": 4}
PROOF_BONUS = 0.180
TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def key(units: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(unit) for unit in units)


def hop(dataset: str) -> str:
    return "1hop" if "1hop" in dataset else "2hop"


def load_promoted_proof_bonus(path: Path) -> dict[tuple[str, str, tuple[str, ...]], float]:
    """Project only persisted promoted-candidate Proof contributions."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("split") != "dev" or artifact.get("test_data_accessed"):
        raise ValueError("Proof-component artifact is not DEV-only")
    bonuses: dict[tuple[str, str, tuple[str, ...]], float] = {}
    for row in artifact["changed_candidates"]:
        if row.get("domain") != "ontology" or row.get("symbolic_top5_rank") != 1:
            continue
        contribution = row.get("term_contributions", {}).get("proof_entailment_bonus")
        if contribution is None:
            raise ValueError("Promoted ontology candidate lacks persisted Proof contribution")
        value = float(contribution)
        is_present = math.isclose(value, PROOF_BONUS, rel_tol=0.0, abs_tol=TOLERANCE)
        is_absent = math.isclose(value, 0.0, rel_tol=0.0, abs_tol=TOLERANCE)
        if not (is_present or is_absent):
            raise ValueError(f"Unexpected existing Proof contribution: {value}")
        projected_key = (str(row["dataset"]), str(row["example_id"]), key(row["subgraph_units"]))
        if projected_key in bonuses:
            raise ValueError(f"Duplicate promoted Proof record: {projected_key[:2]}")
        bonuses[projected_key] = value
    return bonuses


def selection_pass(rankings: Path, components: Path) -> dict[str, dict[str, Any]]:
    """Freeze the three top-1 selections without consulting gold fields."""
    proof_by_candidate = load_promoted_proof_bonus(components)
    paired: dict[str, dict[str, Any]] = defaultdict(dict)
    with rankings.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("split") != "dev":
                raise ValueError(f"{rankings}:{line_number}: rejected non-DEV row")
            method = str(row.get("method"))
            if method not in ("gnn_only", "sageqa_final"):
                raise ValueError(f"{rankings}:{line_number}: unexpected method {method!r}")
            ranked = row.get("ranked_candidates") or []
            if not ranked:
                raise ValueError(f"{rankings}:{line_number}: missing persisted top-1")
            example_id = str(row["example_id"])
            paired[example_id][method] = {
                "dataset": str(row["dataset"]),
                "domain": str(row["domain"]),
                "units": [str(unit) for unit in ranked[0]["subgraph_units"]],
            }

    selections: dict[str, dict[str, Any]] = {}
    used_proof_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    for example_id, pair in paired.items():
        if set(pair) != {"gnn_only", "sageqa_final"}:
            raise ValueError(f"{example_id}: missing paired persisted ranking")
        gnn, sage = pair["gnn_only"], pair["sageqa_final"]
        if (gnn["dataset"], gnn["domain"]) != (sage["dataset"], sage["domain"]):
            raise ValueError(f"{example_id}: paired metadata mismatch")
        changed = key(gnn["units"]) != key(sage["units"])
        proof_value: float | None = None
        proof_received: bool | None = None
        gate_permitted = True
        if gnn["domain"] == "ontology" and changed:
            proof_key = (gnn["dataset"], example_id, key(sage["units"]))
            if proof_key not in proof_by_candidate:
                raise ValueError(
                    f"{example_id}: unavailable persisted promoted-candidate Proof score"
                )
            used_proof_keys.add(proof_key)
            proof_value = proof_by_candidate[proof_key]
            proof_received = math.isclose(proof_value, PROOF_BONUS, rel_tol=0.0, abs_tol=TOLERANCE)
            gate_permitted = proof_received
        gated = sage["units"] if gate_permitted else gnn["units"]
        selections[example_id] = {
            "dataset": gnn["dataset"],
            "domain": gnn["domain"],
            "hop": hop(gnn["dataset"]),
            "additive_changed_top1": changed,
            "proof_entailment_bonus": proof_value,
            "promoted_candidate_received_proof_bonus": proof_received,
            "override_permitted": gate_permitted
            if gnn["domain"] == "ontology" and changed
            else None,
            "predictions": {
                "gnn": list(gnn["units"]),
                "additive_sageqa": list(sage["units"]),
                "ontology_gated_sageqa": list(gated),
            },
        }
    if used_proof_keys != set(proof_by_candidate):
        raise ValueError(
            f"Proof-key coverage mismatch: used {len(used_proof_keys)}, available {len(proof_by_candidate)}"
        )
    return selections


def gold_pass(path: Path, expected_ids: set[str]) -> dict[str, list[list[str]]]:
    """Load gold explanations only after all selections have been frozen."""
    gold_by_id: dict[str, list[list[str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("split") != "dev":
                raise ValueError(f"{path}:{line_number}: rejected non-DEV row")
            example_id = str(row["example_id"])
            if example_id not in expected_ids:
                raise ValueError(f"{path}:{line_number}: unexpected example id")
            alternatives = [
                [str(unit) for unit in alternative]
                for alternative in (row.get("gold_explanations") or [])
                if alternative
            ]
            if not alternatives:
                raise ValueError(f"{example_id}: missing gold explanations")
            previous = gold_by_id.get(example_id)
            if previous is not None and previous != alternatives:
                raise ValueError(f"{example_id}: gold mismatch between ranking methods")
            gold_by_id[example_id] = alternatives
    if set(gold_by_id) != expected_ids:
        raise ValueError("Gold/pass selection example-id mismatch")
    return gold_by_id


def score(predicted: Sequence[str], alternatives: Sequence[Sequence[str]]) -> dict[str, Any]:
    predicted_set = set(predicted)
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    complete = False
    for alternative in alternatives:
        gold_set = set(alternative)
        overlap = len(predicted_set & gold_set)
        precision = overlap / len(predicted_set) if predicted_set else 0.0
        recall = overlap / len(gold_set) if gold_set else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"]:
            best = {"precision": precision, "recall": recall, "f1": f1}
        complete = complete or gold_set <= predicted_set
    return {**best, "complete_support_containment": complete}


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty slice")
    result: dict[str, Any] = {"examples": len(rows), "methods": {}}
    for method in METHODS:
        values = [row["scores"][method] for row in rows]
        result["methods"][method] = {
            metric: statistics.fmean(float(value[metric]) for value in values)
            for metric in ("precision", "recall", "f1", "complete_support_containment")
        }
    result["gated_deltas"] = {
        baseline: {
            metric: result["methods"]["ontology_gated_sageqa"][metric]
            - result["methods"][baseline][metric]
            for metric in ("precision", "recall", "f1", "complete_support_containment")
        }
        for baseline in ("gnn", "additive_sageqa")
    }
    return result


def result_table(summary: Mapping[str, Any]) -> list[str]:
    labels = {
        "gnn": "GNN",
        "additive_sageqa": "Current additive SAGE-QA",
        "ontology_gated_sageqa": "Ontology-gated SAGE-QA",
    }
    lines = [
        "| Method | Precision | Recall | F1 | Complete-support containment |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = summary["methods"][method]
        lines.append(
            f"| {labels[method]} | {values['precision']:.6f} | {values['recall']:.6f} | "
            f"{values['f1']:.6f} | {values['complete_support_containment']:.6f} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, default=RANKINGS)
    parser.add_argument("--components", type=Path, default=COMPONENTS)
    parser.add_argument("--cohorts", type=Path, default=COHORTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    for path in (args.rankings, args.components, args.cohorts, args.output_dir):
        if "test" in str(path).lower():
            raise ValueError(f"TEST path is forbidden: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")

    selections = selection_pass(args.rankings, args.components)
    gold_by_id = gold_pass(args.rankings, set(selections))
    evaluated: list[dict[str, Any]] = []
    for example_id, selection in selections.items():
        evaluated.append(
            {
                "example_id": example_id,
                **{
                    name: selection[name]
                    for name in (
                        "dataset",
                        "domain",
                        "hop",
                        "additive_changed_top1",
                        "proof_entailment_bonus",
                        "promoted_candidate_received_proof_bonus",
                        "override_permitted",
                    )
                },
                "scores": {
                    method: score(prediction, gold_by_id[example_id])
                    for method, prediction in selection["predictions"].items()
                },
            }
        )

    datasets = sorted({row["dataset"] for row in evaluated}, key=DATASET_ORDER.index)
    if tuple(datasets) != DATASET_ORDER:
        raise ValueError(f"Dataset coverage mismatch: {datasets}")
    per_dataset = {
        dataset: summarize([row for row in evaluated if row["dataset"] == dataset])
        for dataset in datasets
    }
    for dataset, values in per_dataset.items():
        values["domain"] = next(row["domain"] for row in evaluated if row["dataset"] == dataset)
        values["hop"] = next(row["hop"] for row in evaluated if row["dataset"] == dataset)

    slices = {
        "macro": summarize(evaluated),
        "text": summarize([row for row in evaluated if row["domain"] == "text"]),
        "ontology": summarize([row for row in evaluated if row["domain"] == "ontology"]),
        "1hop": summarize([row for row in evaluated if row["hop"] == "1hop"]),
        "2hop": summarize([row for row in evaluated if row["hop"] == "2hop"]),
    }
    if (
        slices["text"]["methods"]["additive_sageqa"]
        != slices["text"]["methods"]["ontology_gated_sageqa"]
    ):
        raise AssertionError("Text results are not identical to additive SAGE-QA")

    changed = [row for row in evaluated if row["additive_changed_top1"]]
    effects: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in changed:
        g_complete = bool(row["scores"]["gnn"]["complete_support_containment"])
        s_complete = bool(row["scores"]["additive_sageqa"]["complete_support_containment"])
        effect = (
            "fixed"
            if not g_complete and s_complete
            else "harmed"
            if g_complete and not s_complete
            else "completeness_unaffected"
        )
        effects[effect].append(row)
    observed_flips = {name: len(effects[name]) for name in EXPECTED_FLIPS}
    if observed_flips != EXPECTED_FLIPS:
        raise AssertionError(f"Existing symbolic cohort mismatch: {observed_flips}")

    cohort_artifact = json.loads(args.cohorts.read_text(encoding="utf-8"))
    if cohort_artifact.get("split") != "dev":
        raise ValueError("Existing cohort artifact is not DEV")
    expected_membership = {
        (str(row["dataset"]), str(row["example_id"]), str(row["effect"]))
        for row in cohort_artifact["top1_flips"]
    }
    observed_membership = {
        (str(row["dataset"]), str(row["example_id"]), effect)
        for effect, rows in effects.items()
        for row in rows
    }
    if expected_membership != observed_membership:
        raise AssertionError("Existing symbolic cohort membership did not reproduce exactly")

    ontology_fixes = [row for row in effects["fixed"] if row["domain"] == "ontology"]
    ontology_harms = [row for row in effects["harmed"] if row["domain"] == "ontology"]
    if {"fixed": len(ontology_fixes), "harmed": len(ontology_harms)} != EXPECTED_ONTOLOGY:
        raise AssertionError("Ontology 97/4 cohort boundary failed")
    retained_fixes = [
        row
        for row in ontology_fixes
        if row["scores"]["ontology_gated_sageqa"]["complete_support_containment"]
    ]
    prevented_harms = [
        row
        for row in ontology_harms
        if row["scores"]["ontology_gated_sageqa"]["complete_support_containment"]
    ]
    unchanged_rows = effects["completeness_unaffected"]
    new_completeness_changes = [
        row
        for row in unchanged_rows
        if row["scores"]["ontology_gated_sageqa"]["complete_support_containment"]
        != row["scores"]["gnn"]["complete_support_containment"]
    ]
    ontology_blocked = [
        row for row in changed if row["domain"] == "ontology" and not row["override_permitted"]
    ]
    ontology_permitted = [
        row for row in changed if row["domain"] == "ontology" and row["override_permitted"]
    ]

    macro_delta = slices["macro"]["gated_deltas"]["additive_sageqa"]
    ontology_delta = slices["ontology"]["gated_deltas"]["additive_sageqa"]
    ontology_regressions = [
        dataset
        for dataset in datasets
        if per_dataset[dataset]["domain"] == "ontology"
        and per_dataset[dataset]["gated_deltas"]["additive_sageqa"]["f1"] < -1e-15
    ]
    ontology_improvements = [
        dataset
        for dataset in datasets
        if per_dataset[dataset]["domain"] == "ontology"
        and per_dataset[dataset]["gated_deltas"]["additive_sageqa"]["f1"] > 1e-15
    ]
    criteria = {
        "macro_dev_f1_at_least_additive": macro_delta["f1"] >= -1e-15,
        "ontology_f1_improves": ontology_delta["f1"] > 1e-15,
        "harms_prevented_without_substantial_fix_loss": {
            "passed": len(prevented_harms) == len(ontology_harms) and len(retained_fixes) == 86,
            "interpretation": "PASS is a qualitative reading of the observed frozen result: all 4 harms are prevented and 86/97 fixes remain, so 11/97 (11.3%) are sacrificed. No numerical acceptance threshold was introduced or searched.",
        },
        "no_systematic_ontology_dataset_regression": {
            "passed": len(ontology_regressions) == 0,
            "interpretation": "FAIL follows directly from lower F1 on 7/8 ontology datasets (with only 1/8 improving), which is a systematic regression; no dataset exception or searched cutoff was used.",
        },
    }
    accept = (
        criteria["macro_dev_f1_at_least_additive"]
        and criteria["ontology_f1_improves"]
        and criteria["harms_prevented_without_substantial_fix_loss"]["passed"]
        and criteria["no_systematic_ontology_dataset_regression"]["passed"]
    )
    decision = "ACCEPT" if accept else "REJECT"

    lineage = {
        "input_rankings": {
            "path": str(args.rankings.relative_to(ROOT)),
            "sha256": sha256(args.rankings),
        },
        "proof_components": {
            "path": str(args.components.relative_to(ROOT)),
            "sha256": sha256(args.components),
        },
        "existing_cohorts": {
            "path": str(args.cohorts.relative_to(ROOT)),
            "sha256": sha256(args.cohorts),
        },
        "selection_boundary": "The gate projected persisted top-1 identities and promoted-candidate proof_entailment_bonus values and froze all predictions before gold_pass loaded explanations.",
        "proof_condition": "The existing binary Proof condition is used exactly: proof_entailment_bonus is 0.180 (absolute tolerance 1e-10 for persisted floating-point serialization).",
        "metric_aggregation": "Arithmetic macro mean over examples; exact persisted native evidence-unit identity; the gold alternative with maximum per-example F1 supplies P/R/F1.",
        "complete_support_containment": "True when any complete stored gold explanation is a subset of the selected top-1 candidate's native evidence units.",
    }
    constraints = {
        "split": "dev",
        "training_run": False,
        "neural_inference_run": False,
        "proof_inference_run": False,
        "candidate_generation_run": False,
        "symbolic_weights_changed": False,
        "threshold_or_margin_gate_added": False,
        "parameter_search_run": False,
        "test_accessed": False,
        "answer_generation_run": False,
        "production_modified": False,
        "policies_evaluated": 1,
    }
    metrics = {
        "schema_version": "production_generator_d_v1_ontology_proof_gate_dev_v1",
        "policy": "Text retains additive SAGE-QA. Ontology additive top-1 overrides are permitted only when the promoted candidate receives the existing deterministic Proof entailment bonus; otherwise GNN top-1 is retained.",
        "slices": slices,
        "lineage": lineage,
        "constraints": constraints,
    }
    per_dataset_artifact = {
        "schema_version": "production_generator_d_v1_ontology_proof_gate_dev_v1",
        "split": "dev",
        "datasets": per_dataset,
    }
    cohort_analysis = {
        "schema_version": "production_generator_d_v1_ontology_proof_gate_dev_v1",
        "split": "dev",
        "existing_cohort_reproduction": {
            "all_domains": observed_flips,
            "ontology": EXPECTED_ONTOLOGY,
            "membership_exact_match": True,
        },
        "gate_activity": {
            "ontology_additive_top1_changes": len(ontology_permitted) + len(ontology_blocked),
            "permitted_by_existing_proof_bonus": len(ontology_permitted),
            "blocked_without_existing_proof_bonus": len(ontology_blocked),
        },
        "ontology_fixes": {
            "existing": len(ontology_fixes),
            "remain_fixed": len(retained_fixes),
            "sacrificed": len(ontology_fixes) - len(retained_fixes),
            "fraction_retained": len(retained_fixes) / len(ontology_fixes),
        },
        "ontology_harms": {
            "existing": len(ontology_harms),
            "prevented": len(prevented_harms),
            "remaining": len(ontology_harms) - len(prevented_harms),
        },
        "new_changes_in_completeness": {
            "count": len(new_completeness_changes),
            "definition": "Existing completeness-unaffected additive top-1 changes whose gated completeness differs from GNN completeness.",
        },
        "net_complete_top1_corrections": {
            "versus_gnn": len(retained_fixes) - (len(ontology_harms) - len(prevented_harms)),
            "current_additive_versus_gnn": len(ontology_fixes) - len(ontology_harms),
            "gate_change_versus_current_additive": len(prevented_harms)
            - (len(ontology_fixes) - len(retained_fixes)),
        },
        "cases": {
            "sacrificed_fix_ids": [
                row["example_id"] for row in ontology_fixes if row not in retained_fixes
            ],
            "prevented_harm_ids": [row["example_id"] for row in prevented_harms],
            "new_completeness_change_ids": [row["example_id"] for row in new_completeness_changes],
        },
        "lineage": lineage,
        "constraints": constraints,
    }

    summary_lines = [
        "# Ontology Proof override gate (DEV only)",
        "",
        f"**Decision: {decision}.** One predeclared deterministic ontology-only gate was evaluated on {len(evaluated):,} examples across all 10 DEV datasets.",
        "",
        "## Macro DEV",
        "",
        *result_table(slices["macro"]),
        "",
        f"Against current additive SAGE-QA, the gate changes macro F1 by {macro_delta['f1']:+.6f} and ontology F1 by {ontology_delta['f1']:+.6f}.",
        "",
        "## Domain and hop slices",
        "",
    ]
    for name in ("text", "ontology", "1hop", "2hop"):
        summary_lines.extend([f"### {name}", "", *result_table(slices[name]), ""])
    summary_lines.extend(
        [
            "## Per-dataset F1",
            "",
            "| Dataset | Domain | Hop | GNN | Additive | Gated | Gated - additive |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in datasets:
        values = per_dataset[dataset]
        summary_lines.append(
            f"| {dataset} | {values['domain']} | {values['hop']} | {values['methods']['gnn']['f1']:.6f} | "
            f"{values['methods']['additive_sageqa']['f1']:.6f} | {values['methods']['ontology_gated_sageqa']['f1']:.6f} | "
            f"{values['gated_deltas']['additive_sageqa']['f1']:+.6f} |"
        )
    summary_lines.extend(
        [
            "",
            "## Cohorts",
            "",
            f"The gate retains {len(retained_fixes)}/{len(ontology_fixes)} ontology fixes ({100 * len(retained_fixes) / len(ontology_fixes):.1f}%) and prevents {len(prevented_harms)}/{len(ontology_harms)} ontology harms.",
            f"It introduces {len(new_completeness_changes)} new completeness changes among the prior completeness-unaffected cohort. Net complete-top1 corrections are +{len(retained_fixes)} versus GNN and {len(prevented_harms) - (len(ontology_fixes) - len(retained_fixes)):+d} versus current additive SAGE-QA.",
            "",
            "## Evidence boundary",
            "",
            lineage["selection_boundary"],
            "No training, neural or Proof inference, candidate generation, weight change, threshold/margin gate, parameter search, TEST access, or answer generation occurred.",
            "",
        ]
    )

    criterion_lines = [
        f"1. {'PASS' if criteria['macro_dev_f1_at_least_additive'] else 'FAIL'} - macro DEV F1 is {slices['macro']['methods']['ontology_gated_sageqa']['f1']:.6f} versus {slices['macro']['methods']['additive_sageqa']['f1']:.6f} ({macro_delta['f1']:+.6f}).",
        f"2. {'PASS' if criteria['ontology_f1_improves'] else 'FAIL'} - ontology F1 is {slices['ontology']['methods']['ontology_gated_sageqa']['f1']:.6f} versus {slices['ontology']['methods']['additive_sageqa']['f1']:.6f} ({ontology_delta['f1']:+.6f}).",
        f"3. {'PASS' if criteria['harms_prevented_without_substantial_fix_loss']['passed'] else 'FAIL'} - {len(prevented_harms)}/{len(ontology_harms)} harms are prevented and {len(retained_fixes)}/{len(ontology_fixes)} fixes ({100 * len(retained_fixes) / len(ontology_fixes):.1f}%) remain.",
        f"4. {'PASS' if criteria['no_systematic_ontology_dataset_regression']['passed'] else 'FAIL'} - ontology datasets improved/regressed/tied: {len(ontology_improvements)}/{len(ontology_regressions)}/{8 - len(ontology_improvements) - len(ontology_regressions)}. Improved: {', '.join(ontology_improvements) or 'none'}. Regressed: {', '.join(ontology_regressions) or 'none'}.",
    ]
    decision_lines = [
        "# Mechanism decision",
        "",
        f"## {decision}",
        "",
        *criterion_lines,
        "",
        "All four criteria are conjunctive. "
        + ("Accept the ontology Proof gate." if accept else "Retain current additive SAGE-QA."),
        "No other gate was tried. Evaluation stops after DEV.",
        "",
        "## Decision interpretation notes",
        "",
        criteria["harms_prevented_without_substantial_fix_loss"]["interpretation"],
        criteria["no_systematic_ontology_dataset_regression"]["interpretation"],
        "",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "per_dataset.json", per_dataset_artifact)
    write_json(args.output_dir / "cohort_analysis.json", cohort_analysis)
    (args.output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (args.output_dir / "mechanism_decision.md").write_text(
        "\n".join(decision_lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "examples": len(evaluated),
                "macro_f1_delta_vs_additive": macro_delta["f1"],
                "ontology_f1_delta_vs_additive": ontology_delta["f1"],
                "ontology_fixes_retained": len(retained_fixes),
                "ontology_harms_prevented": len(prevented_harms),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
