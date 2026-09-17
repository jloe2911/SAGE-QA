"""Stage-wise TEST error analysis over frozen per-question predictions.

This is an evaluation-only program.  It validates the frozen inputs, joins
support alternatives only after loading the already frozen answer predictions,
and writes derived diagnostics without modifying any source artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation import generate_owl_answers_with_llm as owl_reader
from generation import run_production_test_answer_generation as production


FINAL_DIR = ROOT / "outputs/final_results/final_manuscript_test_end_to_end"
SELECTION_DIR = ROOT / "outputs/final_results/question_candidate_cross_encoder_v1_adaptive_test_a40"
GOLD_DIR = ROOT / "outputs/final_results/production_generator_d_v2_hard_pair_test_retrieval"
DEFAULT_OUTPUT = ROOT / "outputs/final_results/final_manuscript_stagewise_error_analysis"

CONDITIONS = (
    ("cross_encoder", "k1", "Cross-Encoder ($k=1$)"),
    ("final_sageqa", "k1", "SAGE-QA ($k=1$)"),
    ("final_sageqa", "adaptive", "SAGE-QA (adaptive)"),
)
OUTCOMES = ("exact", "complete_nonminimal", "incomplete", "disjoint")
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
DISPLAY = {
    "HotpotQA": "HotpotQA",
    "2WikiMultiHopQA": "2WikiMultiHopQA",
    "FamilyOWL_1hop": "Family 1-hop",
    "FamilyOWL_2hop": "Family 2-hop",
    "pizza_100_1hop": "Pizza100 1-hop",
    "pizza_100_2hop": "Pizza100 2-hop",
    "pizza_250_1hop": "Pizza250 1-hop",
    "pizza_250_2hop": "Pizza250 2-hop",
    "OWL2Bench_1hop": "OWL2Bench 1-hop",
    "OWL2Bench_2hop": "OWL2Bench 2-hop",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def manifest_hash(manifest: Mapping[str, Any], name: str) -> str:
    for group in ("frozen_inputs", "derived_files", "files"):
        files = manifest.get(group, {})
        if not isinstance(files, Mapping) or name not in files:
            continue
        value = files[name]
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            return str(value["sha256"])
    raise ValueError(f"Manifest has no hash for {name}")


def validate_frozen_sources() -> dict[str, str]:
    paths = {
        "predictions.jsonl": FINAL_DIR / "predictions.jsonl",
        "frozen_reader_inputs.jsonl": FINAL_DIR / "frozen_reader_inputs.jsonl",
        "per_example_end_to_end.jsonl": FINAL_DIR / "per_example_end_to_end.jsonl",
        "test_predictions_frozen.jsonl": SELECTION_DIR / "test_predictions_frozen.jsonl",
        "gold_per_example_test_retrieval.jsonl": GOLD_DIR / "per_example_test_retrieval.jsonl",
    }
    final_manifest = json.loads((FINAL_DIR / "artifact_manifest.json").read_text(encoding="utf-8"))
    selection_manifest = json.loads((SELECTION_DIR / "artifact_manifest.json").read_text(encoding="utf-8"))
    gold_manifest = json.loads((GOLD_DIR / "artifact_manifest.json").read_text(encoding="utf-8"))
    expected = {
        "predictions.jsonl": manifest_hash(final_manifest, "predictions.jsonl"),
        "frozen_reader_inputs.jsonl": manifest_hash(final_manifest, "frozen_reader_inputs.jsonl"),
        "per_example_end_to_end.jsonl": manifest_hash(final_manifest, "per_example_end_to_end.jsonl"),
        "test_predictions_frozen.jsonl": manifest_hash(selection_manifest, "test_predictions_frozen.jsonl"),
        "gold_per_example_test_retrieval.jsonl": manifest_hash(gold_manifest, "per_example_test_retrieval.jsonl"),
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    if actual != expected:
        mismatch = {name: {"expected": expected[name], "actual": actual[name]} for name in actual if actual[name] != expected[name]}
        raise ValueError(f"Frozen source hash mismatch: {mismatch}")
    freeze = json.loads((FINAL_DIR / "generation_freeze.json").read_text(encoding="utf-8"))
    if freeze.get("status") != "complete_frozen" or freeze.get("predictions_sha256") != actual["predictions.jsonl"]:
        raise ValueError("Answer predictions are not complete and hash-frozen")
    selection_freeze = json.loads((SELECTION_DIR / "generation_freeze.json").read_text(encoding="utf-8"))
    if not str(selection_freeze.get("status", "")).startswith("complete_frozen") or selection_freeze.get("predictions_sha256") != actual["test_predictions_frozen.jsonl"]:
        raise ValueError("Retrieval selections are not complete and hash-frozen")
    return actual


def unique_sets(alternatives: Sequence[Sequence[str]]) -> list[frozenset[str]]:
    result: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    for alternative in alternatives:
        current = frozenset(str(unit) for unit in alternative)
        if current and current not in seen:
            seen.add(current)
            result.append(current)
    return result


def choose_reference(retrieved: frozenset[str], alternatives: Sequence[frozenset[str]]) -> tuple[int, frozenset[str]]:
    return min(
        enumerate(alternatives),
        key=lambda item: (
            len(item[1] - retrieved),
            len(retrieved - item[1]),
            len(item[1]),
            item[0],
        ),
    )


def minimal_entailing_subset(metadata: Mapping[str, Any], support: Sequence[str]) -> list[str] | None:
    """Return a deterministic deletion-minimal proof support when supported."""
    kept = list(dict.fromkeys(str(unit) for unit in support))
    if owl_reader.infer_owl_boolean_answer(dict(metadata), kept) is None:
        return None
    index = 0
    while index < len(kept):
        trial = kept[:index] + kept[index + 1 :]
        if owl_reader.infer_owl_boolean_answer(dict(metadata), trial) is not None:
            kept = trial
        else:
            index += 1
    return kept


def is_true_answer(answer: str) -> bool:
    return str(answer).strip().lower() in {"true", "yes", "1"}


def query_form(metadata: Mapping[str, Any]) -> str:
    query = str(metadata.get("sparql_query") or "").upper()
    if "ASK" in query:
        return "ASK"
    if "SELECT" in query:
        return "SELECT"
    return "UNKNOWN"


def classify_support(
    retrieved_units: Sequence[str],
    alternatives: Sequence[Sequence[str]],
    *,
    domain: str,
    metadata: Mapping[str, Any],
    gold_answer: str,
) -> dict[str, Any]:
    retrieved = frozenset(str(unit) for unit in retrieved_units)
    references = unique_sets(alternatives)
    if not references:
        raise ValueError("Stage-wise analysis requires at least one non-empty reference support")

    exact = [(index, gold) for index, gold in enumerate(references) if retrieved == gold]
    complete = [(index, gold) for index, gold in enumerate(references) if gold < retrieved]
    reasoner_rescue = False
    proof_support: list[str] | None = None
    if exact:
        reference_index, reference = exact[0]
        outcome = "exact"
    elif complete:
        reference_index, reference = min(complete, key=lambda item: (len(retrieved - item[1]), len(item[1]), item[0]))
        outcome = "complete_nonminimal"
    else:
        reference_index, reference = choose_reference(retrieved, references)
        if domain == "ontology" and query_form(metadata) == "ASK" and is_true_answer(gold_answer):
            proof_support = minimal_entailing_subset(metadata, list(retrieved_units))
        if proof_support is not None:
            proof_set = frozenset(proof_support)
            outcome = "exact" if proof_set == retrieved else "complete_nonminimal"
            reasoner_rescue = True
            reference_index = None
            reference = proof_set
        elif any(retrieved & gold for gold in references):
            outcome = "incomplete"
        else:
            outcome = "disjoint"

    return {
        "retrieval_outcome": outcome,
        "reference_index": reference_index,
        "reference_size": len(reference),
        "retrieved_size": len(retrieved),
        "missing_units": len(reference - retrieved),
        "additional_units": len(retrieved - reference),
        "reference_alternative_count": len(references),
        "reasoner_validated_alternative": reasoner_rescue,
        "minimal_reasoner_proof_size": len(proof_support) if proof_support is not None else None,
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate an empty row collection")
    n = len(rows)
    counts = Counter(str(row["retrieval_outcome"]) for row in rows)
    cells = Counter((str(row["retrieval_outcome"]), bool(row["answer_correct"])) for row in rows)
    complete = [row for row in rows if row["retrieval_outcome"] in {"exact", "complete_nonminimal"}]
    exact = [row for row in rows if row["retrieval_outcome"] == "exact"]
    incomplete = [row for row in rows if row["retrieval_outcome"] == "incomplete"]
    disjoint = [row for row in rows if row["retrieval_outcome"] == "disjoint"]

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    def accuracy(items: Sequence[Mapping[str, Any]]) -> float | None:
        return rate(sum(bool(row["answer_correct"]) for row in items), len(items))

    return {
        "n": n,
        "counts": {outcome: counts[outcome] for outcome in OUTCOMES},
        "cells": {
            f"{outcome}_{'correct' if correct else 'wrong'}": cells[(outcome, correct)]
            for outcome in OUTCOMES
            for correct in (True, False)
        },
        "retrieval_failure_rate": rate(counts["incomplete"] + counts["disjoint"], n),
        "exact_retrieval_rate": rate(counts["exact"], n),
        "complete_nonminimal_rate": rate(counts["complete_nonminimal"], n),
        "downstream_failure_given_complete": rate(sum(not row["answer_correct"] for row in complete), len(complete)),
        "downstream_failure_given_exact": rate(sum(not row["answer_correct"] for row in exact), len(exact)),
        "downstream_failure_given_nonminimal": rate(
            cells[("complete_nonminimal", False)], counts["complete_nonminimal"]
        ),
        "answer_accuracy_given_incomplete": accuracy(incomplete),
        "answer_accuracy_given_disjoint": accuracy(disjoint),
        "answer_correct_without_complete_count": sum(bool(row["answer_correct"]) for row in incomplete + disjoint),
        "answer_correct_without_complete_rate": rate(
            sum(bool(row["answer_correct"]) for row in incomplete + disjoint), n
        ),
        "mean_missing_units": fmean(float(row["missing_units"]) for row in rows),
        "mean_additional_units": fmean(float(row["additional_units"]) for row in rows),
        "reasoner_validated_alternatives": sum(bool(row["reasoner_validated_alternative"]) for row in rows),
        "multiple_reference_questions": sum(int(row["reference_alternative_count"]) > 1 for row in rows),
    }


def group_aggregates(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: aggregate(group) for name, group in sorted(groups.items())}


def ontology_routing(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ontology = [row for row in rows if row["domain"] == "ontology"]
    routes: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ontology:
        form = str(row["query_form"])
        source = str(row["answer_source"])
        if form == "ASK" and source == "deterministic_owl_proof":
            route = "ask_deterministic_proof"
        elif form == "ASK":
            route = "ask_llm_no_proof"
        elif form == "SELECT":
            route = "select_llm"
        else:
            route = "unknown"
        routes[route].append(row)

    result: dict[str, Any] = {}
    for route, route_rows in sorted(routes.items()):
        complete_rows = [
            row for row in route_rows
            if row["retrieval_outcome"] in {"exact", "complete_nonminimal"}
        ]
        complete_wrong = [
            row for row in route_rows
            if row["retrieval_outcome"] in {"exact", "complete_nonminimal"} and not row["answer_correct"]
        ]
        result[route] = {
            "n": len(route_rows),
            "answer_errors": sum(not row["answer_correct"] for row in route_rows),
            "complete_support_n": len(complete_rows),
            "complete_support_answer_errors": len(complete_wrong),
            "answer_failure_given_complete_support": (
                len(complete_wrong) / len(complete_rows) if complete_rows else None
            ),
        }

    attribution = Counter()
    for row in ontology:
        if row["retrieval_outcome"] not in {"exact", "complete_nonminimal"} or row["answer_correct"]:
            continue
        form = str(row["query_form"])
        source = str(row["answer_source"])
        if form == "ASK" and source == "deterministic_owl_proof":
            attribution["proof_procedure"] += 1
        elif form == "ASK" and is_true_answer(str(row["gold_answer"])):
            attribution["proof_coverage_then_llm_failure"] += 1
        else:
            attribution["llm_answer_generation"] += 1
    result["complete_support_error_attribution"] = dict(attribution)
    return result


def paired_transitions(
    rows: Sequence[Mapping[str, Any]],
    left: tuple[str, str],
    right: tuple[str, str],
) -> dict[str, Any]:
    by_condition = {
        condition: {str(row["example_id"]): row for row in rows if (row["method"], row["setting"]) == condition}
        for condition in (left, right)
    }
    if set(by_condition[left]) != set(by_condition[right]):
        raise ValueError(f"Paired cohorts differ for {left} versus {right}")
    transitions = Counter(
        (
            str(by_condition[left][example_id]["retrieval_outcome"]),
            str(by_condition[right][example_id]["retrieval_outcome"]),
        )
        for example_id in by_condition[left]
    )
    return {f"{source}_to_{target}": count for (source, target), count in sorted(transitions.items())}


def pct(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}"


def cell(count: int, n: int) -> str:
    return f"{count} ({100 * count / n:.1f}\\%)"


def latex_table(per_dataset: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{landscape}",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}llrrrrrr@{}}",
        r"\caption{Stage-wise support-retrieval and answer outcomes on support-bearing TEST questions. Each cell gives count (percentage within the dataset--method row). The Disjoint column reports correct/wrong counts.}\label{7:tab_stagewise_errors}\\",
        r"\toprule",
        r"Dataset & Method & Exact+corr. & Non-min.+corr. & Complete+wrong & Incomp.+corr. & Incomp.+wrong & Disjoint c/w \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Dataset & Method & Exact+corr. & Non-min.+corr. & Complete+wrong & Incomp.+corr. & Incomp.+wrong & Disjoint c/w \\",
        r"\midrule",
        r"\endhead",
    ]
    for dataset in DATASET_ORDER:
        for method, setting, label in CONDITIONS:
            stats = per_dataset[f"{method}/{setting}"][dataset]
            n, cells = stats["n"], stats["cells"]
            exact_correct = cells["exact_correct"]
            nonminimal_correct = cells["complete_nonminimal_correct"]
            complete_wrong = cells["exact_wrong"] + cells["complete_nonminimal_wrong"]
            incomplete_correct = cells["incomplete_correct"]
            incomplete_wrong = cells["incomplete_wrong"]
            disjoint = f"{cells['disjoint_correct']}/{cells['disjoint_wrong']} ({100 * stats['counts']['disjoint'] / n:.1f}\\%)"
            lines.append(
                f"{DISPLAY[dataset]} & {label} & {cell(exact_correct, n)} & {cell(nonminimal_correct, n)} & "
                f"{cell(complete_wrong, n)} & {cell(incomplete_correct, n)} & {cell(incomplete_wrong, n)} & {disjoint} \\\\"
            )
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup", r"\end{landscape}"])
    return "\n".join(lines) + "\n"


def conditional_table(overall: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Conditional stage-wise TEST rates (\%). Retrieval statistics use support-bearing questions only.}",
        r"\label{7:tab_stagewise_conditional}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & Retrieval fail & Exact & Non-minimal & Wrong $\mid$ complete & Wrong $\mid$ exact & Correct $\mid$ incomplete & Correct $\mid$ disjoint \\",
        r"\midrule",
    ]
    for method, setting, label in CONDITIONS:
        stats = overall[f"{method}/{setting}"]
        lines.append(
            f"{label} & {pct(stats['retrieval_failure_rate'])} & {pct(stats['exact_retrieval_rate'])} & "
            f"{pct(stats['complete_nonminimal_rate'])} & {pct(stats['downstream_failure_given_complete'])} & "
            f"{pct(stats['downstream_failure_given_exact'])} & {pct(stats['answer_accuracy_given_incomplete'])} & "
            f"{pct(stats['answer_accuracy_given_disjoint'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stage-wise TEST error analysis",
        "",
        "All counts below were joined from frozen per-question retrieval selections and answer predictions; no count was derived from aggregate metrics. The denominator is the 3,509 TEST questions with at least one non-empty reference support.",
        "",
        "| Method | Retrieval failure | Exact | Complete non-minimal | Wrong given complete | Wrong given exact | Correct given incomplete | Correct given disjoint | Mean missing | Mean additional |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, setting, label in CONDITIONS:
        stats = summary["overall"][f"{method}/{setting}"]
        lines.append(
            f"| {label.replace('$', '')} | {pct(stats['retrieval_failure_rate'])}% | {pct(stats['exact_retrieval_rate'])}% | "
            f"{pct(stats['complete_nonminimal_rate'])}% | {pct(stats['downstream_failure_given_complete'])}% | "
            f"{pct(stats['downstream_failure_given_exact'])}% | {pct(stats['answer_accuracy_given_incomplete'])}% | "
            f"{pct(stats['answer_accuracy_given_disjoint'])}% | {stats['mean_missing_units']:.3f} | {stats['mean_additional_units']:.3f} |"
        )
    lines.extend([
        "",
        "Ontology reference alternatives are evaluated as alternatives, not unioned. For true ASK questions, the frozen deterministic reasoner is additionally executed over retrieved support; a deletion-minimal entailing subset is treated as a valid alternative proof. This prevents a reasoner-validated alternative proof from being counted as a retrieval failure. The reasoner check does not apply to SELECT questions or establish negative entailment.",
        "",
        "The `summary.json` file contains per-dataset, modality, hop, paired-transition, ontology-routing, and exact cell counts used for manuscript interpretation.",
    ])
    return "\n".join(lines) + "\n"


def analyze(output_dir: Path) -> dict[str, Any]:
    source_hashes = validate_frozen_sources()
    predictions = load_jsonl(FINAL_DIR / "predictions.jsonl")
    scored = load_jsonl(FINAL_DIR / "per_example_end_to_end.jsonl")
    inputs = load_jsonl(FINAL_DIR / "frozen_reader_inputs.jsonl")
    selections = load_jsonl(SELECTION_DIR / "test_predictions_frozen.jsonl")
    gold_rows = load_jsonl(GOLD_DIR / "per_example_test_retrieval.jsonl")

    prediction_by_key = {(row["example_id"], row["method"], row["setting"]): row for row in predictions}
    scored_by_key = {(row["example_id"], row["method"], row["setting"]): row for row in scored}
    input_by_key = {(row["example_id"], row["method"], row["setting"]): row for row in inputs}
    selection_by_id = {row["example_id"]: row for row in selections}
    gold_by_id = {row["example_id"]: row for row in gold_rows}
    if len(selection_by_id) != 4249 or len(gold_by_id) != 4249:
        raise ValueError("Expected 4,249 unique TEST questions")

    # Answer gold is opened only after validating/loading the complete frozen
    # predictions. It is used solely to guard positive ASK entailment checks.
    ids_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in selections:
        ids_by_dataset[str(row["dataset"])].add(str(row["example_id"]))
    gold_answer_by_id: dict[str, str] = {}
    for dataset, ids in ids_by_dataset.items():
        answers = production.load_gold_answers(dataset, ROOT / production.DATASET_INFO[dataset][2], ids)
        gold_answer_by_id.update(answers)

    rows: list[dict[str, Any]] = []
    for method, setting, _label in CONDITIONS:
        for example_id, gold_row in gold_by_id.items():
            alternatives = gold_row.get("gold_explanations") or []
            if not unique_sets(alternatives):
                continue
            key = (example_id, method, setting)
            prediction = prediction_by_key[key]
            score = scored_by_key[key]
            frozen_input = input_by_key[key]
            selected = selection_by_id[example_id]["methods"][method][setting]["retrieved_evidence_units"]
            if list(prediction["selected_support"]) != list(selected) or list(frozen_input["selected_support"]) != list(selected):
                raise ValueError(f"Frozen support mismatch for {key}")
            metadata = dict(frozen_input["metadata"])
            domain = str(prediction["domain"])
            classified = classify_support(
                selected,
                alternatives,
                domain=domain,
                metadata=metadata,
                gold_answer=gold_answer_by_id.get(example_id, ""),
            )
            rows.append({
                "dataset": prediction["dataset"],
                "domain": domain,
                "hop": metadata.get("hop") or ("1hop" if "_1hop" in prediction["dataset"] else "2hop"),
                "example_id": example_id,
                "method": method,
                "setting": setting,
                "answer_correct": float(score["answer_em"]) == 1.0,
                "answer_em": float(score["answer_em"]),
                "answer_source": prediction.get("answer_source", ""),
                "query_form": query_form(metadata) if domain == "ontology" else "text",
                "gold_answer": gold_answer_by_id.get(example_id, ""),
                **classified,
            })

    expected_rows = 3509 * len(CONDITIONS)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} stage-wise rows, found {len(rows)}")

    overall: dict[str, Any] = {}
    per_dataset: dict[str, Any] = {}
    by_domain: dict[str, Any] = {}
    by_hop: dict[str, Any] = {}
    ontology_by_hop: dict[str, Any] = {}
    routing: dict[str, Any] = {}
    for method, setting, _label in CONDITIONS:
        condition_rows = [row for row in rows if row["method"] == method and row["setting"] == setting]
        name = f"{method}/{setting}"
        overall[name] = aggregate(condition_rows)
        per_dataset[name] = group_aggregates(condition_rows, "dataset")
        by_domain[name] = group_aggregates(condition_rows, "domain")
        by_hop[name] = group_aggregates(condition_rows, "hop")
        ontology_by_hop[name] = group_aggregates(
            [row for row in condition_rows if row["domain"] == "ontology"], "hop"
        )
        routing[name] = ontology_routing(condition_rows)

    summary = {
        "schema_version": "stagewise_test_error_analysis_v1",
        "status": "complete_evaluation_only",
        "definitions": {
            "answer_correct": "Answer EM = 1",
            "retrieval_success": "Retrieved support contains at least one complete reference explanation, or a true ASK query is entailed by a deterministic reasoner-validated alternative proof.",
            "reference_selection_for_distances": "Minimize missing units, then additional units, then reference size, then persisted reference order.",
            "reasoner_scope": "Positive ASK entailment only; SELECT and negative ASK questions remain reference-explanation based.",
        },
        "source_hashes": source_hashes,
        "denominators": {"answer_questions": 4249, "support_bearing_questions_per_method": 3509, "derived_rows": len(rows)},
        "overall": overall,
        "per_dataset": per_dataset,
        "by_domain": by_domain,
        "by_hop": by_hop,
        "ontology_by_hop": ontology_by_hop,
        "ontology_routing": routing,
        "paired_transitions": {
            "cross_encoder_k1_to_sageqa_k1": paired_transitions(rows, ("cross_encoder", "k1"), ("final_sageqa", "k1")),
            "sageqa_k1_to_adaptive": paired_transitions(rows, ("final_sageqa", "k1"), ("final_sageqa", "adaptive")),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_example_stagewise.jsonl", rows)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "stagewise_table.tex").write_text(latex_table(per_dataset), encoding="utf-8", newline="\n")
    (output_dir / "conditional_rates_table.tex").write_text(conditional_table(overall), encoding="utf-8", newline="\n")
    (output_dir / "SUMMARY.md").write_text(summary_markdown(summary), encoding="utf-8", newline="\n")

    csv_fields = [
        "dataset", "domain", "hop", "example_id", "method", "setting", "retrieval_outcome",
        "answer_correct", "answer_source", "query_form", "retrieved_size", "reference_size",
        "missing_units", "additional_units", "reference_alternative_count",
        "reasoner_validated_alternative", "minimal_reasoner_proof_size",
    ]
    with (output_dir / "per_example_stagewise.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    derived = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    manifest = {
        "schema_version": "stagewise_test_error_analysis_manifest_v1",
        "status": "complete_evaluation_only",
        "frozen_sources": source_hashes,
        "derived_files": derived,
        "prediction_or_retrieval_generation_performed": False,
        "aggregate_metrics_used_to_derive_counts": False,
    }
    write_json(output_dir / "artifact_manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = analyze(args.output_dir.resolve())
    print(json.dumps({"status": summary["status"], "denominators": summary["denominators"]}, indent=2))


if __name__ == "__main__":
    main()
