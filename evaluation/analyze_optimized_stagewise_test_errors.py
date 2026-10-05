"""Stage-wise error analysis for the frozen optimized SAGE-QA TEST run.

This evaluation-only driver reuses the historical Chapter 7 support taxonomy.
It verifies the optimized retrieval and answer freezes before opening answer or
support gold, recomputes only optimized SAGE-QA k=1/adaptive rows, and carries
the persisted historical Cross-Encoder aggregates forward unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import analyze_stagewise_test_errors as historical
from generation import run_production_test_answer_generation as production


ANSWER_DIR = ROOT / "outputs/final_results/symbolic_coefficients_original_v1_test_end_to_end"
SELECTION_DIR = Path(
    r"C:\Users\julie\github\PhD\SAGEQA_GPU_handoff\outputs\final_results"
    r"\symbolic_coefficients_original_v1_test"
)
GOLD_PATH = (
    ROOT
    / "outputs/final_results/production_generator_d_v2_hard_pair_test_retrieval"
    / "per_example_test_retrieval.jsonl"
)
HISTORICAL_DIR = ROOT / "outputs/final_results/final_manuscript_stagewise_error_analysis"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/final_results/symbolic_coefficients_original_v1_stagewise_error_analysis"
)
EXPECTED_ANSWER_QUESTIONS = 4249
EXPECTED_SUPPORT_QUESTIONS = 3509
CONDITIONS = (
    ("sageqa_optimized", "k1", "Optimized SAGE-QA ($k=1$)"),
    ("sageqa_optimized", "adaptive", "Optimized SAGE-QA (adaptive)"),
)
CROSS_ENCODER = ("cross_encoder", "k1", "Cross-Encoder ($k=1$)")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_hash(manifest: Mapping[str, Any], name: str) -> str:
    for group in ("files", "frozen_inputs", "derived_files"):
        files = manifest.get(group, {})
        if not isinstance(files, Mapping) or name not in files:
            continue
        value = files[name]
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            return str(value["sha256"])
    raise ValueError(f"Manifest has no hash for {name}")


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = historical.sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, found {actual}")
    return actual


def validate_frozen_sources(
    answer_dir: Path, selection_dir: Path, gold_path: Path, historical_dir: Path
) -> dict[str, str]:
    """Validate every frozen input before any original TEST gold is opened."""
    answer_freeze = _json(answer_dir / "generation_freeze.json")
    if answer_freeze.get("status") != "complete_frozen":
        raise ValueError("Optimized answer predictions are not complete_frozen")
    if int(answer_freeze.get("prediction_count", -1)) != 2 * EXPECTED_ANSWER_QUESTIONS:
        raise ValueError("Optimized answer freeze does not contain exactly 8,498 rows")
    expected_configurations = [
        {"method": method, "setting": setting} for method, setting, _label in CONDITIONS
    ]
    if answer_freeze.get("configurations") != expected_configurations:
        raise ValueError("Optimized answer freeze has an unexpected configuration scope")

    source_hashes = {
        "optimized_answer_predictions.jsonl": _require_hash(
            answer_dir / "predictions.jsonl",
            str(answer_freeze["predictions_sha256"]),
            "Optimized answer predictions",
        ),
        "optimized_frozen_reader_inputs.jsonl": _require_hash(
            answer_dir / "frozen_reader_inputs.jsonl",
            str(answer_freeze["reader_inputs_sha256"]),
            "Optimized reader inputs",
        ),
    }

    selection_manifest = _json(selection_dir / "artifact_manifest.json")
    selection_freeze = _json(selection_dir / "generation_freeze.json")
    selection_path = selection_dir / "test_predictions_frozen.jsonl"
    selection_hash = _require_hash(
        selection_path,
        _manifest_hash(selection_manifest, selection_path.name),
        "Optimized retrieval predictions",
    )
    if selection_hash != selection_freeze.get("predictions_sha256"):
        raise ValueError("Optimized retrieval manifest and freeze disagree")
    if selection_hash != answer_freeze.get("optimized_retrieval_sha256"):
        raise ValueError("Answer freeze does not reference the optimized retrieval freeze")
    if selection_freeze.get("status") != "complete_frozen_before_gold_join":
        raise ValueError("Optimized retrieval was not frozen before its gold join")
    if selection_freeze.get("gold_accessed_before_freeze") is not False:
        raise ValueError("Optimized retrieval freeze lacks the gold-blind attestation")
    if int(selection_freeze.get("prediction_examples", -1)) != EXPECTED_ANSWER_QUESTIONS:
        raise ValueError("Optimized retrieval freeze does not contain 4,249 questions")
    source_hashes["optimized_retrieval_predictions.jsonl"] = selection_hash

    gold_manifest = _json(gold_path.parent / "artifact_manifest.json")
    source_hashes["original_test_support_gold.jsonl"] = _require_hash(
        gold_path, _manifest_hash(gold_manifest, gold_path.name), "Original TEST support gold"
    )

    historical_manifest = _json(historical_dir / "artifact_manifest.json")
    for name in ("summary.json", "stagewise_table.tex", "conditional_rates_table.tex"):
        source_hashes[f"historical_{name}"] = _require_hash(
            historical_dir / name,
            _manifest_hash(historical_manifest, name),
            f"Historical {name}",
        )

    # The evaluated rows are derived after the immutable prediction freeze.  A
    # current hash is recorded, and every answer score is independently checked
    # against the original evaluator below rather than trusted as an input.
    source_hashes["optimized_per_example_end_to_end.jsonl"] = historical.sha256(
        answer_dir / "per_example_end_to_end.jsonl"
    )
    return source_hashes


def _unique_by_key(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str], label: str
) -> dict[tuple[str, ...], Mapping[str, Any]]:
    result: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        if key in result:
            raise ValueError(f"Duplicate {label} key: {key}")
        result[key] = row
    return result


def _stagewise_row(
    stats: Mapping[str, Any], label: str, *, dataset: str | None = None
) -> str:
    n, cells = int(stats["n"]), stats["cells"]
    values = (
        historical.cell(int(cells["exact_correct"]), n),
        historical.cell(int(cells["complete_nonminimal_correct"]), n),
        historical.cell(int(cells["exact_wrong"]) + int(cells["complete_nonminimal_wrong"]), n),
        historical.cell(int(cells["incomplete_correct"]), n),
        historical.cell(int(cells["incomplete_wrong"]), n),
        f"{cells['disjoint_correct']}/{cells['disjoint_wrong']} "
        f"({100 * int(stats['counts']['disjoint']) / n:.1f}\\%)",
    )
    prefix = f"{dataset} & " if dataset is not None else ""
    return prefix + label + " & " + " & ".join(values) + r" \\"


def stagewise_table(
    per_dataset: Mapping[str, Any], historical_per_dataset: Mapping[str, Any]
) -> str:
    lines = [
        r"\begin{landscape}",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}llrrrrrr@{}}",
        r"\caption{Stage-wise support-retrieval and answer outcomes on the original support-bearing TEST cohort. Historical Cross-Encoder values are retained unchanged; optimized SAGE-QA values are recomputed from frozen predictions. Each cell gives count (percentage within the dataset--method row). The Disjoint column reports correct/wrong counts.}\label{7:tab_stagewise_errors}\\",
        r"\toprule",
        r"Dataset & Method & Exact+corr. & Non-min.+corr. & Complete+wrong & Incomp.+corr. & Incomp.+wrong & Disjoint c/w \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Dataset & Method & Exact+corr. & Non-min.+corr. & Complete+wrong & Incomp.+corr. & Incomp.+wrong & Disjoint c/w \\",
        r"\midrule",
        r"\endhead",
    ]
    for dataset in historical.DATASET_ORDER:
        display = historical.DISPLAY[dataset]
        lines.append(
            _stagewise_row(
                historical_per_dataset["cross_encoder/k1"][dataset],
                CROSS_ENCODER[2],
                dataset=display,
            )
        )
        for method, setting, label in CONDITIONS:
            lines.append(
                _stagewise_row(per_dataset[f"{method}/{setting}"][dataset], label, dataset=display)
            )
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup", r"\end{landscape}"])
    return "\n".join(lines) + "\n"


def conditional_table(
    overall: Mapping[str, Any], historical_overall: Mapping[str, Any]
) -> str:
    combined = [(CROSS_ENCODER, historical_overall["cross_encoder/k1"])] + [
        (condition, overall[f"{condition[0]}/{condition[1]}"]) for condition in CONDITIONS
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Conditional stage-wise TEST rates (\%). Retrieval statistics use the original support-bearing questions only.}",
        r"\label{7:tab_stagewise_conditional}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & Retrieval fail & Exact & Non-minimal & Wrong $\mid$ complete & Wrong $\mid$ exact & Correct $\mid$ incomplete & Correct $\mid$ disjoint \\",
        r"\midrule",
    ]
    for (_method, _setting, label), stats in combined:
        lines.append(
            f"{label} & {historical.pct(stats['retrieval_failure_rate'])} & "
            f"{historical.pct(stats['exact_retrieval_rate'])} & "
            f"{historical.pct(stats['complete_nonminimal_rate'])} & "
            f"{historical.pct(stats['downstream_failure_given_complete'])} & "
            f"{historical.pct(stats['downstream_failure_given_exact'])} & "
            f"{historical.pct(stats['answer_accuracy_given_incomplete'])} & "
            f"{historical.pct(stats['answer_accuracy_given_disjoint'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def sageqa_stagewise_rows(per_dataset: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for dataset in historical.DATASET_ORDER:
        for method, setting, label in CONDITIONS:
            lines.append(
                _stagewise_row(
                    per_dataset[f"{method}/{setting}"][dataset],
                    label,
                    dataset=historical.DISPLAY[dataset],
                )
            )
    return "\n".join(lines) + "\n"


def summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Optimized SAGE-QA stage-wise TEST error analysis",
        "",
        "All optimized SAGE-QA counts were recomputed from hash-verified frozen per-question retrieval and answer predictions. The original TEST support cohort contains 3,509 questions. Historical Cross-Encoder aggregates were hash-verified and carried forward unchanged; they were not recomputed.",
        "",
        "| Method | Retrieval failure | Exact | Complete non-minimal | Wrong given complete | Wrong given exact | Correct given incomplete | Correct given disjoint | Mean missing | Mean additional |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cross = summary["historical_cross_encoder"]["overall"]
    rows = [(CROSS_ENCODER[2], cross)] + [
        (label, summary["overall"][f"{method}/{setting}"])
        for method, setting, label in CONDITIONS
    ]
    for label, stats in rows:
        lines.append(
            f"| {label.replace('$', '')} | {historical.pct(stats['retrieval_failure_rate'])}% | "
            f"{historical.pct(stats['exact_retrieval_rate'])}% | "
            f"{historical.pct(stats['complete_nonminimal_rate'])}% | "
            f"{historical.pct(stats['downstream_failure_given_complete'])}% | "
            f"{historical.pct(stats['downstream_failure_given_exact'])}% | "
            f"{historical.pct(stats['answer_accuracy_given_incomplete'])}% | "
            f"{historical.pct(stats['answer_accuracy_given_disjoint'])}% | "
            f"{stats['mean_missing_units']:.3f} | {stats['mean_additional_units']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Ontology explanations were treated as alternative valid references, never unioned. The historical positive-ASK reasoner rescue and exact/non-minimal/incomplete/disjoint definitions were reused without change.",
            "",
            "Regression gates: 8,498 frozen answer predictions, 4,249 aligned TEST questions, 3,509 support-bearing questions per method, 7,018 derived optimized rows, and zero saved-versus-recomputed answer-score mismatches.",
        ]
    )
    return "\n".join(lines) + "\n"


def manuscript_update_report(summary: Mapping[str, Any]) -> str:
    k1 = summary["overall"]["sageqa_optimized/k1"]
    adaptive = summary["overall"]["sageqa_optimized/adaptive"]
    transitions = summary["paired_transitions"]["optimized_k1_to_adaptive"]
    rescued_nonminimal = int(transitions.get("incomplete_to_complete_nonminimal", 0)) + int(
        transitions.get("disjoint_to_complete_nonminimal", 0)
    )
    rescued_exact = int(transitions.get("incomplete_to_exact", 0)) + int(
        transitions.get("disjoint_to_exact", 0)
    )
    adaptive_hop = summary["ontology_by_hop"]["sageqa_optimized/adaptive"]
    adaptive_routing = summary["ontology_routing"]["sageqa_optimized/adaptive"]
    deterministic = adaptive_routing["ask_deterministic_proof"]
    select = adaptive_routing["select_llm"]
    return "\n".join(
        [
            "# Chapter 7 numerical update audit",
            "",
            "Replace the two SAGE-QA method rows in the conditional-rate table; retain the historical Cross-Encoder row unchanged.",
            "",
            f"- Retrieval failure: optimized k=1 {historical.pct(k1['retrieval_failure_rate'])}%; adaptive {historical.pct(adaptive['retrieval_failure_rate'])}% (historical SAGE-QA: 48.1% and 43.9%).",
            f"- Exact retrieval: optimized k=1 {historical.pct(k1['exact_retrieval_rate'])}%; adaptive {historical.pct(adaptive['exact_retrieval_rate'])}% (historical: 34.6% and 31.9%).",
            f"- Complete non-minimal: optimized k=1 {historical.pct(k1['complete_nonminimal_rate'])}%; adaptive {historical.pct(adaptive['complete_nonminimal_rate'])}% (historical: 17.4% and 24.3%).",
            f"- Adaptive aggregation converts {rescued_nonminimal + rescued_exact} optimized k=1 failures to complete retrieval: {rescued_nonminimal} non-minimal and {rescued_exact} exact (historical: 147 = 113 + 34).",
            f"- Mean missing units change from {k1['mean_missing_units']:.3f} to {adaptive['mean_missing_units']:.3f}; mean additional units change from {k1['mean_additional_units']:.3f} to {adaptive['mean_additional_units']:.3f} (historical: 0.692 to 0.605 and 0.644 to 0.849).",
            f"- Adaptive answer failure given complete retrieval is {historical.pct(adaptive['downstream_failure_given_complete'])}%; given exact retrieval {historical.pct(adaptive['downstream_failure_given_exact'])}%; given complete non-minimal retrieval {historical.pct(adaptive['downstream_failure_given_nonminimal'])}% (historical: 27.5%, 22.5%, 34.0%).",
            f"- Adaptive answers correct without complete annotated support: {adaptive['answer_correct_without_complete_count']} ({historical.pct(adaptive['answer_correct_without_complete_rate'])}% of 3,509; historical: 536, 15.3%).",
            f"- Adaptive ontology retrieval failure: 1-hop {historical.pct(adaptive_hop['1hop']['retrieval_failure_rate'])}%; 2-hop {historical.pct(adaptive_hop['2hop']['retrieval_failure_rate'])}% (historical: 20.8% and 52.1%).",
            f"- Adaptive deterministic ASK-proof route: {deterministic['n']} cases and {deterministic['answer_errors']} answer errors (historical: 515 and 0).",
            f"- Adaptive sufficient-support ontology answer errors on the SELECT/LLM route: {select['complete_support_answer_errors']} (historical: 250); the deterministic ASK-proof and ASK-without-proof routes contribute zero such errors.",
            "",
            "Appendix update scope:",
            "",
            "- Replace the SAGE-QA k=1 and adaptive rows for nine datasets in the six-column stage-wise appendix table. The Pizza250 2-hop six-cell rows are unchanged for both methods.",
            "- Replace both SAGE-QA rows in the conditional-rates appendix table.",
            "- Retain all ten historical Cross-Encoder rows and the historical Cross-Encoder conditional-rate row unchanged.",
            "- Use `stagewise_sageqa_rows.tex` for the per-dataset replacements, `overall_stagewise_sageqa_rows.tex` for the two aggregate six-cell rows, and `conditional_rates_sageqa_rows.tex` for the two conditional-rate rows.",
        ]
    ) + "\n"


def analyze(
    output_dir: Path,
    *,
    answer_dir: Path = ANSWER_DIR,
    selection_dir: Path = SELECTION_DIR,
    gold_path: Path = GOLD_PATH,
    historical_dir: Path = HISTORICAL_DIR,
) -> dict[str, Any]:
    source_hashes = validate_frozen_sources(answer_dir, selection_dir, gold_path, historical_dir)

    predictions = historical.load_jsonl(answer_dir / "predictions.jsonl")
    frozen_inputs = historical.load_jsonl(answer_dir / "frozen_reader_inputs.jsonl")
    scored = historical.load_jsonl(answer_dir / "per_example_end_to_end.jsonl")
    selections = historical.load_jsonl(selection_dir / "test_predictions_frozen.jsonl")
    gold_rows = historical.load_jsonl(gold_path)

    key_fields = ("example_id", "method", "setting")
    prediction_by_key = _unique_by_key(predictions, key_fields, "prediction")
    input_by_key = _unique_by_key(frozen_inputs, key_fields, "reader input")
    score_by_key = _unique_by_key(scored, key_fields, "evaluated answer")
    selection_by_id = _unique_by_key(selections, ("example_id",), "retrieval selection")
    gold_by_id = _unique_by_key(gold_rows, ("example_id",), "support-gold")
    expected_keys = {
        (example_id, method, setting)
        for (example_id,) in selection_by_id
        for method, setting, _label in CONDITIONS
    }
    if len(selection_by_id) != EXPECTED_ANSWER_QUESTIONS or set(selection_by_id) != set(gold_by_id):
        raise ValueError("Original TEST retrieval and support-gold populations must align at 4,249")
    if set(prediction_by_key) != expected_keys or set(input_by_key) != expected_keys:
        raise ValueError("Optimized answer predictions/inputs do not cover the exact requested scope")
    if set(score_by_key) != expected_keys:
        raise ValueError("Saved optimized answer evaluation does not cover the exact requested scope")

    # Gold is first opened only after all frozen hashes and populations pass.
    ids_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in selections:
        ids_by_dataset[str(row["dataset"])].add(str(row["example_id"]))
    gold_answer_by_id: dict[str, str] = {}
    for dataset, ids in ids_by_dataset.items():
        gold_answer_by_id.update(
            production.load_gold_answers(dataset, ROOT / production.DATASET_INFO[dataset][2], ids)
        )

    support_ids = {
        example_id
        for (example_id,), row in gold_by_id.items()
        if historical.unique_sets(row.get("gold_explanations") or [])
    }
    if len(support_ids) != EXPECTED_SUPPORT_QUESTIONS:
        raise ValueError(
            f"Expected the original 3,509 support-bearing TEST questions, found {len(support_ids)}"
        )

    rows: list[dict[str, Any]] = []
    score_regression_mismatches = 0
    for method, setting, _label in CONDITIONS:
        for example_id in sorted(support_ids):
            key = (example_id, method, setting)
            prediction = prediction_by_key[key]
            frozen_input = input_by_key[key]
            saved_score = score_by_key[key]
            selection = selection_by_id[(example_id,)]
            selected = selection[setting]["retrieved_evidence_units"]
            if (
                list(prediction["selected_support"]) != list(selected)
                or list(frozen_input["selected_support"]) != list(selected)
            ):
                raise ValueError(f"Frozen support mismatch for {key}")
            gold_answer = gold_answer_by_id[example_id]
            recomputed = production._answer_scores(
                str(prediction["domain"]), str(prediction["predicted_answer"]), gold_answer
            )
            if (
                abs(float(saved_score["answer_em"]) - recomputed[0]) > 1e-12
                or abs(float(saved_score["answer_f1"]) - recomputed[1]) > 1e-12
            ):
                score_regression_mismatches += 1
            metadata = dict(frozen_input["metadata"])
            domain = str(prediction["domain"])
            classified = historical.classify_support(
                selected,
                gold_by_id[(example_id,)].get("gold_explanations") or [],
                domain=domain,
                metadata=metadata,
                gold_answer=gold_answer,
            )
            rows.append(
                {
                    "dataset": prediction["dataset"],
                    "domain": domain,
                    "hop": metadata.get("hop")
                    or ("1hop" if "_1hop" in str(prediction["dataset"]) else "2hop"),
                    "example_id": example_id,
                    "method": method,
                    "setting": setting,
                    "answer_correct": recomputed[0] == 1.0,
                    "answer_em": recomputed[0],
                    "answer_source": prediction.get("answer_source", ""),
                    "query_form": historical.query_form(metadata) if domain == "ontology" else "text",
                    "gold_answer": gold_answer,
                    **classified,
                }
            )
    if score_regression_mismatches:
        raise ValueError(
            f"Saved optimized answer evaluation disagrees with recomputation in "
            f"{score_regression_mismatches} rows"
        )
    if len(rows) != EXPECTED_SUPPORT_QUESTIONS * len(CONDITIONS):
        raise ValueError("Optimized stage-wise row count is not exactly 7,018")

    overall: dict[str, Any] = {}
    per_dataset: dict[str, Any] = {}
    by_domain: dict[str, Any] = {}
    by_hop: dict[str, Any] = {}
    ontology_by_hop: dict[str, Any] = {}
    routing: dict[str, Any] = {}
    for method, setting, _label in CONDITIONS:
        condition_rows = [
            row for row in rows if row["method"] == method and row["setting"] == setting
        ]
        name = f"{method}/{setting}"
        overall[name] = historical.aggregate(condition_rows)
        per_dataset[name] = historical.group_aggregates(condition_rows, "dataset")
        by_domain[name] = historical.group_aggregates(condition_rows, "domain")
        by_hop[name] = historical.group_aggregates(condition_rows, "hop")
        ontology_by_hop[name] = historical.group_aggregates(
            [row for row in condition_rows if row["domain"] == "ontology"], "hop"
        )
        routing[name] = historical.ontology_routing(condition_rows)

    old_summary = _json(historical_dir / "summary.json")
    transition = historical.paired_transitions(
        rows, ("sageqa_optimized", "k1"), ("sageqa_optimized", "adaptive")
    )
    summary = {
        "schema_version": "optimized_stagewise_test_error_analysis_v1",
        "status": "complete_evaluation_only",
        "definitions": old_summary["definitions"],
        "source_hashes": source_hashes,
        "denominators": {
            "answer_questions": EXPECTED_ANSWER_QUESTIONS,
            "support_bearing_questions_per_method": EXPECTED_SUPPORT_QUESTIONS,
            "recomputed_sageqa_rows": len(rows),
        },
        "regression_checks": {
            "saved_answer_score_mismatches": score_regression_mismatches,
            "historical_cross_encoder_recomputed": False,
            "historical_cross_encoder_source_summary_sha256": source_hashes[
                "historical_summary.json"
            ],
        },
        "historical_cross_encoder": {
            "overall": old_summary["overall"]["cross_encoder/k1"],
            "per_dataset": old_summary["per_dataset"]["cross_encoder/k1"],
        },
        "overall": overall,
        "per_dataset": per_dataset,
        "by_domain": by_domain,
        "by_hop": by_hop,
        "ontology_by_hop": ontology_by_hop,
        "ontology_routing": routing,
        "paired_transitions": {"optimized_k1_to_adaptive": transition},
        "historical_sageqa_for_change_audit": {
            "overall": {
                name: old_summary["overall"][name]
                for name in ("final_sageqa/k1", "final_sageqa/adaptive")
            },
            "ontology_by_hop": {
                name: old_summary["ontology_by_hop"][name]
                for name in ("final_sageqa/k1", "final_sageqa/adaptive")
            },
            "ontology_routing": {
                name: old_summary["ontology_routing"][name]
                for name in ("final_sageqa/k1", "final_sageqa/adaptive")
            },
            "paired_transitions": old_summary["paired_transitions"]["sageqa_k1_to_adaptive"],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    historical.write_jsonl(output_dir / "per_example_stagewise.jsonl", rows)
    historical.write_json(output_dir / "summary.json", summary)
    (output_dir / "stagewise_table.tex").write_text(
        stagewise_table(per_dataset, old_summary["per_dataset"]), encoding="utf-8", newline="\n"
    )
    (output_dir / "conditional_rates_table.tex").write_text(
        conditional_table(overall, old_summary["overall"]), encoding="utf-8", newline="\n"
    )
    (output_dir / "stagewise_sageqa_rows.tex").write_text(
        sageqa_stagewise_rows(per_dataset), encoding="utf-8", newline="\n"
    )
    (output_dir / "overall_stagewise_sageqa_rows.tex").write_text(
        "\n".join(
            _stagewise_row(overall[f"{method}/{setting}"], label)
            for method, setting, label in CONDITIONS
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "SUMMARY.md").write_text(
        summary_markdown(summary), encoding="utf-8", newline="\n"
    )
    (output_dir / "MANUSCRIPT_UPDATE_REPORT.md").write_text(
        manuscript_update_report(summary), encoding="utf-8", newline="\n"
    )
    (output_dir / "conditional_rates_sageqa_rows.tex").write_text(
        "\n".join(
            conditional_table(overall, old_summary["overall"]).splitlines()[11:13]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    csv_fields = [
        "dataset", "domain", "hop", "example_id", "method", "setting",
        "retrieval_outcome", "answer_correct", "answer_source", "query_form",
        "retrieved_size", "reference_size", "missing_units", "additional_units",
        "reference_alternative_count", "reasoner_validated_alternative",
        "minimal_reasoner_proof_size",
    ]
    with (output_dir / "per_example_stagewise.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    derived = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": historical.sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    historical.write_json(
        output_dir / "artifact_manifest.json",
        {
            "schema_version": "optimized_stagewise_test_error_analysis_manifest_v1",
            "status": "complete_evaluation_only",
            "frozen_sources": source_hashes,
            "derived_files": derived,
            "prediction_or_retrieval_generation_performed": False,
            "historical_cross_encoder_recomputed": False,
            "aggregate_metrics_used_to_derive_counts": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--answer-dir", type=Path, default=ANSWER_DIR)
    parser.add_argument("--selection-dir", type=Path, default=SELECTION_DIR)
    args = parser.parse_args()
    result = analyze(
        args.output_dir.resolve(),
        answer_dir=args.answer_dir.resolve(),
        selection_dir=args.selection_dir.resolve(),
    )
    print(json.dumps({"status": result["status"], "denominators": result["denominators"]}, indent=2))


if __name__ == "__main__":
    main()
