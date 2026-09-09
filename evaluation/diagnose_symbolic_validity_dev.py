"""DEV-only diagnosis of existing symbolic structural-validity indicators.

This evaluator consumes frozen rankings/candidate rows and the preceding symbolic
mechanism diagnostic.  It does not train, score a neural model, generate
candidates, generate answers, or access TEST data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_gnn_subgraph_retriever import (  # noqa: E402
    _bridge_overlap_score,
    _candidate_question_overlap,
    _comparison_question_score,
    _cross_page_score,
    _kg_connectivity_score,
    _parse_sent_unit,
    _question_title_coverage,
    _size_chain_penalty,
    fact_rule_mix_symbolic,
    query_entity_coverage,
    query_property_match,
    schema_count,
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
EFFECTS = ("fixed", "harmed")
EXPECTED_COUNTS = {"fixed": 113, "harmed": 21}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_key(dataset: str, example_id: str, units: list[str]) -> tuple[str, str, tuple[str, ...]]:
    return dataset, example_id, tuple(str(unit) for unit in units)


def frequency(cases: list[dict[str, Any]], indicator: str) -> dict[str, Any]:
    applicable = [case for case in cases if case["indicators"].get(indicator) is not None]
    count = sum(bool(case["indicators"][indicator]) for case in applicable)
    return {
        "count": count,
        "denominator": len(applicable),
        "frequency": count / len(applicable) if applicable else None,
        "not_applicable_or_not_defined": len(cases) - len(applicable),
    }


def summarize_domain(cases: list[dict[str, Any]], indicators: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for indicator in indicators:
        result[indicator] = {
            effect: frequency([case for case in cases if case["effect"] == effect], indicator)
            for effect in EFFECTS
        }
    return result


def load_candidate_rows(
    data_root: Path,
    needed: set[tuple[str, str, tuple[str, ...]]],
) -> tuple[dict[tuple[str, str, tuple[str, ...]], dict[str, Any]], list[dict[str, str]]]:
    rows: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    lineage = []
    for dataset in DATASETS:
        path = data_root / dataset / "dev_subgraph_retrieval.jsonl"
        if "test" in str(path).lower():
            raise ValueError(f"TEST path is forbidden: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        lineage.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                key = candidate_key(dataset, raw["example_id"], raw.get("subgraph_units", []))
                if key in needed:
                    # Explicit gold firewall for structural indicator computation.
                    rows[key] = {
                        "dataset": dataset,
                        "example_id": raw["example_id"],
                        "question": raw.get("question", ""),
                        "sparql_query": raw.get("sparql_query", ""),
                        "subgraph_units": list(raw.get("subgraph_units", []) or []),
                        "subgraph_size": int(raw.get("subgraph_size", len(raw.get("subgraph_units", []) or []))),
                        "graph_context_units": list(raw.get("graph_context_units", []) or []),
                        "symbolic_features": list(raw.get("symbolic_features", []) or []),
                    }
    missing = sorted(needed - set(rows))
    if missing:
        raise AssertionError(f"Missing {len(missing)} promoted candidate rows")
    return rows, lineage


def text_case(base: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    question = str(row["question"])
    units = list(row["subgraph_units"])
    kg_units = list(row["graph_context_units"])
    titles = [_parse_sent_unit(unit)[0] for unit in units if _parse_sent_unit(unit)[0]]
    q_title = float(_question_title_coverage(question, units))
    q_overlap = float(_candidate_question_overlap(question, units))
    cross_page = float(_cross_page_score(units))
    bridge = float(_bridge_overlap_score(units))
    kg_connectivity = float(_kg_connectivity_score(question, units, kg_units))
    comparison = float(_comparison_question_score(question, units))
    size_penalty = float(_size_chain_penalty(units))
    duplicate_page_only = len(units) >= 2 and len(set(titles)) <= 1
    return {
        **base,
        "promoted_candidate_units": units,
        "existing_measurements": {
            "question_title_coverage": q_title,
            "question_candidate_overlap": q_overlap,
            "cross_page_score": cross_page,
            "bridge_overlap_score": bridge,
            "kg_connectivity_score": kg_connectivity,
            "comparison_question_score": comparison,
            "size_chain_penalty": size_penalty,
        },
        "indicators": {
            "question_title_coverage_positive": q_title > 0.0,
            "question_title_coverage_complete": math.isclose(q_title, 1.0, abs_tol=1e-12),
            "question_candidate_overlap_positive": q_overlap > 0.0,
            "question_candidate_overlap_complete": math.isclose(q_overlap, 1.0, abs_tol=1e-12),
            "cross_page_support": cross_page >= 0.8,
            "valid_bridge_connectivity": bridge > 0.0,
            "no_bridge_connectivity": math.isclose(bridge, 0.0, abs_tol=1e-12),
            "kg_bridge_connectivity": kg_connectivity > 0.0,
            "no_kg_bridge_connectivity": math.isclose(kg_connectivity, 0.0, abs_tol=1e-12),
            "comparison_coverage": comparison > 0.0,
            "excessive_support_size": size_penalty > 0.0,
            "duplicate_page_only_redundancy": duplicate_page_only,
        },
        "requested_concepts_not_separately_defined": {
            "complete_query_to_support_chain": "Text-Chain defines graded coverage/connectivity terms, not this Boolean.",
            "coverage_of_all_required_query_linked_components": "No required-component set is defined by Text-Chain.",
            "disconnected_evidence": "No unified disconnection Boolean is defined; exact zero bridge indicators are reported separately.",
            "partial_chain_only": "Text-Chain does not define a partial-versus-complete chain state.",
            "structural_redundancy": "Only the existing duplicate-page and excessive-size penalties are reported.",
        },
    }


def ontology_case(
    base: dict[str, Any],
    row: dict[str, Any],
    persisted_proof_bonus: float,
) -> dict[str, Any]:
    proof_complete = math.isclose(persisted_proof_bonus, 0.180, rel_tol=0.0, abs_tol=1e-10)
    proof_absent = math.isclose(persisted_proof_bonus, 0.0, rel_tol=0.0, abs_tol=1e-10)
    if not (proof_complete or proof_absent):
        raise AssertionError(f"Unexpected persisted proof bonus: {persisted_proof_bonus}")
    entity_coverage = float(query_entity_coverage(row))
    n_schema = int(schema_count(row))
    size = int(row["subgraph_size"])
    return {
        **base,
        "promoted_candidate_units": list(row["subgraph_units"]),
        "existing_measurements": {
            "persisted_proof_entailment_bonus": persisted_proof_bonus,
            "query_entity_coverage": entity_coverage,
            "schema_count": n_schema,
            "subgraph_size": size,
        },
        "indicators": {
            "complete_proof_consistent_support": proof_complete,
            "proof_incompleteness": proof_absent,
            "query_property_match": bool(query_property_match(row)),
            "complete_query_entity_coverage": math.isclose(entity_coverage, 1.0, abs_tol=1e-12),
            "fact_rule_mix": bool(fact_rule_mix_symbolic(row)),
            "oversized_support": size > 2,
            "multiple_schema_axioms": n_schema > 1,
        },
        "requested_concepts_not_separately_defined": {
            "all_required_proof_dependencies": "Entailment implies sufficient dependencies, but the reranker does not expose dependency identity as a separate indicator.",
            "valid_ontology_connectivity": "The Proof reranker exposes entailment and fact/rule mixture, not a separate connectivity Boolean.",
            "partial_proof_only": "The existing proof score is binary (entailed or not entailed), with no partial-proof state.",
            "irrelevant_extra_axioms": "The method penalizes multiple schema axioms as often-noisy supersets but does not prove irrelevance.",
        },
    }


def pct(value: dict[str, Any]) -> str:
    if value["frequency"] is None:
        return "not defined"
    return f'{value["count"]}/{value["denominator"]} ({100.0 * value["frequency"]:.1f}%)'


def markdown_table(summary: dict[str, Any]) -> list[str]:
    lines = ["| Existing indicator | Fixes | Harms |", "|---|---:|---:|"]
    for indicator, values in summary.items():
        lines.append(f"| `{indicator}` | {pct(values['fixed'])} | {pct(values['harmed'])} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohorts",
        type=Path,
        default=ROOT / "outputs/diagnostics/production_generator_d_v1_symbolic_mechanism_dev/fixed_vs_harmed.json",
    )
    parser.add_argument(
        "--component-analysis",
        type=Path,
        default=ROOT / "outputs/diagnostics/production_generator_d_v1_symbolic_mechanism_dev/symbolic_component_analysis.json",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/production_generator_d_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/diagnostics/production_generator_d_v1_symbolic_validity_diagnosis",
    )
    args = parser.parse_args()

    for path in (args.cohorts, args.component_analysis, args.data_root):
        if "test" in str(path).lower():
            raise ValueError(f"TEST path is forbidden: {path}")

    cohort_artifact = json.loads(args.cohorts.read_text(encoding="utf-8"))
    if cohort_artifact.get("split") != "dev":
        raise AssertionError("Cohort artifact is not DEV")
    flips = [row for row in cohort_artifact["top1_flips"] if row["effect"] in EFFECTS]
    counts = Counter(row["effect"] for row in flips)
    if dict(counts) != EXPECTED_COUNTS:
        raise AssertionError(f"Expected exact 113/21 cohorts, found {dict(counts)}")

    component_artifact = json.loads(args.component_analysis.read_text(encoding="utf-8"))
    if component_artifact.get("split") != "dev" or component_artifact.get("test_data_accessed"):
        raise AssertionError("Component analysis is not a DEV-only artifact")
    promoted_components = {}
    for item in component_artifact["changed_candidates"]:
        if item.get("symbolic_top5_rank") == 1:
            promoted_components[candidate_key(item["dataset"], item["example_id"], item["subgraph_units"])] = item

    needed = {
        candidate_key(row["dataset"], row["example_id"], row["symbolic_top1_units"])
        for row in flips
    }
    candidate_rows, candidate_lineage = load_candidate_rows(args.data_root, needed)

    cases = []
    for flip in flips:
        key = candidate_key(flip["dataset"], flip["example_id"], flip["symbolic_top1_units"])
        base = {
            "dataset": flip["dataset"],
            "domain": flip["domain"],
            "hop": flip["hop"],
            "example_id": flip["example_id"],
            "effect": flip["effect"],
        }
        row = candidate_rows[key]
        if flip["domain"] == "text":
            cases.append(text_case(base, row))
        else:
            component = promoted_components.get(key)
            if not component or not component.get("term_contributions"):
                raise AssertionError(f"Missing persisted proof contribution for {key[:2]}")
            proof_bonus = component["term_contributions"].get("proof_entailment_bonus")
            if proof_bonus is None:
                raise AssertionError(f"Unavailable persisted proof contribution for {key[:2]}")
            cases.append(ontology_case(base, row, float(proof_bonus)))

    text_cases = [case for case in cases if case["domain"] == "text"]
    ontology_cases = [case for case in cases if case["domain"] == "ontology"]
    text_indicators = list(text_cases[0]["indicators"])
    ontology_indicators = list(ontology_cases[0]["indicators"])
    text_summary = summarize_domain(text_cases, text_indicators)
    ontology_summary = summarize_domain(ontology_cases, ontology_indicators)

    proof_fix = ontology_summary["complete_proof_consistent_support"]["fixed"]
    proof_harm = ontology_summary["complete_proof_consistent_support"]["harmed"]
    proof_gap = proof_fix["frequency"] - proof_harm["frequency"]
    decision = {
        "outcome": "recommend_one_deterministic_symbolic_gate",
        "condition": "complete_proof_consistent_support",
        "scope": "ontology symbolic overrides only",
        "conceptual_rule": "Permit an ontology symbolic override only when its winning candidate receives the existing deterministic Proof entailment bonus; otherwise retain the GNN ordering.",
        "gate_evaluated": False,
        "text_rule_recommended": False,
        "basis": {
            "ontology_fixes": proof_fix,
            "ontology_harms": proof_harm,
            "absolute_frequency_gap": proof_gap,
            "note": "This is a descriptive DEV diagnosis; the ontology-harm denominator is four. No indicator combinations or thresholds were searched.",
        },
    }

    lineage = {
        "cohorts": {"path": str(args.cohorts.relative_to(ROOT)), "sha256": sha256(args.cohorts)},
        "component_analysis": {
            "path": str(args.component_analysis.relative_to(ROOT)),
            "sha256": sha256(args.component_analysis),
            "use": "persisted promoted-candidate Proof entailment bonus; no proof or answer inference rerun",
        },
        "dev_candidate_files": candidate_lineage,
    }
    invariants = {
        "split": "dev",
        "exact_cohorts": dict(counts),
        "test_data_accessed": False,
        "training_run": False,
        "neural_inference_run": False,
        "symbolic_proof_inference_run": False,
        "candidate_generation_run": False,
        "weight_tuning_run": False,
        "parameter_search_run": False,
        "answer_generation_run": False,
        "gate_evaluated": False,
        "gold_fields_presented_to_indicator_functions": False,
    }

    text_output = {
        "schema_version": "production_generator_d_v1_text_symbolic_validity_v1",
        "invariants": invariants,
        "cohort_counts": Counter(case["effect"] for case in text_cases),
        "indicator_definitions": {
            "source": "Existing sageqa_text_chain helper outputs and exact penalty states; no composite chain rule was created.",
            "not_defined_policy": "Requested concepts without an existing deterministic implementation are recorded per case but excluded from frequencies.",
        },
        "frequencies": text_summary,
        "cases": text_cases,
        "lineage": lineage,
    }
    ontology_output = {
        "schema_version": "production_generator_d_v1_ontology_symbolic_validity_v1",
        "invariants": invariants,
        "cohort_counts": Counter(case["effect"] for case in ontology_cases),
        "indicator_definitions": {
            "complete_proof_consistent_support": "The promoted candidate's persisted proof_entailment_bonus equals the frozen Proof reranker's 0.180 deterministic entailment bonus.",
            "proof_incompleteness": "The promoted candidate's persisted deterministic entailment bonus is zero.",
            "other_indicators": "Existing query_property_match, query_entity_coverage, fact_rule_mix_symbolic, size, and schema-count concepts.",
        },
        "frequencies": ontology_summary,
        "cases": ontology_cases,
        "lineage": lineage,
    }
    comparison_output = {
        "schema_version": "production_generator_d_v1_symbolic_validity_comparison_v1",
        "invariants": invariants,
        "total_cohorts": dict(counts),
        "domain_cohorts": {
            domain: dict(Counter(case["effect"] for case in cases if case["domain"] == domain))
            for domain in ("text", "ontology")
        },
        "text_frequencies": text_summary,
        "ontology_frequencies": ontology_summary,
        "single_condition_assessment": decision,
        "cases": [
            {
                "dataset": case["dataset"],
                "domain": case["domain"],
                "hop": case["hop"],
                "example_id": case["example_id"],
                "effect": case["effect"],
                "indicators": case["indicators"],
            }
            for case in cases
        ],
        "lineage": lineage,
    }

    summary_lines = [
        "# Symbolic structural-validity diagnosis (DEV only)",
        "",
        "## Outcome",
        "",
        "One existing condition strongly separates the ontology cohorts: `complete_proof_consistent_support` occurs in "
        f"{pct(proof_fix)} of ontology fixes and {pct(proof_harm)} of ontology harms (an {100.0 * proof_gap:.1f}-percentage-point gap). "
        "This supports one ontology-only deterministic gate recommendation. The four-case ontology-harm denominator makes the result descriptive and motivates evaluation of the gate in a separately authorized study; no gate was evaluated here.",
        "",
        "Text-Chain indicators do not clearly separate text fixes from text harms. In particular, cross-page support is 16/16 versus 17/17, bridge connectivity is 15/16 versus 17/17, and KG bridge connectivity is 14/16 versus 15/17. No text gate is recommended.",
        "",
        "## Cohorts",
        "",
        "The input cohort artifact reproduced exactly 113 fixes and 21 harms. By branch, text contains 16 fixes and 17 harms; ontology contains 97 fixes and 4 harms.",
        "",
        "## Text: existing Text-Chain indicators",
        "",
        *markdown_table(text_summary),
        "",
        "Text-Chain has no separately implemented Boolean for a complete query-to-support chain, all required query-linked components, unified disconnection, or partial-chain-only status. Those labels were not manufactured from new conjunctions. The exact existing coverage, bridge, cross-page, comparison, size, and duplicate-page indicators are reported instead.",
        "",
        "## Ontology: existing Proof indicators",
        "",
        *markdown_table(ontology_summary),
        "",
        "The Proof reranker has a deterministic binary entailment bonus, but it does not separately expose dependency identity, a partial-proof state, ontology connectivity, or proven irrelevance of extra axioms. Existing fact/rule mixture, query coverage, size, and multiple-schema indicators are reported without relabelling them as those unavailable concepts.",
        "",
        "## Protocol",
        "",
        "This diagnostic read frozen DEV cohort/component artifacts and the matching DEV candidate rows. Indicator functions received a gold-stripped row projection. It performed no training, neural inference, proof inference, candidate generation, tuning, parameter search, answer generation, TEST access, or gate evaluation.",
    ]
    decision_lines = [
        "# Mechanism decision",
        "",
        "## Recommend one deterministic symbolic gate (not evaluated)",
        "",
        "For the ontology branch only: permit a symbolic override only when the winning symbolic candidate satisfies the existing deterministic Proof entailment condition (`proof_entailment_bonus = 0.180`); otherwise retain the GNN ordering.",
        "",
        f"Rationale: complete proof-consistent support occurs in {pct(proof_fix)} of ontology fixes and {pct(proof_harm)} of ontology harms. This is the only existing semantic validity condition with a strong descriptive separation. No combinations, thresholds, alternative rules, or text gates were searched.",
        "",
        "The gate has not been evaluated. This file is a recommendation from a cheap DEV-only diagnosis, not evidence of downstream metric improvement. Stop after this diagnosis.",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, tuple[str, str | dict[str, Any]]] = {
        "summary.md": ("text", "\n".join(summary_lines) + "\n"),
        "text_validity_analysis.json": ("json", text_output),
        "ontology_validity_analysis.json": ("json", ontology_output),
        "fixes_vs_harms.json": ("json", comparison_output),
        "mechanism_decision.md": ("text", "\n".join(decision_lines) + "\n"),
    }
    for name, (kind, content) in outputs.items():
        path = args.output_dir / name
        if kind == "json":
            path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            path.write_text(str(content), encoding="utf-8")

    print(json.dumps({
        "cohorts": dict(counts),
        "domain_cohorts": comparison_output["domain_cohorts"],
        "proof_condition": {"fixes": proof_fix, "harms": proof_harm},
        "decision": decision["outcome"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
