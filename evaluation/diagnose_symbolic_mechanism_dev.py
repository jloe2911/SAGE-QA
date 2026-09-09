"""Training-free decomposition of persisted DEV symbolic reranking flips."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_gnn_subgraph_retriever import (
    _bridge_overlap_score,
    _candidate_question_overlap,
    _comparison_question_score,
    _cross_page_score,
    _kg_connectivity_score,
    _parse_sent_unit,
    _question_title_coverage,
    _size_chain_penalty,
    extract_query_entities,
    extract_query_properties,
    fact_rule_mix_symbolic,
    query_entity_coverage,
    query_property_match,
    schema_count,
)


DATASETS = (
    ("HotpotQA", "text", "2hop"), ("2WikiMultiHopQA", "text", "2hop"),
    ("FamilyOWL_1hop", "ontology", "1hop"), ("FamilyOWL_2hop", "ontology", "2hop"),
    ("pizza_100_1hop", "ontology", "1hop"), ("pizza_100_2hop", "ontology", "2hop"),
    ("pizza_250_1hop", "ontology", "1hop"), ("pizza_250_2hop", "ontology", "2hop"),
    ("OWL2Bench_1hop", "ontology", "1hop"), ("OWL2Bench_2hop", "ontology", "2hop"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(units) -> tuple[str, ...]:
    return tuple(str(unit) for unit in units)


def complete(row: Mapping[str, Any]) -> bool:
    return bool(row.get("contains_any_gold_explanation")) or float(row.get("rank_target", 0.0)) >= 0.9


def text_terms(row: Mapping[str, Any]) -> dict[str, float]:
    question = str(row.get("question", ""))
    units = list(row.get("subgraph_units", []) or [])
    kg_units = list(row.get("graph_context_units", []) or [])
    titles = [_parse_sent_unit(unit)[0] for unit in units if _parse_sent_unit(unit)[0]]
    duplicate = -0.010 if len(units) >= 2 and len(set(titles)) <= 1 else 0.0
    return {
        "question_title_coverage_bonus": 0.030 * _question_title_coverage(question, units),
        "question_candidate_overlap_bonus": 0.020 * _candidate_question_overlap(question, units),
        "cross_page_bonus": 0.020 * _cross_page_score(units),
        "bridge_overlap_bonus": 0.025 * _bridge_overlap_score(units),
        "kg_connectivity_bonus": 0.030 * _kg_connectivity_score(question, units, kg_units),
        "comparison_coverage_bonus": 0.020 * _comparison_question_score(question, units),
        "support_size_penalty": -float(_size_chain_penalty(units)),
        "duplicate_page_penalty": duplicate,
    }


def ontology_nonproof_terms(row: Mapping[str, Any]) -> dict[str, float]:
    units = list(row.get("subgraph_units", []) or [])
    size = len(units)
    oversize = max(0, size - 2)
    query = str(row.get("sparql_query", "") or row.get("question", ""))
    query_tokens = {
        token.lower()
        for token in __import__("re").findall(r"[A-Za-z_][A-Za-z0-9_]*", query)
        if len(token) > 2 and not token.lower().startswith("http")
    }
    unit_text = " ".join(str(unit).lower() for unit in units)
    query_coverage = sum(token in unit_text for token in query_tokens) / max(len(query_tokens), 1)
    prop = query_property_match(dict(row))
    entity = query_entity_coverage(dict(row))
    mix = fact_rule_mix_symbolic(dict(row))
    schemas = schema_count(dict(row))
    return {
        "proof_entailment_bonus": 0.0,  # filled from the persisted total residual
        "query_token_coverage_bonus": 0.030 * query_coverage,
        "fact_rule_mix_bonus": 0.020 * mix,
        "compact_query_property_bonus": 0.350 * 0.025 * prop,
        "compact_query_entity_bonus": 0.350 * 0.015 * entity,
        "compact_fact_rule_mix_bonus": 0.350 * 0.020 * mix,
        "compact_size_penalty": 0.350 * -0.006 * oversize,
        "compact_extra_schema_penalty": 0.350 * -0.004 * max(0, schemas - 1),
        "extra_unit_penalty": -0.010 * oversize,
    }


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean_absolute": statistics.fmean(abs(value) for value in values) if values else None,
        "positive_fraction": sum(value > 0 for value in values) / len(values) if values else None,
        "negative_fraction": sum(value < 0 for value in values) / len(values) if values else None,
        "sum": sum(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, default=ROOT / "outputs/development_runs/production_generator_d_v1_k_sensitivity/per_example_rankings.jsonl")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/production_generator_d_v1")
    parser.add_argument("--existing-symbolic-diagnostic", type=Path, default=ROOT / "outputs/diagnostics/production_generator_d_v1_causal_retrieval_diagnosis/symbolic_reranking_analysis.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/diagnostics/production_generator_d_v1_symbolic_mechanism_dev")
    args = parser.parse_args()

    persisted_totals = {}
    existing = json.loads(args.existing_symbolic_diagnostic.read_text(encoding="utf-8"))
    for example in existing.get("examples", []):
        for name in ("gnn_top1", "final_top1"):
            candidate = example.get(name, {})
            total = candidate.get("symbolic_terms", {}).get("proof_adjustment_total")
            if total is not None:
                persisted_totals[(example["dataset"], example["example_id"], key(candidate.get("subgraph_units", [])))] = float(total)

    records = defaultdict(dict)
    with args.rankings.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "dev":
                raise ValueError("Only DEV rankings are allowed")
            records[(row["dataset"], row["example_id"])][row["method"]] = row

    candidate_rows = {}
    accessed = []
    for dataset, _, _ in DATASETS:
        path = args.data_root / dataset / "dev_subgraph_retrieval.jsonl"
        if "test" in path.name.lower() or not path.is_file():
            raise FileNotFoundError(path)
        accessed.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
        needed = {example_id for ds, example_id in records if ds == dataset}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                example_id = row["example_id"]
                if example_id in needed:
                    candidate_rows[(dataset, example_id, key(row.get("subgraph_units", [])))] = row

    changed_candidates = []
    flips = []
    for (dataset, example_id), methods in records.items():
        gnn = methods["gnn_only"]
        final = methods["sageqa_final"]
        gnn_rows = {key(row["subgraph_units"]): row for row in gnn["ranked_candidates"]}
        final_rows = {key(row["subgraph_units"]): row for row in final["ranked_candidates"]}
        gnn_order = list(gnn_rows)
        final_order = list(final_rows)
        domain = gnn["domain"]
        hop = "1hop" if "1hop" in dataset else "2hop"

        for candidate_key in sorted(set(gnn_rows) | set(final_rows)):
            g_rank = gnn_order.index(candidate_key) + 1 if candidate_key in gnn_rows else None
            f_rank = final_order.index(candidate_key) + 1 if candidate_key in final_rows else None
            if g_rank == f_rank:
                continue
            raw = candidate_rows[(dataset, example_id, candidate_key)]
            persisted = final_rows.get(candidate_key)
            terms = text_terms(raw) if domain == "text" else ontology_nonproof_terms(raw)
            stored_total = persisted_totals.get((dataset, example_id, candidate_key))
            total_available = persisted is not None or stored_total is not None
            if domain == "ontology" and total_available:
                total = (float(persisted["adjusted_score"]) - float(persisted["score"])) if persisted is not None else stored_total
                residual = total - sum(terms.values())
                terms["proof_entailment_bonus"] = residual
            changed_candidates.append({
                "dataset": dataset, "domain": domain, "hop": hop, "example_id": example_id,
                "subgraph_units": list(candidate_key), "gnn_top5_rank": g_rank,
                "symbolic_top5_rank": f_rank, "complete": complete(raw),
                "neural_score": float((persisted or gnn_rows[candidate_key])["score"]),
                "term_contributions": terms if domain == "text" or total_available else None,
                "ontology_total_unavailable_outside_persisted_symbolic_top5": domain == "ontology" and not total_available,
            })

        g_key, f_key = gnn_order[0], final_order[0]
        if g_key == f_key:
            continue
        g_raw = candidate_rows[(dataset, example_id, g_key)]
        f_raw = candidate_rows[(dataset, example_id, f_key)]
        g_complete, f_complete = complete(g_raw), complete(f_raw)
        effect = "fixed" if not g_complete and f_complete else "harmed" if g_complete and not f_complete else "completeness_unaffected"
        g_persisted, f_persisted = final_rows.get(g_key), final_rows[f_key]
        g_terms = text_terms(g_raw) if domain == "text" else ontology_nonproof_terms(g_raw)
        f_terms = text_terms(f_raw) if domain == "text" else ontology_nonproof_terms(f_raw)
        g_stored_total = persisted_totals.get((dataset, example_id, g_key))
        decomposable = domain == "text" or g_persisted is not None or g_stored_total is not None
        if domain == "ontology":
            f_total = float(f_persisted["adjusted_score"]) - float(f_persisted["score"])
            f_terms["proof_entailment_bonus"] = f_total - sum(f_terms.values())
            if g_persisted is not None or g_stored_total is not None:
                g_total = (float(g_persisted["adjusted_score"]) - float(g_persisted["score"])) if g_persisted is not None else g_stored_total
                g_terms["proof_entailment_bonus"] = g_total - sum(g_terms.values())
            else:
                g_terms["proof_entailment_bonus"] = None
        flips.append({
            "dataset": dataset, "domain": domain, "hop": hop, "example_id": example_id,
            "effect": effect, "gnn_top1_complete": g_complete, "symbolic_top1_complete": f_complete,
            "gnn_top1_units": list(g_key), "symbolic_top1_units": list(f_key),
            "neural_score_gap_symbolic_winner_minus_gnn_winner": float(f_persisted["score"]) - float(gnn_rows[g_key]["score"]),
            "term_difference_symbolic_winner_minus_gnn_winner": {
                name: (f_terms[name] - g_terms[name] if g_terms[name] is not None else None)
                for name in f_terms
            },
            "fully_decomposable_from_persisted_scores": decomposable,
        })

    groups = {}
    for effect in ("fixed", "harmed", "completeness_unaffected"):
        groups[effect] = {}
        subset = [row for row in flips if row["effect"] == effect]
        term_names = sorted({name for row in subset for name in row["term_difference_symbolic_winner_minus_gnn_winner"]})
        for term in term_names:
            groups[effect][term] = {}
            for slice_name, predicate in {
                "all": lambda row: True,
                "text": lambda row: row["domain"] == "text",
                "ontology": lambda row: row["domain"] == "ontology",
                "1hop": lambda row: row["hop"] == "1hop",
                "2hop": lambda row: row["hop"] == "2hop",
            }.items():
                values = [row["term_difference_symbolic_winner_minus_gnn_winner"][term] for row in subset if predicate(row) and term in row["term_difference_symbolic_winner_minus_gnn_winner"] and row["term_difference_symbolic_winner_minus_gnn_winner"][term] is not None]
                groups[effect][term][slice_name] = summarize(values)

    counts = defaultdict(int)
    for row in flips:
        counts[row["effect"]] += 1
    analysis = {
        "schema_version": "production_generator_d_v1_symbolic_component_analysis_v1",
        "split": "dev", "test_data_accessed": False, "training_run": False,
        "neural_inference_run": False, "candidate_generation_run": False, "answer_generation_run": False,
        "rank_change_scope": "union of persisted GNN top-5 and SAGE-QA top-5 candidates; ranks outside a persisted top-5 are recorded as null",
        "top1_flip_counts": dict(counts), "component_summaries": groups,
        "changed_candidates": changed_candidates,
        "lineage": {"rankings": str(args.rankings.relative_to(ROOT)), "rankings_sha256": sha256(args.rankings), "dev_candidate_files": accessed},
    }
    fixed_harmed = {"split": "dev", "counts": dict(counts), "top1_flips": flips}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "symbolic_component_analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "fixed_vs_harmed.json").write_text(json.dumps(fixed_harmed, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"counts": dict(counts), "changed_candidates": len(changed_candidates), "fully_decomposable_flips": sum(row["fully_decomposable_from_persisted_scores"] for row in flips)}, indent=2))


if __name__ == "__main__":
    main()
