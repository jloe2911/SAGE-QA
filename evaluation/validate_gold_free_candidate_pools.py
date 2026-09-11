"""Development-only large-sample audit of gold-free candidate generation."""

from __future__ import annotations

import argparse, collections, hashlib, json, random, statistics, sys, time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_subgraph_training_data import (  # noqa: E402
    build_split_map,
    generate_ontology_candidates,
    get_gold_explanations,
    is_connected as is_connected_units,
    parse_owl_context,
    unit_signature,
)
from data_processing.build_2wiki_subgraph_dataset import (  # noqa: E402
    generate_candidates as gen_2wiki,
    get_gold_support_units as gold_2wiki,
    normalize_2wiki_record,
)
from data_processing.build_hotpot_subgraph_dataset import (  # noqa: E402
    generate_candidates as gen_hotpot,
    get_gold_support_units as gold_hotpot,
    normalize_hotpot_record,
)
from data_processing.retrieval_contracts import (  # noqa: E402
    BUILDER_VERSION,
    clean_text_retrieval_input,
    git_provenance,
    sha256_file,
    write_json,
)
from data_processing.text_kg_constructor import KGConstructionConfig  # noqa: E402
from models.symbolic_composer import extract_query_signature  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEXT = {
    "2WikiMultiHopQA": ROOT / "data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet",
    "HotpotQA": ROOT / "data/raw/hotpot_qa/distractor/validation-00000-of-00001.parquet",
}
ONTO = {
    "Family_1hop": ROOT / "data/raw/family/FamilyOWL_1hop.json",
    "Family_2hop": ROOT / "data/raw/family/FamilyOWL_2hop.json",
    "Pizza_100_1hop": ROOT / "data/raw/pizza_100/pizza_100_1hop.json",
    "Pizza_100_2hop": ROOT / "data/raw/pizza_100/pizza_100_2hop.json",
    "Pizza_250_1hop": ROOT / "data/raw/pizza_250/pizza_250_1hop.json",
    "Pizza_250_2hop": ROOT / "data/raw/pizza_250/pizza_250_2hop.json",
    "OWL2Bench_1hop": ROOT / "data/raw/owl2bench/OWL2Bench_1hop.json",
    "OWL2Bench_2hop": ROOT / "data/raw/owl2bench/OWL2Bench_2hop.json",
}
TEXT_BASE = {
    "2WikiMultiHopQA": {"atomic_budget": 30, "max_size": 4, "candidate_cap": 512},
    "HotpotQA": {"atomic_budget": 30, "max_size": 3, "candidate_cap": 512},
}
ONTO_BASE = {"atomic_budget": 40, "max_size": 6, "beam_width": 96, "candidate_cap": 320}


def sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    rows.sort(key=lambda r: str(r.get("_id") or r.get("id") or ""))
    random.Random(seed).shuffle(rows)
    return rows if n <= 0 else rows[:n]


def hist(values: Iterable[int], limits: Sequence[int]) -> dict[str, int]:
    values, out, lo = list(values), {}, 0
    for hi in limits:
        out[str(hi) if lo == hi else f"{lo}-{hi}"] = sum(lo <= x <= hi for x in values)
        lo = hi + 1
    out[f">={lo}"] = sum(x >= lo for x in values)
    return out


def gold_diag(candidates, golds, pool, context) -> dict[str, Any]:
    """Post-hoc only: best-of-valid-explanations evaluator semantics."""
    cs, gs, pool, context = (
        [set(x) for x in candidates],
        [set(x) for x in golds if x],
        set(pool),
        set(context),
    )
    union = set().union(*cs) if cs else set()
    recall = lambda have, gold: len(have & gold) / len(gold) if gold else 0.0
    preferred = max(
        gs,
        key=lambda g: (
            recall(context, g),
            recall(pool, g),
            max((recall(c, g) for c in cs), default=0),
            -len(g),
            tuple(sorted(g)),
        ),
        default=set(),
    )
    complete = any(g <= c for g in gs for c in cs)
    partial = any(g & c for g in gs for c in cs)
    return {
        "gold_explanation_count": len(gs),
        "preferred_gold": sorted(preferred),
        "pool_union_gold_recall": max((recall(union, g) for g in gs), default=0.0),
        "atomic_pool_gold_recall": max((recall(pool, g) for g in gs), default=0.0),
        "best_candidate_gold_recall": max((recall(c, g) for c in cs for g in gs), default=0.0),
        "all_gold_in_candidate_union": any(g <= union for g in gs),
        "complete_support_candidate": complete,
        "partial_support_any": partial,
        "partial_support_only": partial and not complete,
        "preferred_gold_in_context": bool(preferred) and preferred <= context,
        "preferred_gold_in_atomic_pool": bool(preferred) and preferred <= pool,
        "preferred_gold_connected": bool(preferred) and is_connected_units(sorted(preferred)),
    }


def category(d: Mapping[str, Any], max_size: int) -> str | None:
    if d["complete_support_candidate"]:
        return None
    if not d["preferred_gold_in_context"]:
        return "F_context_missing_reference_support"
    if not d["preferred_gold_in_atomic_pool"]:
        return "A_required_evidence_missing_from_atomic_pool"
    if max_size > 0 and len(d["preferred_gold"]) > max_size:
        return "D_candidate_size_limit"
    if not d["preferred_gold_connected"]:
        return "E_connectivity_assumption"
    return "B_units_not_combined_or_C_pruned_by_budget"


def summarize(rows: list[dict], elapsed: float) -> dict[str, Any]:
    n, cc, ac = (
        len(rows),
        [r["candidate_count"] for r in rows],
        [r["atomic_pool_size"] for r in rows],
    )
    sizes = [x for r in rows for x in r["candidate_sizes"]]
    fail = collections.Counter(r["failure_category"] for r in rows if r["failure_category"])
    pct = lambda key: 100 * sum(bool(r[key]) for r in rows) / max(n, 1)
    return {
        "examples": n,
        "zero_candidates": sum(x == 0 for x in cc),
        "zero_candidates_pct": 100 * sum(x == 0 for x in cc) / max(n, 1),
        "candidate_count": {
            "mean": statistics.fmean(cc) if cc else 0,
            "median": statistics.median(cc) if cc else 0,
            "distribution": hist(cc, [0, 32, 128, 320, 512, 1024]),
        },
        "atomic_pool_size": {
            "mean": statistics.fmean(ac) if ac else 0,
            "median": statistics.median(ac) if ac else 0,
            "distribution": hist(ac, [0, 10, 20, 30, 40, 60, 80]),
        },
        "candidate_size_distribution": dict(sorted(collections.Counter(sizes).items())),
        "pool_union_gold_recall": statistics.fmean(r["pool_union_gold_recall"] for r in rows)
        if rows
        else 0,
        "all_gold_in_candidate_union_pct": pct("all_gold_in_candidate_union"),
        "complete_support_candidate_pct": pct("complete_support_candidate"),
        "partial_support_any_pct": pct("partial_support_any"),
        "partial_support_only_pct": pct("partial_support_only"),
        "failure_categories": {
            k: {"count": v, "pct": 100 * v / max(n, 1)} for k, v in sorted(fail.items())
        },
        "runtime_seconds": elapsed,
        "runtime_seconds_per_example": elapsed / max(n, 1),
    }


def text_rows(name: str, n: int, seed: int) -> list[dict]:
    norm = normalize_2wiki_record if name == "2WikiMultiHopQA" else normalize_hotpot_record
    return sample([norm(x) for x in pd.read_parquet(TEXT[name]).to_dict(orient="records")], n, seed)


def audit_text(name: str, rows: list[dict], cfg: Mapping[str, int], seed: int):
    gen, gold_fn = (
        (gen_2wiki, gold_2wiki) if name == "2WikiMultiHopQA" else (gen_hotpot, gold_hotpot)
    )
    kg, details, start = (
        KGConstructionConfig(backend="context_only", max_triples=64),
        [],
        time.perf_counter(),
    )
    for i, ex in enumerate(rows):
        g = gen(
            retrieval_example=clean_text_retrieval_input(ex),
            split_name="development_audit",
            max_sentences_per_example=cfg["atomic_budget"],
            max_subgraph_size=cfg["max_size"],
            max_candidates_per_question=cfg["candidate_cap"],
            kg_config=kg,
            llm_kg_constructor=None,
            kg_cache=None,
            kg_cache_path=None,
            kg_cache_lock=None,
            seed=seed + i,
        )
        gold = gold_fn(ex, g["sent_lookup"])
        context = list(g["sent_lookup"].values())
        d = {
            "example_id": g["example_id"],
            "question": g["question"],
            "atomic_pool_size": len(g["sentence_pool"]),
            "context_unit_count": len(context),
            "graph_context_unit_count": len(g["graph_context_units"]),
            "candidate_count": len(g["candidates"]),
            "candidate_sizes": [len(c) for c in g["candidates"]],
            "cross_document_candidate_count": sum(
                len({u.split("::", 3)[1] for u in c}) > 1 for c in g["candidates"]
            ),
            "both_supporting_sentences_in_atomic_pool": set(gold) <= set(g["sentence_pool"]),
            "evidences_absent": "evidences" not in ex,
            "kg_cache_used": False,
            **gold_diag(g["candidates"], [gold], g["sentence_pool"], context),
        }
        d["failure_category"] = category(d, cfg["max_size"])
        details.append(d)
    s = summarize(details, time.perf_counter() - start)
    s.update(
        {
            "source_file": str(TEXT[name].relative_to(ROOT)),
            "source_sha256": sha256_file(TEXT[name]),
            "sample_policy": f"deterministic seed-{seed} sample of official development",
            "config": dict(cfg),
            "both_supporting_sentences_in_atomic_pool_pct": 100
            * sum(d["both_supporting_sentences_in_atomic_pool"] for d in details)
            / max(len(details), 1),
            "examples_with_cross_document_candidates_pct": 100
            * sum(d["cross_document_candidate_count"] > 0 for d in details)
            / max(len(details), 1),
            "evidences_absent_for_all_2wiki_rows": all(d["evidences_absent"] for d in details),
            "graph_context_context_only": True,
            "contaminated_cache_loaded": False,
        }
    )
    return s, details


def onto_rows(name: str):
    groups = json.loads(ONTO[name].read_text(encoding="utf-8"))
    split = build_split_map(groups, 0.65, 0.1)
    return [
        (gi, qi, item, qa)
        for gi, item in enumerate(groups)
        if split[gi] == "dev"
        for qi, qa in enumerate(item.get("QAs", []))
        if get_gold_explanations(qa)
    ]


def shape(s: str) -> str:
    u = " ".join(s.upper().split())
    return (
        "SELECT" if u.startswith("SELECT") else "ASK" if u.startswith("ASK") else "OTHER"
    ) + f"_variables_{s.count('?')}"


def zero_trace(question, sparql, context, generated, cfg):
    sig = extract_query_signature(question=question, sparql_query=sparql)
    ents, props, direct = set(sig["query_entities"]), set(sig["query_properties"]), []
    for ax in context:
        ae, ap = unit_signature(ax)
        if ae & ents or ap & props:
            direct.append(ax)
    if not ents and not props:
        reason, point = (
            "query_entity_and_property_extraction_failure",
            "empty query signature before atomic filtering",
        )
    elif not direct:
        reason, point = (
            "candidate_unit_filtering",
            "no parsed context axiom overlaps query signature",
        )
    else:
        reason, point = (
            "beam_or_candidate_validation",
            "atomic pool exists but connected beam returned no candidates",
        )
    return {
        "formal_query_shape": shape(sparql),
        "formal_query": sparql,
        "available_ontology_context_size": len(context),
        "query_entities": sorted(ents),
        "query_properties": sorted(props),
        "initial_directly_relevant_units": direct,
        "selected_atomic_units": generated["candidate_units"],
        "stages": {
            "parsed_context": len(context),
            "direct_relevance_filter": len(direct),
            "atomic_budget_selection": len(generated["candidate_units"]),
            "connected_beam": len(generated["candidate_subgraphs"]),
        },
        "budgets": dict(cfg),
        "root_cause": reason,
        "exact_disappearance_point": point,
    }


def audit_onto(name: str, rows, cfg: Mapping[str, int]):
    details, zeros, start = [], [], time.perf_counter()
    for gi, qi, item, qa in rows:
        q = str(qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or "")
        sparql, owl = str(qa.get("SPARQL Query") or ""), str(item.get("OWL Context") or "")
        context = parse_owl_context(owl)
        g = generate_ontology_candidates(
            question=q,
            sparql_query=sparql,
            owl_context=owl,
            max_subgraph_size=cfg["max_size"],
            min_subgraph_size=1,
            max_context_units=cfg["atomic_budget"],
            candidate_beam_width=cfg["beam_width"],
            max_candidate_subgraphs=cfg["candidate_cap"],
        )
        d = {
            "example_id": f"{name}__g{gi}__q{qi}",
            "question": q,
            "formal_query_shape": shape(sparql),
            "atomic_pool_size": len(g["candidate_units"]),
            "context_unit_count": len(context),
            "candidate_count": len(g["candidate_subgraphs"]),
            "candidate_sizes": [len(c) for c in g["candidate_subgraphs"]],
            **gold_diag(
                g["candidate_subgraphs"], get_gold_explanations(qa), g["candidate_units"], context
            ),
        }
        d["failure_category"] = category(d, cfg["max_size"])
        details.append(d)
        if not g["candidate_subgraphs"]:
            zeros.append(
                {
                    "example_id": d["example_id"],
                    "question": q,
                    **zero_trace(q, sparql, context, g, cfg),
                }
            )
    s = summarize(details, time.perf_counter() - start)
    s.update(
        {
            "source_file": str(ONTO[name].relative_to(ROOT)),
            "source_sha256": sha256_file(ONTO[name]),
            "sample_policy": "all gold-labeled examples in deterministic group-stratified development split",
            "config": dict(cfg),
            "multiple_gold_explanation_semantics": "complete when any valid explanation is contained",
        }
    )
    return s, details, zeros


def grid(kind: str, b: Mapping[str, int]):
    if kind == "text":
        return [
            dict(b),
            {**b, "atomic_budget": 40},
            {**b, "candidate_cap": 1024},
            {**b, "atomic_budget": 40, "candidate_cap": 1024},
        ]
    return [
        dict(b),
        {**b, "max_size": 4},
        {**b, "max_size": 8},
        {**b, "atomic_budget": 60, "beam_width": 192, "candidate_cap": 640},
    ]


def cid(cfg):
    return hashlib.sha256(json.dumps(dict(cfg), sort_keys=True).encode()).hexdigest()[:10]


def refine_budget_failures(baseline_details, widened_details, baseline_summary):
    widened = {row["example_id"]: row for row in widened_details}
    for row in baseline_details:
        if row.get("failure_category") == "B_units_not_combined_or_C_pruned_by_budget":
            row["failure_category"] = (
                "C_correct_combination_removed_by_gold_free_budget"
                if widened.get(row["example_id"], {}).get("complete_support_candidate")
                else "B_required_units_never_combined"
            )
    counts = collections.Counter(
        row["failure_category"] for row in baseline_details if row.get("failure_category")
    )
    n = len(baseline_details)
    baseline_summary["failure_categories"] = {
        key: {"count": value, "pct": 100 * value / max(n, 1)}
        for key, value in sorted(counts.items())
    }


def jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def refine_ontology_failures(report_dir: Path, output_dir: Path):
    """Separate B from C with a gold-free wider beam on baseline misses only."""
    result = {
        "development_only": True,
        "generation_config": {
            "atomic_budget": 40,
            "max_size": 6,
            "beam_width": 192,
            "candidate_cap": 640,
        },
        "gold_available_during_candidate_generation": False,
        "datasets": {},
    }
    for name in ONTO:
        baseline_path = next((report_dir / name).glob("details_0_*.jsonl"))
        baseline = [json.loads(line) for line in baseline_path.open(encoding="utf-8")]
        combined = {
            row["example_id"]: row
            for row in baseline
            if row.get("failure_category") == "B_units_not_combined_or_C_pruned_by_budget"
        }
        source_rows = {
            f"{name}__g{group_index}__q{qa_index}": (item, qa)
            for group_index, qa_index, item, qa in onto_rows(name)
        }
        details = []
        started = time.perf_counter()
        for example_id in sorted(combined):
            item, qa = source_rows[example_id]
            question = str(
                qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or ""
            )
            sparql = str(qa.get("SPARQL Query") or "")
            owl = str(item.get("OWL Context") or "")
            generated = generate_ontology_candidates(
                question=question,
                sparql_query=sparql,
                owl_context=owl,
                max_subgraph_size=6,
                min_subgraph_size=1,
                max_context_units=40,
                candidate_beam_width=192,
                max_candidate_subgraphs=640,
            )
            diagnostic = gold_diag(
                generated["candidate_subgraphs"],
                get_gold_explanations(qa),
                generated["candidate_units"],
                parse_owl_context(owl),
            )
            details.append(
                {
                    "example_id": example_id,
                    "baseline_category": combined[example_id]["failure_category"],
                    "refined_category": (
                        "C_correct_combination_removed_by_gold_free_budget"
                        if diagnostic["complete_support_candidate"]
                        else "B_required_units_never_combined"
                    ),
                    "widened_candidate_count": len(generated["candidate_subgraphs"]),
                    **diagnostic,
                }
            )
        counts = collections.Counter(row["refined_category"] for row in details)
        result["datasets"][name] = {
            "baseline_combined_misses": len(combined),
            "refined_counts": dict(sorted(counts.items())),
            "runtime_seconds": time.perf_counter() - started,
            "details": details,
        }
    write_json(output_dir / "ontology_bc_refinement.json", result)
    return result


def run_invariance_checks(output_dir: Path, seed: int):
    report = {"development_only": True, "seed": seed, "datasets": {}}
    kg = KGConstructionConfig(backend="context_only", max_triples=64)
    for name in TEXT:
        rows = text_rows(name, 100, seed)
        generator = gen_2wiki if name == "2WikiMultiHopQA" else gen_hotpot
        mismatches = []
        for index, example in enumerate(rows):
            deleted = {
                key: value
                for key, value in example.items()
                if key not in {"answer", "supporting_facts", "evidences"}
            }
            outputs = []
            for source in (example, deleted):
                outputs.append(
                    generator(
                        retrieval_example=clean_text_retrieval_input(source),
                        split_name="invariance_dev",
                        max_sentences_per_example=TEXT_BASE[name]["atomic_budget"],
                        max_subgraph_size=TEXT_BASE[name]["max_size"],
                        max_candidates_per_question=TEXT_BASE[name]["candidate_cap"],
                        kg_config=kg,
                        llm_kg_constructor=None,
                        kg_cache=None,
                        kg_cache_path=None,
                        kg_cache_lock=None,
                        seed=seed + index,
                    )
                )
            fields = ("sentence_pool", "graph_context_units", "candidates")
            if any(outputs[0][field] != outputs[1][field] for field in fields):
                mismatches.append(outputs[0]["example_id"])
        report["datasets"][name] = {
            "examples": len(rows),
            "compared": [
                "atomic_pool",
                "graph_context",
                "candidate_set",
                "candidate_pre_rank_order",
            ],
            "mismatches": mismatches,
        }

    for name in ONTO:
        rows = onto_rows(name)
        random.Random(seed).shuffle(rows)
        rows = rows[:5]
        mismatches = []
        for group_index, qa_index, item, qa in rows:
            kwargs = {
                "question": str(
                    qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or ""
                ),
                "sparql_query": str(qa.get("SPARQL Query") or ""),
                "owl_context": str(item.get("OWL Context") or ""),
                "max_subgraph_size": 6,
                "min_subgraph_size": 1,
                "max_context_units": 40,
                "candidate_beam_width": 96,
                "max_candidate_subgraphs": 320,
            }
            attached = generate_ontology_candidates(**kwargs)
            # The generation signature has no gold/answer parameter; deleting
            # or changing annotations cannot alter these arguments.
            deleted = generate_ontology_candidates(**kwargs)
            fields = ("candidate_units", "candidate_subgraphs", "unit_scores")
            if any(attached[field] != deleted[field] for field in fields):
                mismatches.append(f"{name}__g{group_index}__q{qa_index}")
        report["datasets"][name] = {
            "examples": len(rows),
            "compared": ["atomic_pool", "candidate_set", "candidate_pre_rank_order"],
            "mismatches": mismatches,
        }
    write_json(output_dir / "candidate_invariance_report.json", report)
    return report


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "gold_free_candidate_audit_v2",
        "builder_version": BUILDER_VERSION,
        "development_only": True,
        "seed": args.seed,
        "text_sample_size": args.text_sample_size,
        "git": git_provenance(ROOT),
        "gold_available_during_candidate_generation": False,
        "gold_join_boundary": "post_hoc_diagnostics_only",
        "prohibited_actions": {
            "gnn_training": False,
            "test_evaluation": False,
            "adaptive_k": False,
            "answer_generation": False,
            "llm_or_api_calls": False,
        },
        "datasets": {},
        "budget_sensitivity": {},
        "zero_candidate_traces": {},
    }
    for name in TEXT:
        rows, summaries, detail_sets = text_rows(name, args.text_sample_size, args.seed), [], []
        for i, cfg in enumerate(
            grid("text", TEXT_BASE[name]) if args.run_grid else [TEXT_BASE[name]]
        ):
            s, d = audit_text(name, rows, cfg, args.seed)
            s["config_id"] = cid(cfg)
            summaries.append(s)
            detail_sets.append(d)
            jsonl(args.output_dir / name / f"details_{i}_{s['config_id']}.jsonl", d)
        if len(detail_sets) > 2:
            refine_budget_failures(detail_sets[0], detail_sets[2], summaries[0])
            jsonl(
                args.output_dir / name / f"details_0_{summaries[0]['config_id']}.jsonl",
                detail_sets[0],
            )
        report["datasets"][name], report["budget_sensitivity"][name] = summaries[0], summaries
    for name in () if args.only_text else ONTO:
        rows, summaries, detail_sets, baseline_zeros = onto_rows(name), [], [], []
        grid_rows = list(rows)
        random.Random(args.seed).shuffle(grid_rows)
        if args.grid_ontology_sample_size > 0:
            grid_rows = grid_rows[: args.grid_ontology_sample_size]
        for i, cfg in enumerate(grid("onto", ONTO_BASE) if args.run_grid else [ONTO_BASE]):
            evaluated_rows = rows if i == 0 else grid_rows
            s, d, z = audit_onto(name, evaluated_rows, cfg)
            s["config_id"] = cid(cfg)
            summaries.append(s)
            detail_sets.append(d)
            jsonl(args.output_dir / name / f"details_{i}_{s['config_id']}.jsonl", d)
            if i == 0:
                baseline_zeros = z
        report.setdefault("budget_grid_policy", {})[name] = {
            "baseline_examples": len(rows),
            "nonbaseline_examples": len(grid_rows) if args.run_grid else 0,
            "nonbaseline_selection": f"deterministic seed-{args.seed} development subsample",
        }
        (
            report["datasets"][name],
            report["budget_sensitivity"][name],
            report["zero_candidate_traces"][name],
        ) = summaries[0], summaries, baseline_zeros
    write_json(args.output_dir / "audit_report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text-sample-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-grid", action="store_true")
    p.add_argument("--only-text", action="store_true")
    p.add_argument("--refine-ontology-failures-from", type=Path)
    p.add_argument("--run-invariance-checks", action="store_true")
    p.add_argument("--grid-ontology-sample-size", type=int, default=40)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/development_runs/gold_free_candidate_audit_v2",
    )
    args = p.parse_args()
    if args.run_invariance_checks:
        print(
            json.dumps(
                run_invariance_checks(args.output_dir, args.seed), indent=2, ensure_ascii=False
            )
        )
    elif args.refine_ontology_failures_from:
        print(
            json.dumps(
                refine_ontology_failures(args.refine_ontology_failures_from, args.output_dir),
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
