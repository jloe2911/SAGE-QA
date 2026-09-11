"""Post-hoc semantic-sufficiency diagnostic for frozen ontology DEV candidates.

This script replays candidate construction only to recover candidate contents that
were not persisted by the protected-query-anchor comparison.  It refuses to use
the replay unless every candidate digest matches that frozen comparison.  Gold
answers and reference explanations are read only after both A and C are frozen.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_subgraph_training_data import (  # noqa: E402
    generate_ontology_candidates,
    get_gold_explanations,
    local_name,
    parse_owl_context,
)
from data_processing.evidence_graph_candidates import progressive_connected_supports  # noqa: E402
from data_processing.retrieval_contracts import git_provenance, sha256_file, write_json  # noqa: E402
from evaluation.compare_unified_evidence_graph_candidates import (  # noqa: E402
    DISPLAY_NAMES,
    PROTECTED_CONFIG,
    _canonical,
    _digest,
    build_ontology_evidence_graph,
)
from evaluation.validate_gold_free_candidate_pools import (  # noqa: E402
    ONTO,
    ONTO_BASE,
    ROOT,
    onto_rows,
)

FROZEN_ROOT = ROOT / "outputs/development_runs/protected_query_anchor_comparison_v1"
DEFAULT_OUTPUT = ROOT / "outputs/development_runs/ontology_semantic_sufficiency_v1"
MAX_PROOF_AXIOMS = 24
MAX_PROOFS_PER_FACT = 20_000


@dataclass(frozen=True)
class Rule:
    kind: str
    args: tuple[str, ...]
    unit: str


def _query(query: str) -> tuple[str, tuple[str, str, str]]:
    kind = "SELECT" if query.lstrip().upper().startswith("SELECT") else "ASK"
    match = re.search(
        r"\{\s*(<[^>]+>|\?\w+)\s+(<[^>]+>|\?\w+)\s+(<[^>]+>|\?\w+)\s*\}",
        query,
        flags=re.I,
    )
    if not match:
        raise ValueError(f"Unsupported formal query: {query}")

    def term(value: str) -> str:
        if value.startswith("?"):
            return value
        name = local_name(value)
        return "rdf:type" if name == "type" and "rdf-syntax-ns" in value else name

    return kind, tuple(term(value) for value in match.groups())  # type: ignore[return-value]


def _parse_unit(unit: str) -> tuple[tuple[str, str, str] | None, Rule | None]:
    patterns = (
        ("inverse", r"InverseObjectProperties\(([^,]+),([^\)]+)\)"),
        ("subproperty", r"SubObjectPropertyOf\(([^,]+),([^\)]+)\)"),
        ("equivalent_property", r"EquivalentObjectProperties\(([^,]+),([^\)]+)\)"),
        ("symmetric", r"SymmetricObjectProperty\(([^\)]+)\)"),
        ("transitive", r"TransitiveObjectProperty\(([^\)]+)\)"),
    )
    for kind, pattern in patterns:
        if match := re.fullmatch(pattern, unit, flags=re.I):
            return None, Rule(kind, tuple(part.strip() for part in match.groups()), unit)
    if match := re.fullmatch(r"ObjectPropertyChain\((.+)->([^\)]+)\)", unit, flags=re.I):
        chain = tuple(part.strip() for part in match.group(1).split(","))
        return None, Rule("chain", (*chain, match.group(2).strip()), unit)
    for token, kind in (
        (" SubClassOf ", "subclass"),
        (" domain ", "domain"),
        (" range ", "range"),
    ):
        if token.lower() in unit.lower():
            left, right = re.split(re.escape(token), unit, maxsplit=1, flags=re.I)
            return None, Rule(kind, (left.strip(), right.strip()), unit)
    parts = unit.split()
    if len(parts) >= 3:
        relation = parts[1]
        relation = "rdf:type" if relation.lower() in {"rdf:type", "type"} else relation
        return (parts[0], relation, " ".join(parts[2:])), None
    return None, None


def _add_proof(
    proofs: dict[tuple[str, str, str], list[frozenset[str]]],
    fact: tuple[str, str, str],
    proof: frozenset[str],
) -> bool:
    if len(proof) > MAX_PROOF_AXIOMS:
        return False
    existing = proofs.setdefault(fact, [])
    if any(old <= proof for old in existing):
        return False
    existing[:] = [old for old in existing if not proof < old]
    existing.append(proof)
    if len(existing) > MAX_PROOFS_PER_FACT:
        raise RuntimeError(f"Proof antichain overflow for {fact}: {len(existing)}")
    return True


def _joins(
    left: Sequence[frozenset[str]], right: Sequence[frozenset[str]], rule: str
) -> Iterable[frozenset[str]]:
    for a in left:
        for b in right:
            yield a | b | {rule}


def derive(units: Iterable[str]) -> dict[tuple[str, str, str], list[frozenset[str]]]:
    proofs: dict[tuple[str, str, str], list[frozenset[str]]] = {}
    rules: list[Rule] = []
    for unit in sorted(set(units)):
        fact, rule = _parse_unit(unit)
        if fact:
            _add_proof(proofs, fact, frozenset({unit}))
        elif rule:
            rules.append(rule)

    changed = True
    while changed:
        changed = False
        snapshot = list(proofs.items())
        by_relation: dict[str, list[tuple[tuple[str, str, str], list[frozenset[str]]]]] = (
            collections.defaultdict(list)
        )
        for fact, supports in snapshot:
            by_relation[fact[1]].append((fact, supports))
        for rule in rules:
            kind, args = rule.kind, rule.args
            emissions: list[tuple[tuple[str, str, str], frozenset[str]]] = []
            if kind in {"subclass", "subproperty", "domain", "range"}:
                source, target = args
                relation = "rdf:type" if kind == "subclass" else source
                for (s, _, o), supports in by_relation.get(relation, []):
                    if kind == "subclass" and o != source:
                        continue
                    out = (
                        (s, "rdf:type", target)
                        if kind in {"subclass", "domain"}
                        else (o, "rdf:type", target)
                        if kind == "range"
                        else (s, target, o)
                    )
                    emissions.extend((out, proof | {rule.unit}) for proof in supports)
            elif kind in {"inverse", "equivalent_property"}:
                left, right = args
                for relation, target in ((left, right), (right, left)):
                    for (s, _, o), supports in by_relation.get(relation, []):
                        out = (o, target, s) if kind == "inverse" else (s, target, o)
                        emissions.extend((out, proof | {rule.unit}) for proof in supports)
            elif kind == "symmetric":
                for (s, p, o), supports in by_relation.get(args[0], []):
                    emissions.extend(((o, p, s), proof | {rule.unit}) for proof in supports)
            elif kind == "transitive":
                relation = args[0]
                facts = by_relation.get(relation, [])
                for (s, _, mid), left in facts:
                    for (mid2, _, o), right in facts:
                        if mid == mid2:
                            emissions.extend(
                                ((s, relation, o), proof)
                                for proof in _joins(left, right, rule.unit)
                            )
            elif kind == "chain":
                chain, target = args[:-1], args[-1]
                states = [
                    ((s, o), supports) for (s, _, o), supports in by_relation.get(chain[0], [])
                ]
                for relation in chain[1:]:
                    next_states = []
                    for (start, mid), left in states:
                        for (mid2, _, end), right in by_relation.get(relation, []):
                            if mid == mid2:
                                next_states.append(
                                    ((start, end), list(_joins(left, right, rule.unit)))
                                )
                    states = next_states
                for (s, o), supports in states:
                    emissions.extend(((s, target, o), proof | {rule.unit}) for proof in supports)
            for fact, proof in emissions:
                changed |= _add_proof(proofs, fact, proof)
    return proofs


def _answer_proofs(
    units: Iterable[str], formal_query: str, required_answers: set[str]
) -> tuple[str, list[frozenset[str]]]:
    kind, (subject, relation, obj) = _query(formal_query)
    facts = derive(units)
    if kind == "ASK":
        return kind, list(facts.get((subject, relation, obj), []))
    output: list[frozenset[str]] = []
    for (s, p, o), proofs in facts.items():
        if s == subject and p == relation and (obj.startswith("?") or obj == o):
            if not required_answers or o in required_answers:
                output.extend(proofs)
    return kind, output


def _coverable(proof: frozenset[str], availability: Mapping[str, int], k: int) -> bool:
    remaining = set(proof)

    def visit(left: set[str], chosen: int, depth: int) -> bool:
        if not left:
            return True
        if depth == k:
            return False
        unit = min(left, key=lambda value: (availability.get(value, 0).bit_count(), value))
        choices = availability.get(unit, 0) & ~chosen
        while choices:
            bit = choices & -choices
            choices -= bit
            covered = {value for value in left if availability.get(value, 0) & bit}
            if visit(left - covered, chosen | bit, depth + 1):
                return True
        return False

    return visit(remaining, 0, 0)


def semantic_metrics(
    candidates: Sequence[Sequence[str]], formal_query: str, answer: Any
) -> dict[str, Any]:
    availability: dict[str, int] = collections.defaultdict(int)
    for index, candidate in enumerate(candidates):
        for unit in candidate:
            availability[unit] |= 1 << index
    required = {value.strip() for value in str(answer or "").split(";") if value.strip()}
    kind, proofs = _answer_proofs(availability, formal_query, required)
    return {
        "query_kind": kind,
        "semantic_sufficiency_at_1": any(_coverable(proof, availability, 1) for proof in proofs),
        "semantic_sufficiency_at_2": any(_coverable(proof, availability, 2) for proof in proofs),
        "semantic_sufficiency_at_3": any(_coverable(proof, availability, 3) for proof in proofs),
        "semantic_sufficiency_all": bool(proofs),
        "semantic_proof_count": len(proofs),
    }


def reference_exceptions(
    explanations: Sequence[Sequence[str]], formal_query: str, answer: Any
) -> list[int]:
    failures = []
    for index, explanation in enumerate(explanations):
        if not semantic_metrics([explanation], formal_query, answer)["semantic_sufficiency_at_1"]:
            failures.append(index)
    return failures


def _rates(
    rows: Sequence[Mapping[str, Any]], method: str, kind: str | None = None
) -> dict[str, Any]:
    selected = [row for row in rows if kind is None or row["query_kind"] == kind]
    n = len(selected)
    result: dict[str, Any] = {"examples": n}
    for suffix in ("at_1", "at_2", "at_3", "all"):
        ref_key = f"{method}_reference_coverage_{suffix}"
        sem_key = f"{method}_semantic_sufficiency_{suffix}"
        ref_count = sum(bool(row[ref_key]) for row in selected)
        sem_count = sum(bool(row[sem_key]) for row in selected)
        rescued = sum(not row[ref_key] and row[sem_key] for row in selected)
        misses = n - ref_count
        result[suffix] = {
            "reference_count": ref_count,
            "reference_pct": 100 * ref_count / max(n, 1),
            "semantic_count": sem_count,
            "semantic_pct": 100 * sem_count / max(n, 1),
            "difference_percentage_points": 100 * (sem_count - ref_count) / max(n, 1),
            "reference_misses": misses,
            "reference_misses_rescued": rescued,
            "reference_misses_rescued_pct": 100 * rescued / max(misses, 1),
        }
    return result


def _read_frozen(name: str) -> dict[str, dict[str, Any]]:
    path = FROZEN_ROOT / name / "details.jsonl"
    return {
        row["example_id"]: row
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    }


def run(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic: {output_dir}")
    output_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": "ontology_semantic_sufficiency_v1",
        "development_only": True,
        "methods": {
            "A": "current clean candidates",
            "C": "unified + protected query-anchor candidates",
        },
        "semantic_policy": "ASK entails the ground triple; SELECT yields at least one required answer, matching per-answer stored references",
        "top_k_policy": "existence of a deduplicated union of at most k generated candidate supports, matching the existing reference-coverability metric",
        "candidate_recovery_policy": "deterministic replay accepted only after exact per-example digest equality to the frozen protected comparison",
        "test_data_used": False,
        "training_run": False,
        "adaptive_k_run": False,
        "answer_generation_run": False,
        "git": git_provenance(ROOT),
        "frozen_comparison_sha256": sha256_file(FROZEN_ROOT / "comparison.json"),
        "datasets": {},
    }
    all_rows = []
    total_reference_exceptions = 0
    for name in ONTO:
        frozen = _read_frozen(name)
        frozen_rows: list[
            tuple[
                str,
                list[tuple[str, ...]],
                list[tuple[str, ...]],
                Mapping[str, Any],
                Mapping[str, Any],
            ]
        ] = []
        for group_index, qa_index, item, qa in onto_rows(name):
            question = str(
                qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or ""
            )
            sparql = str(qa.get("SPARQL Query") or "")
            owl_context = str(item.get("OWL Context") or "")
            current_generated = generate_ontology_candidates(
                question=question,
                sparql_query=sparql,
                owl_context=owl_context,
                max_subgraph_size=ONTO_BASE["max_size"],
                min_subgraph_size=1,
                max_context_units=ONTO_BASE["atomic_budget"],
                candidate_beam_width=ONTO_BASE["beam_width"],
                max_candidate_subgraphs=ONTO_BASE["candidate_cap"],
            )
            current = _canonical(current_generated["candidate_subgraphs"])
            graph = build_ontology_evidence_graph(parse_owl_context(owl_context), question, sparql)
            protected = _canonical(
                progressive_connected_supports(graph, PROTECTED_CONFIG).candidates
            )
            example_id = f"{name}__g{group_index}__q{qa_index}"
            lock = frozen.get(example_id)
            if lock is None:
                raise RuntimeError(f"Frozen example missing: {example_id}")
            if _digest(current) != lock["current_candidate_digest"]:
                raise RuntimeError(f"A candidate digest mismatch: {example_id}")
            if _digest(protected) != lock["protected_candidate_digest"]:
                raise RuntimeError(f"C candidate digest mismatch: {example_id}")
            frozen_rows.append((example_id, current, protected, qa, lock))

        # Gold/reference boundary: all candidate contents and frozen digests above.
        rows = []
        for example_id, current, protected, qa, lock in frozen_rows:
            sparql = str(qa.get("SPARQL Query") or "")
            answer = qa.get("Answer")
            gold = get_gold_explanations(dict(qa))
            row: dict[str, Any] = {"dataset": DISPLAY_NAMES[name], "example_id": example_id}
            for method, candidates, prefix in (
                ("A", current, "current"),
                ("C", protected, "protected"),
            ):
                sem = semantic_metrics(candidates, sparql, answer)
                row["query_kind"] = sem.pop("query_kind")
                for suffix in ("at_1", "at_2", "at_3"):
                    row[f"{method}_reference_coverage_{suffix}"] = bool(
                        lock[f"{prefix}_oracle_support_coverage_{suffix}"]
                    )
                row[f"{method}_reference_coverage_all"] = bool(
                    lock[f"{prefix}_all_candidate_union_coverage"]
                )
                row.update({f"{method}_{key}": value for key, value in sem.items()})
            exceptions = reference_exceptions(gold, sparql, answer)
            row["reference_explanation_count"] = len(gold)
            row["insufficient_reference_explanation_indices"] = exceptions
            total_reference_exceptions += len(exceptions)
            rows.append(row)
            all_rows.append(row)
        detail_path = output_dir / name / "details.jsonl"
        detail_path.parent.mkdir(parents=True)
        with detail_path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        report["datasets"][DISPLAY_NAMES[name]] = {
            "source_file": str(ONTO[name].relative_to(ROOT)),
            "source_sha256": sha256_file(ONTO[name]),
            "candidate_digest_mismatches": 0,
            "reference_explanations": sum(row["reference_explanation_count"] for row in rows),
            "insufficient_reference_explanations": sum(
                len(row["insufficient_reference_explanation_indices"]) for row in rows
            ),
            "A": {
                "all": _rates(rows, "A"),
                "ASK": _rates(rows, "A", "ASK"),
                "SELECT": _rates(rows, "A", "SELECT"),
            },
            "C": {
                "all": _rates(rows, "C"),
                "ASK": _rates(rows, "C", "ASK"),
                "SELECT": _rates(rows, "C", "SELECT"),
            },
        }
    report["overall"] = {
        "examples": len(all_rows),
        "reference_explanations": sum(row["reference_explanation_count"] for row in all_rows),
        "insufficient_reference_explanations": total_reference_exceptions,
        "A": {
            "all": _rates(all_rows, "A"),
            "ASK": _rates(all_rows, "A", "ASK"),
            "SELECT": _rates(all_rows, "A", "SELECT"),
        },
        "C": {
            "all": _rates(all_rows, "C"),
            "ASK": _rates(all_rows, "C", "ASK"),
            "SELECT": _rates(all_rows, "C", "SELECT"),
        },
    }
    write_json(output_dir / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
