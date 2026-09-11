from __future__ import annotations

import copy

import pytest

from data.build_subgraph_training_data import (
    build_rows_for_qa,
    generate_ontology_candidates,
    normalize_explanation,
    parse_owl_context,
    relevant_context_axioms,
)
from data_processing.build_2wiki_subgraph_dataset import (
    build_rows_for_example as build_2wiki_rows,
    generate_candidates as generate_2wiki_candidates,
    normalize_2wiki_record,
)
from data_processing.build_hotpot_subgraph_dataset import (
    build_rows_for_example as build_hotpot_rows,
)
from data_processing.retrieval_contracts import (
    cap_inference_candidate_rows,
    select_disjoint_cohorts,
)
from data_processing.text_kg_constructor import KGConstructionConfig, construct_text_kg
from models.symbolic_composer import extract_query_signature


def _text_kwargs():
    return {
        "split_name": "dev",
        "max_sentences_per_example": 8,
        "max_subgraph_size": 2,
        "max_candidates_per_question": 32,
        "kg_config": KGConstructionConfig(backend="context_only"),
        "llm_kg_constructor": None,
        "kg_cache": None,
        "kg_cache_path": None,
        "kg_cache_lock": None,
        "seed": 42,
    }


def _2wiki_example():
    return {
        "_id": "wiki-1",
        "question": "Who is Alice's grandparent?",
        "answer": "Carol",
        "supporting_facts": [["Alice", 0], ["Bob", 0]],
        "evidences": [["Alice", "parent", "Bob"], ["Bob", "parent", "Carol"]],
        "context": [
            ["Alice", ["Alice's parent is Bob.", "Alice is an engineer."]],
            ["Bob", ["Bob's parent is Carol."]],
        ],
    }


def _candidate_units(rows):
    return [tuple(row["subgraph_units"]) for row in rows]


def test_2wiki_generation_rejects_evidences_in_its_input():
    with pytest.raises(ValueError, match="evidences"):
        generate_2wiki_candidates(
            retrieval_example=_2wiki_example(),
            **_text_kwargs(),
        )


def test_2wiki_normalization_drops_evidences():
    assert "evidences" not in normalize_2wiki_record(_2wiki_example())


def test_text_kg_signature_cannot_receive_benchmark_evidences():
    with pytest.raises(TypeError):
        construct_text_kg(
            provided_evidences=[["gold", "relation", "secret"]],
            sentence_records=[],
            question="Question?",
            config=KGConstructionConfig(backend="context_only"),
        )


def test_contaminated_2wiki_cache_is_rejected():
    example = _2wiki_example()
    config = KGConstructionConfig(backend="context_only")
    cache = {
        "2WikiMultiHopQA__dev__wiki-1": {
            "construction_method": "provided_plus_llm",
            "cache_signature": config.cache_signature,
            "kg_triples": [["gold", "relation", "secret"]],
        }
    }
    with pytest.raises(ValueError, match="Contaminated KG cache"):
        build_2wiki_rows(example=example, **{**_text_kwargs(), "kg_cache": cache})


def test_2wiki_candidates_are_identical_with_gold_removed_or_changed():
    original = _2wiki_example()
    removed = {k: v for k, v in original.items() if k not in {"answer", "supporting_facts", "evidences"}}
    changed = copy.deepcopy(original)
    changed["answer"] = "SECRET"
    changed["supporting_facts"] = [["Alice", 1]]
    changed["evidences"] = [["SECRET", "gold", "ONLY"]]

    baseline_rows = build_2wiki_rows(example=original, **_text_kwargs())
    removed_rows = build_2wiki_rows(example=removed, **_text_kwargs())
    changed_rows = build_2wiki_rows(example=changed, **_text_kwargs())
    baseline = _candidate_units(baseline_rows)
    assert baseline == _candidate_units(removed_rows)
    assert baseline == _candidate_units(changed_rows)
    assert baseline_rows[0]["graph_context_units"] == removed_rows[0]["graph_context_units"]
    assert baseline_rows[0]["graph_context_units"] == changed_rows[0]["graph_context_units"]
    assert "SECRET" not in " ".join(changed_rows[0]["graph_context_units"])

    baseline_generated = generate_2wiki_candidates(
        retrieval_example={
            "id": original["_id"],
            "question": original["question"],
            "context": original["context"],
        },
        **_text_kwargs(),
    )
    changed_generated = generate_2wiki_candidates(
        retrieval_example={
            "id": changed["_id"],
            "question": changed["question"],
            "context": changed["context"],
        },
        **_text_kwargs(),
    )
    for field in ("sentence_pool", "graph_context_units", "candidates"):
        assert baseline_generated[field] == changed_generated[field]


def test_hotpot_sentence_pool_and_candidates_ignore_supporting_facts():
    base = {
        "_id": "hotpot-1",
        "question": "Who is Alice's grandparent?",
        "answer": "Carol",
        "supporting_facts": [["Alice", 0], ["Bob", 0]],
        "context": [
            ["Alice", ["Alice's parent is Bob.", "Alice is an engineer."]],
            ["Bob", ["Bob's parent is Carol."]],
        ],
    }
    changed = copy.deepcopy(base)
    changed["supporting_facts"] = [["Alice", 1]]
    baseline = _candidate_units(build_hotpot_rows(example=base, **_text_kwargs()))
    assert baseline == _candidate_units(build_hotpot_rows(example=changed, **_text_kwargs()))


def test_ontology_candidates_do_not_insert_gold_explanation():
    item = {
        "Task Type": "Membership",
        "Answer Type": "BIN",
        "OWL Context": (
            "@prefix ex: <http://example.org/family#> .\n"
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
            "ex:Alice a owl:NamedIndividual ; ex:parent ex:Bob .\n"
        ),
    }
    qa = {
        "NL Question": "Is Alice related to Carol?",
        "SPARQL Query": "ASK WHERE { <http://example.org/family#Alice> <http://example.org/family#related> <http://example.org/family#Carol> }",
        "Answer": "TRUE",
        "Explanations": [["Alice related Carol"]],
    }
    rows = build_rows_for_qa(
        source_name="OWL2Bench_test",
        group_index=0,
        qa_index=0,
        item=item,
        qa=qa,
        split="dev",
        max_subgraph_size=3,
        min_subgraph_size=1,
        max_context_units=20,
        max_negative_per_example=1,
        candidate_beam_width=20,
        max_candidate_subgraphs=40,
    )
    assert rows
    assert all("Alice related Carol" not in row["subgraph_units"] for row in rows)
    assert all(not row["label"] for row in rows)

    baseline = generate_ontology_candidates(
        question=qa["NL Question"],
        sparql_query=qa["SPARQL Query"],
        owl_context=item["OWL Context"],
        max_subgraph_size=3,
        min_subgraph_size=1,
        max_context_units=20,
        candidate_beam_width=20,
        max_candidate_subgraphs=40,
    )
    changed_qa = copy.deepcopy(qa)
    changed_qa["Answer"] = "SECRET"
    changed_qa["Explanations"] = [["SECRET gold ONLY"]]
    changed = generate_ontology_candidates(
        question=changed_qa["NL Question"],
        sparql_query=changed_qa["SPARQL Query"],
        owl_context=item["OWL Context"],
        max_subgraph_size=3,
        min_subgraph_size=1,
        max_context_units=20,
        candidate_beam_width=20,
        max_candidate_subgraphs=40,
    )
    for field in ("candidate_units", "candidate_subgraphs", "unit_scores"):
        assert baseline[field] == changed[field]


@pytest.mark.parametrize(
    ("triple", "entities"),
    [
        ("<http://example.org/Alice> <http://example.org/parent> ?x", ["Alice"]),
        ("?x <http://example.org/parent> <http://example.org/Alice>", ["Alice"]),
    ],
)
def test_select_query_signature_retains_bound_entity_and_property(triple, entities):
    signature = extract_query_signature(
        question="Who is related to Alice?",
        sparql_query=f"SELECT ?x WHERE {{ {triple} }}",
    )
    assert signature["query_entities"] == entities
    assert signature["query_properties"] == ["parent"]


def test_ontology_reference_identity_matches_context_canonicalization():
    turtle = """
        @prefix ex: <http://example.org/> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ex:Child rdfs:subClassOf ex:Person .
        ex:parent rdfs:subPropertyOf ex:related .
        ex:grandRelated owl:propertyChainAxiom (ex:parent ex:related) .
    """
    context = set(parse_owl_context(turtle))
    reference = set(
        normalize_explanation(
            [
                "Child SubClassOf Person",
                "parent rdfs:subPropertyOf related",
                "PropertyChain(parent ∘ related) ⊑ grandRelated",
            ]
        )
    )
    assert reference <= context
    assert normalize_explanation(["PropertyChain(parent âˆ˜ related) âŠ‘ grandRelated"]) == [
        "ObjectPropertyChain(parent,related->grandRelated)"
    ]


def test_query_guided_ontology_pool_expands_through_schema_chain():
    axioms = [
        "alice rdf:type Child",
        "Child SubClassOf Person",
        "Person SubClassOf Agent",
        "unrelated likes noise",
    ]
    selected = relevant_context_axioms(
        axioms,
        question="Is Alice an agent?",
        sparql_query=(
            "ASK WHERE { <http://example.org/alice> "
            "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
            "<http://example.org/Agent> }"
        ),
        max_context_units=3,
    )
    assert set(selected) == set(axioms[:3])


class _GoldGuard(dict):
    def get(self, key, default=None):
        if key in {"label", "rank_target", "best_set_f1_to_gold"}:
            raise AssertionError(f"inference read forbidden field {key}")
        return super().get(key, default)


def test_inference_candidate_cap_never_reads_gold_fields():
    rows = [
        _GoldGuard(
            candidate_pre_rank_score=float(index),
            generation_rank=index,
            subgraph_units=[f"candidate-{index}"],
            label=index % 2,
            rank_target=1.0,
            best_set_f1_to_gold=1.0,
        )
        for index in range(5)
    ]
    selected = cap_inference_candidate_rows(rows, max_candidates=2)
    assert [row["subgraph_units"] for row in selected] == [
        ["candidate-4"],
        ["candidate-3"],
    ]


def test_text_dev_test_selection_is_disjoint_and_deterministic():
    examples = [{"_id": str(index)} for index in range(20)]
    first = select_disjoint_cohorts(
        examples, examples, max_dev_examples=7, max_test_examples=8, seed=42
    )
    second = select_disjoint_cohorts(
        examples, examples, max_dev_examples=7, max_test_examples=8, seed=42
    )
    assert first == second
    assert {row["_id"] for row in first[0]}.isdisjoint(
        row["_id"] for row in first[1]
    )
