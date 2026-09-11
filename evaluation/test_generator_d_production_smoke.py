"""Focused production smoke tests for frozen Generator D."""

from __future__ import annotations

import json
import hashlib

import pytest

from data.build_subgraph_training_data import (
    build_rows_for_qa,
    generate_ontology_candidates,
)
from data_processing.build_2wiki_subgraph_dataset import (
    generate_candidates as generate_2wiki,
    label_candidates as label_2wiki,
)
from data_processing.build_hotpot_subgraph_dataset import (
    generate_candidates as generate_hotpot,
    label_candidates as label_hotpot,
)
from data_processing.evidence_graph_candidates import GENERATOR_D_CONFIG
from data_processing.retrieval_contracts import clean_text_retrieval_input
from data_processing.text_kg_constructor import KGConstructionConfig
from evaluation.validate_gold_free_candidate_pools import onto_rows, text_rows
from training.train_gnn_subgraph_retriever import prepare_examples


def _bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _assert_candidate_bounds(candidates) -> None:
    assert candidates
    assert len(candidates) <= GENERATOR_D_CONFIG.max_candidates
    assert all(
        1 <= len(candidate) <= GENERATOR_D_CONFIG.max_support_size for candidate in candidates
    )


def _smoke_record(name: str, candidates) -> None:
    print(
        json.dumps(
            {
                "dataset": name,
                "candidate_count": len(candidates),
                "candidate_size_min": min(map(len, candidates)),
                "candidate_size_max": max(map(len, candidates)),
                "candidate_digest": hashlib.sha256(_bytes(candidates)).hexdigest(),
                "gold_deleted_identical": True,
                "gnn_preparation": "accepted",
            },
            sort_keys=True,
        )
    )


@pytest.mark.parametrize("name", ["HotpotQA", "2WikiMultiHopQA"])
def test_generator_d_real_text_smoke_is_gold_invariant_and_gnn_compatible(name):
    example = text_rows(name, 1, 42)[0]
    deleted = {
        key: value
        for key, value in example.items()
        if key not in {"answer", "supporting_facts", "evidences", "raw_evidences"}
    }
    clean = clean_text_retrieval_input(example)
    clean_deleted = clean_text_retrieval_input(deleted)
    assert set(clean) == {"id", "question", "context"}
    generator = generate_2wiki if name == "2WikiMultiHopQA" else generate_hotpot
    kwargs = {
        "split_name": "dev",
        "max_sentences_per_example": 30,
        "max_subgraph_size": 6,
        "max_candidates_per_question": 512,
        "kg_config": KGConstructionConfig(backend="context_only", max_triples=64),
        "llm_kg_constructor": None,
        "kg_cache": None,
        "kg_cache_path": None,
        "kg_cache_lock": None,
        "seed": 42,
    }
    normal = generator(retrieval_example=clean, **kwargs)
    gold_deleted = generator(retrieval_example=clean_deleted, **kwargs)
    frozen_fields = ("sentence_pool", "graph_context_units", "candidates")
    assert _bytes({key: normal[key] for key in frozen_fields}) == _bytes(
        {key: gold_deleted[key] for key in frozen_fields}
    )
    _assert_candidate_bounds(normal["candidates"])
    materializer = label_2wiki if name == "2WikiMultiHopQA" else label_hotpot
    test_rows = materializer(normal, supporting_facts=None, max_subgraph_size=6)
    assert test_rows
    assert not {
        "gold_support_units",
        "raw_supporting_facts",
        "label",
        "rank_target",
        "best_set_f1_to_gold",
    }.intersection(test_rows[0])

    # Materialize as an inference row without gold; this is sufficient to
    # exercise the unchanged GNN preparation schema.
    first = normal["candidates"][0]
    row = {
        "example_id": normal["example_id"],
        "dataset": name,
        "question": normal["question"],
        "subgraph_units": first,
        "subgraph_size": len(first),
        "graph_context_units": normal["graph_context_units"],
        "generation_rank": 0,
        "candidate_pre_rank_score": 0.0,
        "symbolic_features": [0.0] * 8,
        "gold_available_during_candidate_generation": False,
        "kg_construction_method": normal["kg_construction_method"],
    }
    assert prepare_examples([row], candidate_selection="inference")
    _smoke_record(name, normal["candidates"])


@pytest.mark.parametrize("name", ["Family_2hop", "Pizza_100_2hop", "OWL2Bench_2hop"])
def test_generator_d_real_ontology_smoke_is_gold_invariant_and_gnn_compatible(name):
    group_index, qa_index, item, qa = onto_rows(name)[0]
    question = str(qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or "")
    sparql = str(qa.get("SPARQL Query") or "")
    kwargs = {
        "question": question,
        "sparql_query": sparql,
        "owl_context": str(item.get("OWL Context") or ""),
        "max_subgraph_size": 6,
        "min_subgraph_size": 1,
        "max_context_units": 40,
        "candidate_beam_width": 96,
        "max_candidate_subgraphs": 512,
    }
    normal = generate_ontology_candidates(**kwargs)
    deleted_qa = {
        key: value
        for key, value in qa.items()
        if key not in {"Answer", "Explanations", "Gold Explanations", "Gold Explanation"}
    }
    gold_deleted = generate_ontology_candidates(
        **{
            **kwargs,
            "question": str(
                deleted_qa.get("NL Question")
                or deleted_qa.get("ABS Question")
                or deleted_qa.get("Task ID")
                or ""
            ),
            "sparql_query": str(deleted_qa.get("SPARQL Query") or ""),
        }
    )
    frozen_fields = ("candidate_units", "candidate_subgraphs", "unit_scores")
    assert _bytes({key: normal[key] for key in frozen_fields}) == _bytes(
        {key: gold_deleted[key] for key in frozen_fields}
    )
    assert normal["adjacency"] == gold_deleted["adjacency"]
    _assert_candidate_bounds(normal["candidate_subgraphs"])

    rows = build_rows_for_qa(
        source_name=name,
        group_index=group_index,
        qa_index=qa_index,
        item=item,
        qa=qa,
        split="test",
        max_subgraph_size=6,
        min_subgraph_size=1,
        max_context_units=40,
        max_negative_per_example=200,
        candidate_beam_width=96,
        max_candidate_subgraphs=512,
    )
    assert rows
    assert not {
        "gold_explanations",
        "gold_units",
        "label",
        "rank_target",
        "best_set_f1_to_gold",
    }.intersection(rows[0])
    assert prepare_examples(rows[:4], candidate_selection="inference")
    _smoke_record(name, normal["candidate_subgraphs"])


def test_generator_d_frozen_configuration_exact():
    assert vars(GENERATOR_D_CONFIG) == {
        "max_support_size": 6,
        "max_candidates": 512,
        "max_seeds": 32,
        "max_first_hops_per_seed": 6,
        "max_neighbors_per_state": 12,
        "per_branch_width": 2,
        "protect_query_anchors": True,
    }
