from __future__ import annotations

import json
import io

import torch

from models.gnn_subgraph_retriever import GNNSubgraphRetriever
from training.run_static_hard_negative_refinement import (
    freeze_except_classifier,
    grouped_jsonl,
    prepare_static_pair,
)


def candidate(example_id, units, target, pre_rank, generation_rank):
    return {
        "example_id": example_id,
        "question": "q",
        "sparql_query": "",
        "source_name": "FamilyOWL_1hop",
        "subgraph_units": units,
        "candidate_pre_rank_score": pre_rank,
        "generation_rank": generation_rank,
        "exact_match_any_gold": target == 1.0,
        "contains_any_gold_explanation": target == 0.9,
        "best_set_f1_to_gold": target / 0.5 if 0.0 < target < 0.9 else 0.0,
        "gold_units": ["gold"],
        "gold_explanations": [["gold"]],
        "gold_available_during_candidate_generation": False,
    }


def test_static_pair_materializes_only_audited_pair_and_full_graph():
    rows = [
        candidate("x", ["irrelevant"], 0.0, 99.0, 0),
        candidate("x", ["partial-low"], 0.1, 1.0, 1),
        candidate("x", ["partial-high"], 0.2, 2.0, 2),
        candidate("x", ["complete-low"], 0.9, 3.0, 3),
        candidate("x", ["complete-best-target"], 1.0, 0.0, 4),
    ]
    example = prepare_static_pair(rows)
    assert example is not None
    assert len(example["candidate_rows"]) == 2
    assert example["candidate_rows"][0]["subgraph_units"] == ["complete-best-target"]
    assert example["candidate_rows"][1]["subgraph_units"] == ["partial-high"]
    assert set(example["candidate_axioms"]) == {
        "irrelevant", "partial-low", "partial-high", "complete-low", "complete-best-target"
    }


def test_irrelevant_fallback():
    rows = [candidate("x", ["complete"], 1.0, 1.0, 0), candidate("x", ["none"], 0.0, 2.0, 1)]
    example = prepare_static_pair(rows)
    assert [row["subgraph_units"] for row in example["candidate_rows"]] == [["complete"], ["none"]]


def test_grouped_jsonl_streams_contiguous_examples(monkeypatch):
    values = [{"example_id": "a", "x": 1}, {"example_id": "a", "x": 2}, {"example_id": "b", "x": 3}]
    content = "".join(json.dumps(value) + "\n" for value in values)
    class FakePath:
        def open(self, *args, **kwargs):
            return io.StringIO(content)
        def __str__(self):
            return "rows.jsonl"
    groups = list(grouped_jsonl(FakePath()))
    assert [(key, len(rows)) for key, rows in groups] == [("a", 2), ("b", 1)]


def test_only_classifier_is_trainable(monkeypatch):
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.gnn_layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.classifier = torch.nn.Sequential(torch.nn.Linear(2, 1))

    model = Stub()
    freeze_except_classifier(model)
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert not any(parameter.requires_grad for name, parameter in model.named_parameters() if not name.startswith("classifier."))
