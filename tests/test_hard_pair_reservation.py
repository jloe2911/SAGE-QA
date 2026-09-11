import copy
import random
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.gnn_subgraph_retriever import GNNSubgraphRetriever
from training.train_gnn_subgraph_retriever import (
    sample_weighted_margin_pairs,
    select_reserved_hard_pair,
    subsample_candidate_rows,
)


def v1_subsample_candidate_rows(
    rows, max_pos=64, max_hard_neg=128, max_easy_neg=128
):
    positives = [r for r in rows if int(r.get("label", 0)) == 1]
    hard_negatives = [
        r
        for r in rows
        if int(r.get("label", 0)) == 0
        and float(r.get("best_set_f1_to_gold", 0.0)) > 0.0
    ]
    easy_negatives = [
        r
        for r in rows
        if int(r.get("label", 0)) == 0
        and float(r.get("best_set_f1_to_gold", 0.0)) == 0.0
    ]
    if len(positives) > max_pos:
        positives = random.sample(positives, max_pos)
    if len(hard_negatives) > max_hard_neg:
        hard_negatives = sorted(
            hard_negatives,
            key=lambda r: float(r.get("best_set_f1_to_gold", 0.0)),
            reverse=True,
        )[:max_hard_neg]
    if len(easy_negatives) > max_easy_neg:
        easy_negatives = random.sample(easy_negatives, max_easy_neg)
    sampled = positives + hard_negatives + easy_negatives
    random.shuffle(sampled)
    return sampled


def v1_sample_weighted_margin_pairs(targets, max_pairs=512):
    exact = [i for i, target in enumerate(targets) if target >= 0.999]
    sufficient = [i for i, target in enumerate(targets) if 0.899 <= target < 0.999]
    partial = [i for i, target in enumerate(targets) if 0.0 < target < 0.899]
    irrelevant = [i for i, target in enumerate(targets) if target == 0.0]
    pairs = []
    for i in exact:
        for j in sufficient:
            pairs.append((i, j, 2.0))
    for i in exact + sufficient:
        for j in partial:
            pairs.append((i, j, 3.0))
    for i in exact + sufficient:
        for j in irrelevant:
            pairs.append((i, j, 2.0))
    for i in partial:
        for j in irrelevant:
            pairs.append((i, j, 0.5))
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)
    return pairs


def row(index, target, pre_rank, *, f1=None, generation_rank=None):
    if f1 is None:
        f1 = target if 0.0 < target < 0.9 else 0.0
    return {
        "row_id": f"r{index}",
        "label": int(target >= 0.9),
        "rank_target": target,
        "best_set_f1_to_gold": f1,
        "candidate_pre_rank_score": pre_rank,
        "generation_rank": index if generation_rank is None else generation_rank,
        "materialization_order": index,
        "subgraph_units": [f"u{index}"],
        "subgraph_size": 1,
    }


def test_complete_uses_target_then_pre_rank_then_generation_order():
    rows = [
        row(0, 0.9, 99.0),
        row(1, 1.0, 2.0, generation_rank=8),
        row(2, 1.0, 2.0, generation_rank=3),
        row(3, 0.3, 1.0),
    ]
    complete, _ = select_reserved_hard_pair(rows)
    assert complete["row_id"] == "r2"


def test_competitor_prefers_highest_pre_ranked_partial_over_irrelevant():
    rows = [
        row(0, 1.0, 1.0),
        row(1, 0.3, 4.0),
        row(2, 0.1, 6.0),
        row(3, 0.0, 100.0),
    ]
    _, competitor = select_reserved_hard_pair(rows)
    assert competitor["row_id"] == "r2"


def test_competitor_falls_back_to_highest_pre_ranked_irrelevant():
    rows = [row(0, 1.0, 1.0), row(1, 0.0, 4.0), row(2, 0.0, 6.0)]
    _, competitor = select_reserved_hard_pair(rows)
    assert competitor["row_id"] == "r2"


def test_competitor_tie_uses_generation_order():
    rows = [
        row(0, 1.0, 1.0),
        row(1, 0.3, 5.0, generation_rank=9),
        row(2, 0.3, 5.0, generation_rank=4),
    ]
    _, competitor = select_reserved_hard_pair(rows)
    assert competitor["row_id"] == "r2"


def test_candidate_reservation_preserves_existing_category_budgets():
    rows = [row(i, 1.0, float(i)) for i in range(4)]
    rows += [row(10 + i, 0.3, float(i)) for i in range(4)]
    rows += [row(20 + i, 0.0, float(i)) for i in range(4)]

    random.seed(42)
    baseline = subsample_candidate_rows(
        copy.deepcopy(rows), max_pos=2, max_hard_neg=2, max_easy_neg=2
    )
    random.seed(42)
    reserved = subsample_candidate_rows(
        copy.deepcopy(rows),
        max_pos=2,
        max_hard_neg=2,
        max_easy_neg=2,
        reserve_hard_pair=True,
    )

    assert len(reserved) == len(baseline) == 6
    assert sum(r.get("_hard_pair_role") == "complete" for r in reserved) == 1
    assert sum(r.get("_hard_pair_role") == "competitor" for r in reserved) == 1


def test_disabled_candidate_reservation_is_rng_equivalent_to_v1():
    rows_a = [row(i, 0.0, float(i)) for i in range(10)]
    rows_b = copy.deepcopy(rows_a)
    random.seed(91)
    sample_a = v1_subsample_candidate_rows(rows_a, max_easy_neg=4)
    state_a = random.getstate()
    random.seed(91)
    sample_b = subsample_candidate_rows(
        rows_b, max_easy_neg=4, reserve_hard_pair=False
    )
    state_b = random.getstate()
    assert [r["row_id"] for r in sample_a] == [r["row_id"] for r in sample_b]
    assert state_a == state_b


def test_disabled_pair_reservation_is_rng_equivalent_to_v1():
    targets = [1.0] * 3 + [0.9] * 3 + [0.3] * 20 + [0.0] * 20
    random.seed(37)
    pairs_a = v1_sample_weighted_margin_pairs(targets, max_pairs=16)
    state_a = random.getstate()
    random.seed(37)
    pairs_b = sample_weighted_margin_pairs(targets, max_pairs=16)
    state_b = random.getstate()
    assert pairs_a == pairs_b
    assert state_a == state_b


def test_margin_reservation_preserves_budget_and_pair_exactly_once():
    targets = [1.0] + [0.3] * 39
    random.seed(7)
    baseline = sample_weighted_margin_pairs(targets, max_pairs=8)
    baseline_state = random.getstate()
    random.seed(7)
    reserved = sample_weighted_margin_pairs(
        targets, max_pairs=8, reserved_pair=(0, 39)
    )
    reserved_state = random.getstate()
    assert len(reserved) == len(baseline) == 8
    assert sum(pair[:2] == (0, 39) for pair in reserved) == 1
    assert len(set(reserved)) == len(reserved)
    assert baseline_state == reserved_state


class DummyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.weight = torch.nn.Parameter(torch.ones(1))


def test_frozen_encoder_does_not_freeze_graphsage_or_classifier():
    with patch("utils.model_loader.load_encoder", return_value=DummyEncoder()):
        model = GNNSubgraphRetriever(
            model_name="synthetic",
            gnn_hidden_dim=8,
            gnn_layers=2,
            classifier_hidden_dim=8,
            freeze_encoder=True,
        )
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.gnn_layers.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_final_training_command_keeps_listwise_disabled():
    from pathlib import Path

    launcher = Path("scripts/run_sageqa_hard_pair_v2_all.sh").read_text(
        encoding="utf-8"
    )
    assert "--listwise-weight 0.0" in launcher
    assert "--freeze-encoder" in launcher
    assert "--hard-pair-reservation" in launcher
