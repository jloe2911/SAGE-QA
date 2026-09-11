from evaluation.compare_size_balanced_candidate_composer import (
    compose_text_balanced,
)
from data_processing.build_2wiki_subgraph_dataset import (
    candidate_pre_rank_score,
    parse_sentence_unit,
    token_set,
)


def test_size_balanced_text_composer_reserves_larger_candidates_deterministically():
    pool = [f"Sentence::{index}" for index in range(10)]
    kwargs = {
        "sentence_pool": pool,
        "question": "sentence",
        "max_size": 3,
        "budget": 20,
        "seed": 42,
        "pre_rank": lambda _question, units: float(-sum(int(unit.split("::")[1]) for unit in units)),
    }
    first = compose_text_balanced(**kwargs)
    second = compose_text_balanced(**kwargs)
    assert first == second
    assert len(first) == 20
    assert sum(len(candidate) >= 2 for candidate in first) >= 10


def test_size_balanced_text_composer_never_exceeds_fixed_cap():
    candidates = compose_text_balanced(
        sentence_pool=["only"],
        question="only",
        max_size=4,
        budget=512,
        seed=42,
        pre_rank=lambda _question, _units: 0.0,
    )
    assert candidates == [["only"]]


def test_fast_text_pre_rank_preserves_existing_pre_rank_order():
    pool = [
        "Sentence::Alpha::0::Alpha met Beta.",
        "Sentence::Beta::1::Beta visited Gamma.",
        "Sentence::Gamma::2::Gamma knew Alpha.",
        "Sentence::Delta::3::Unrelated text.",
    ]
    kwargs = {
        "sentence_pool": pool,
        "question": "How are Alpha and Beta related?",
        "max_size": 3,
        "budget": 12,
        "seed": 42,
        "pre_rank": candidate_pre_rank_score,
    }
    slow = compose_text_balanced(**kwargs)
    fast = compose_text_balanced(
        **kwargs,
        token_setter=token_set,
        unit_parser=parse_sentence_unit,
    )
    assert fast == slow
