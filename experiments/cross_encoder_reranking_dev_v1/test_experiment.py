import json
from pathlib import Path

import pytest

from experiments.cross_encoder_reranking_dev_v1.run_experiment import (
    MODEL_NAME,
    MODEL_PATH_ENV,
    MODEL_REVISION,
    candidate_identity,
    configuration,
    load_pretrained_components,
    pretrained_model_source,
    select_training_pairs,
    serialize_candidate,
    validate_frozen_model_snapshot,
)


def row(units, *, exact=False, complete=False, f1=0.0):
    return {
        "subgraph_units": units,
        "exact_match_any_gold": exact,
        "contains_any_gold_explanation": complete,
        "best_set_f1_to_gold": f1,
    }


def test_text_serialization_preserves_identity_and_order():
    value = serialize_candidate(
        "Who?",
        ["SENT::Page B::7::Second.", "SENT::Page A::2::First."],
        "text",
    )
    assert value == (
        "Question:\nWho?\n\nCandidate evidence:\n"
        "[1] Title: Page B\nSentence index: 7\nText: Second.\n"
        "[2] Title: Page A\nSentence index: 2\nText: First."
    )


def test_ontology_serialization_is_faithful():
    value = serialize_candidate("Is A B?", ["A subClassOf B", "x rdf:type A"], "ontology")
    assert value.endswith("[1] A subClassOf B\n[2] x rdf:type A")


def test_pairs_are_within_example_deterministic_and_ordered_by_target():
    rows = [
        row(["complete"], complete=True),
        row(["partial"], f1=0.5),
        row(["irrelevant"]),
    ]
    first = select_training_pairs("example", rows, max_pairs=8)
    second = select_training_pairs("example", rows, max_pairs=8)
    assert [(candidate_identity(a), candidate_identity(b)) for a, b in first] == [
        (candidate_identity(a), candidate_identity(b)) for a, b in second
    ]
    assert len(first) == 3
    targets = {(a["subgraph_units"][0], b["subgraph_units"][0]) for a, b in first}
    assert ("complete", "partial") in targets
    assert ("complete", "irrelevant") in targets
    assert ("partial", "irrelevant") in targets


def test_local_snapshot_is_validated_and_used_without_changing_identity(
    monkeypatch,
):
    snapshot = Path("frozen-snapshot").resolve()
    monkeypatch.setenv(MODEL_PATH_ENV, str(snapshot))
    monkeypatch.setattr(
        "experiments.cross_encoder_reranking_dev_v1.run_experiment.validate_frozen_model_snapshot",
        lambda path: None,
    )
    source, revision_kwargs = pretrained_model_source()
    assert source == str(snapshot)
    assert revision_kwargs == {}
    assert configuration()["model_name"] == MODEL_NAME
    assert configuration()["model_revision"] == MODEL_REVISION
    assert str(snapshot) not in json.dumps(configuration())


def test_local_snapshot_requires_config_tokenizer_and_weights(monkeypatch):
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with pytest.raises(FileNotFoundError, match="config.json"):
        validate_frozen_model_snapshot(Path("incomplete-snapshot"))


def test_pretrained_loaders_receive_local_snapshot_and_offline_flag(monkeypatch):
    snapshot = Path("frozen-snapshot").resolve()
    calls = []

    class Loader:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append((source, kwargs))
            return object()

    monkeypatch.setenv(MODEL_PATH_ENV, str(snapshot))
    monkeypatch.setattr(
        "experiments.cross_encoder_reranking_dev_v1.run_experiment.validate_frozen_model_snapshot",
        lambda path: None,
    )
    monkeypatch.setattr(
        "experiments.cross_encoder_reranking_dev_v1.run_experiment.AutoTokenizer", Loader
    )
    monkeypatch.setattr(
        "experiments.cross_encoder_reranking_dev_v1.run_experiment.AutoModelForSequenceClassification",
        Loader,
    )
    load_pretrained_components()
    assert calls == [
        (str(snapshot), {"local_files_only": True}),
        (str(snapshot), {"num_labels": 1, "local_files_only": True}),
    ]
