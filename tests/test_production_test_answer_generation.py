import json
import threading
import time
from pathlib import Path

import pytest

from evaluation import adaptive_support_aggregation, adaptive_support_aggregation_v2
from evaluation.evaluate_owl_qa_predictions import answer_set_scores
from evaluation.hotpot_official_eval import exact_match_score, f1_score
from generation import generate_hotpot_answers_with_llm as text_reader_module
from generation import generate_owl_answers_with_llm as ontology_reader_module
from generation import run_production_test_answer_generation as runner


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture()
def production_fixture(tmp_path):
    retrieval_rows = []
    candidate_root = tmp_path / "candidates"
    source_root = tmp_path / "sources"
    expected_support = {}
    for index, (dataset, directory, domain, source_relative) in enumerate(runner.DATASETS):
        example_id = f"{dataset}__test__example-{index}"
        if domain == "ontology":
            example_id = f"{dataset}__g0__q0__task-{index}"
            question = "Is Person an entity?"
            support_base = "Person rdf:type Entity"
            metadata = {
                "example_id": example_id,
                "source_name": dataset,
                "split": "test",
                "question": question,
                "sparql_query": (
                    "ASK WHERE { <http://example#Person> "
                    "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
                    "<http://example#Entity> }"
                ),
                "answer_type": "BIN",
            }
        else:
            question = "Is the frozen support sufficient?"
            support_base = f"SENT::Page {index}::0::The frozen support is sufficient."
            metadata = {
                "example_id": example_id,
                "dataset": dataset,
                "split": "test",
                "question": question,
                "hop": "2hop",
                "answer_type": "OPEN",
            }
        methods = {}
        for method, setting in runner.CONFIGURATIONS:
            support = [f"{support_base} [{method}/{setting}]"]
            expected_support[(example_id, method, setting)] = support
            methods.setdefault(method, {})[setting] = {
                "retrieved_evidence_units": support,
                "selected_k": 1,
            }
        retrieval_rows.append(
            {
                "dataset": dataset,
                "domain": domain,
                "split": "test",
                "example_id": example_id,
                "evaluation_eligible": True,
                "gold_explanations": [[support_base + " [gnn_only/k1]"]],
                "methods": methods,
            }
        )
        _write_jsonl(candidate_root / directory / "test_subgraph_retrieval.jsonl", [metadata])
        source = source_root / source_relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("fixture\n", encoding="utf-8")

    retrieval = tmp_path / "retrieval.jsonl"
    _write_jsonl(retrieval, retrieval_rows)
    return {
        "retrieval": retrieval,
        "rows": retrieval_rows,
        "candidate_root": candidate_root,
        "source_root": source_root,
        "output": tmp_path / "output",
        "expected_support": expected_support,
    }


def _fake_text_reader(calls):
    def reader(question, support, model):
        calls.append((question, list(support), model))
        return {
            "predicted_answer": "yes",
            "explanation": "fixture",
            "raw_response": '{"answer":"yes","explanation":"fixture"}',
            "answer_source": "gpt-4.1-mini_reader",
        }

    return reader


def _fake_ontology_reader(calls):
    def reader(metadata, support, model):
        calls.append((dict(metadata), list(support), model))
        return {
            "predicted_answer": "TRUE",
            "explanation": "fixture",
            "raw_response": '{"answer":"TRUE","explanation":"fixture"}',
            "answer_source": "deterministic_owl_proof",
        }

    return reader


def _generate(fixture, text_calls=None, ontology_calls=None, resume=False, workers=8):
    text_calls = [] if text_calls is None else text_calls
    ontology_calls = [] if ontology_calls is None else ontology_calls
    return runner.generate_phase(
        fixture["retrieval"],
        fixture["candidate_root"],
        fixture["output"],
        resume=resume,
        workers=workers,
        text_reader=_fake_text_reader(text_calls),
        ontology_reader=_fake_ontology_reader(ontology_calls),
    )


def _use_retrieval_rows(fixture, rows) -> None:
    fixture["rows"] = rows
    _write_jsonl(fixture["retrieval"], rows)


def _make_all_configuration_support_identical(fixture) -> None:
    for retrieval_row in fixture["rows"]:
        first_method, first_setting = runner.CONFIGURATIONS[0]
        support = list(retrieval_row["methods"][first_method][first_setting]["retrieved_evidence_units"])
        for method, setting in runner.CONFIGURATIONS:
            retrieval_row["methods"][method][setting]["retrieved_evidence_units"] = list(support)
            fixture["expected_support"][(retrieval_row["example_id"], method, setting)] = list(
                support
            )
    _write_jsonl(fixture["retrieval"], fixture["rows"])


def test_exact_frozen_support_is_passed_and_only_requested_configs_are_generated(
    production_fixture,
):
    text_calls, ontology_calls = [], []
    _generate(production_fixture, text_calls, ontology_calls)
    predictions = runner.load_jsonl(production_fixture["output"] / "predictions.jsonl")

    assert len(predictions) == len(production_fixture["rows"]) * 4
    assert {(row["method"], row["setting"]) for row in predictions} == set(
        runner.CONFIGURATIONS
    )
    assert all(row["setting"] != "k3" for row in predictions)
    for row in predictions:
        key = (row["example_id"], row["method"], row["setting"])
        assert row["selected_support"] == production_fixture["expected_support"][key]
        assert row["frozen_support_sha256"] == runner.canonical_hash(row["selected_support"])

    passed_supports = [support for _question, support, _model in text_calls]
    passed_supports += [support for _metadata, support, _model in ontology_calls]
    assert sorted(map(json.dumps, passed_supports)) == sorted(
        map(json.dumps, production_fixture["expected_support"].values())
    )


def test_adaptive_and_symbolic_selection_are_never_recomputed(
    production_fixture, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("selection/reranking must not be recomputed")

    monkeypatch.setattr(adaptive_support_aggregation, "adaptive_support_aggregate", forbidden)
    monkeypatch.setattr(adaptive_support_aggregation_v2, "adaptive_v2_support_aggregate", forbidden)
    # The fixture contains no ranked candidates or adaptive policy decisions; generation
    # can only succeed by copying retrieved_evidence_units directly.
    _generate(production_fixture)


def test_generate_does_not_open_any_original_gold_source(production_fixture):
    # The source files are irrelevant to Phase 1 and may be absent.
    for _dataset, _directory, _domain, relative in runner.DATASETS:
        (production_fixture["source_root"] / relative).unlink()
    _generate(production_fixture)


def test_resume_neither_duplicates_nor_reissues_completed_examples(production_fixture):
    _generate(production_fixture)
    predictions = production_fixture["output"] / "predictions.jsonl"
    before = predictions.read_bytes()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("completed calls must not be reissued")

    runner.generate_phase(
        production_fixture["retrieval"],
        production_fixture["candidate_root"],
        production_fixture["output"],
        resume=True,
        text_reader=forbidden,
        ontology_reader=forbidden,
    )
    assert predictions.read_bytes() == before
    rows = runner.load_jsonl(predictions)
    assert len({runner.prediction_key(row) for row in rows}) == len(rows)


def test_workers_one_and_many_schedule_the_same_logical_prediction_keys(production_fixture):
    single = dict(production_fixture)
    single["output"] = production_fixture["output"].parent / "single"
    parallel = dict(production_fixture)
    parallel["output"] = production_fixture["output"].parent / "parallel"

    _generate(single, workers=1)
    _generate(parallel, workers=4)

    single_rows = runner.load_jsonl(single["output"] / "predictions.jsonl")
    parallel_rows = runner.load_jsonl(parallel["output"] / "predictions.jsonl")
    expected = runner.expected_prediction_keys(production_fixture["rows"])
    assert {runner.prediction_key(row) for row in single_rows} == expected
    assert {runner.prediction_key(row) for row in parallel_rows} == expected
    assert len(single_rows) == len(expected)
    assert len(parallel_rows) == len(expected)


def test_identical_reader_inputs_make_one_call_and_reuse_output(production_fixture):
    _make_all_configuration_support_identical(production_fixture)
    text_calls, ontology_calls = [], []

    _generate(production_fixture, text_calls, ontology_calls, workers=4)

    assert len(text_calls) + len(ontology_calls) == len(production_fixture["rows"])
    predictions = runner.load_jsonl(production_fixture["output"] / "predictions.jsonl")
    by_example = {}
    for row in predictions:
        output = (
            row["predicted_answer"],
            row["explanation"],
            row["raw_response"],
            row["answer_source"],
        )
        by_example.setdefault(row["example_id"], set()).add(output)
    assert all(len(outputs) == 1 for outputs in by_example.values())


def test_different_supports_make_separate_reader_calls(production_fixture):
    row = production_fixture["rows"][0]
    _use_retrieval_rows(production_fixture, [row])
    shared = ["SENT::Page::0::shared"]
    different = ["SENT::Page::1::different"]
    for method, setting in runner.CONFIGURATIONS:
        row["methods"][method][setting]["retrieved_evidence_units"] = list(shared)
    method, setting = runner.CONFIGURATIONS[-1]
    row["methods"][method][setting]["retrieved_evidence_units"] = list(different)
    _write_jsonl(production_fixture["retrieval"], [row])
    text_calls = []

    _generate(production_fixture, text_calls=text_calls, workers=4)

    assert len(text_calls) == 2
    assert {tuple(call[1]) for call in text_calls} == {tuple(shared), tuple(different)}


def test_parallel_generation_writes_valid_jsonl_only_from_main_thread(
    production_fixture, monkeypatch
):
    main_thread = threading.get_ident()
    writer_threads = []
    original_append = runner.append_jsonl

    def checked_append(path, row):
        writer_threads.append(threading.get_ident())
        return original_append(path, row)

    def delayed_text_reader(question, support, model):
        time.sleep(0.002)
        return _fake_text_reader([])(question, support, model)

    monkeypatch.setattr(runner, "append_jsonl", checked_append)
    runner.generate_phase(
        production_fixture["retrieval"],
        production_fixture["candidate_root"],
        production_fixture["output"],
        workers=4,
        text_reader=delayed_text_reader,
        ontology_reader=_fake_ontology_reader([]),
    )

    assert writer_threads
    assert set(writer_threads) == {main_thread}
    path = production_fixture["output"] / "predictions.jsonl"
    rows = runner.load_jsonl(path)
    assert len(rows) == len(runner.expected_prediction_keys(production_fixture["rows"]))
    assert all(isinstance(json.loads(line), dict) for line in path.read_text(encoding="utf-8").splitlines())


def test_parallel_resume_skips_completed_predictions_without_duplicates(production_fixture):
    _generate(production_fixture, workers=4)
    predictions_path = production_fixture["output"] / "predictions.jsonl"
    all_rows = runner.load_jsonl(predictions_path)
    completed_rows = [row for row in all_rows if row["domain"] == "ontology"][:5]
    completed_keys = {runner.prediction_key(row) for row in completed_rows}
    (production_fixture["output"] / "generation_freeze.json").unlink()
    _write_jsonl(predictions_path, completed_rows)
    text_calls, ontology_calls = [], []

    _generate(
        production_fixture,
        text_calls=text_calls,
        ontology_calls=ontology_calls,
        resume=True,
        workers=4,
    )

    rows = runner.load_jsonl(predictions_path)
    keys = [runner.prediction_key(row) for row in rows]
    assert len(keys) == len(set(keys))
    assert set(keys) == runner.expected_prediction_keys(production_fixture["rows"])
    called_inputs = {
        (call[0]["example_id"], tuple(call[1])) for call in ontology_calls
    }
    completed_inputs = {
        (row["example_id"], tuple(row["selected_support"])) for row in completed_rows
    }
    assert completed_inputs.isdisjoint(called_inputs)
    assert completed_keys <= set(keys)


def test_generation_workers_receive_no_gold_fields(production_fixture):
    seen_worker_threads = set()

    def guarded_ontology_reader(metadata, support, model):
        seen_worker_threads.add(threading.get_ident())
        assert runner.FORBIDDEN_GENERATION_FIELDS.isdisjoint(metadata)
        return _fake_ontology_reader([])(metadata, support, model)

    runner.generate_phase(
        production_fixture["retrieval"],
        production_fixture["candidate_root"],
        production_fixture["output"],
        workers=4,
        text_reader=_fake_text_reader([]),
        ontology_reader=guarded_ontology_reader,
    )

    assert seen_worker_threads
    assert threading.get_ident() not in seen_worker_threads


def test_generate_cli_defaults_to_eight_workers():
    args = runner.build_parser().parse_args(["generate"])
    assert args.workers == 8


def test_gold_is_first_accessed_only_after_complete_generation_freeze(production_fixture):
    _generate(production_fixture)
    accesses = []

    def gold_loader(dataset, source, ids):
        accesses.append((dataset, source, set(ids)))
        domain = runner.DATASET_INFO[dataset][1]
        return {example_id: ("TRUE" if domain == "ontology" else "yes") for example_id in ids}

    metrics = runner.evaluate_phase(
        production_fixture["retrieval"],
        production_fixture["output"],
        source_root=production_fixture["source_root"],
        gold_loader=gold_loader,
    )
    assert len(accesses) == len(runner.DATASETS)
    assert metrics["status"] == "complete_frozen"
    assert (production_fixture["output"] / "metrics.json").is_file()
    assert (production_fixture["output"] / "per_example_end_to_end.jsonl").is_file()
    assert (production_fixture["output"] / "summary.md").is_file()


def test_predictions_cannot_change_after_freeze_and_gold_stays_closed(production_fixture):
    _generate(production_fixture)
    predictions = production_fixture["output"] / "predictions.jsonl"
    with predictions.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    accesses = []

    with pytest.raises(ValueError, match="changed after the generation freeze"):
        runner.evaluate_phase(
            production_fixture["retrieval"],
            production_fixture["output"],
            source_root=production_fixture["source_root"],
            gold_loader=lambda *args: accesses.append(args),
        )
    assert accesses == []


def test_default_ontology_deterministic_proof_is_identical_to_existing_generator():
    metadata = {
        "example_id": "FamilyOWL_1hop__g0__q0__fixture",
        "question": "Is Alice a person?",
        "sparql_query": (
            "ASK WHERE { <http://example#Alice> "
            "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
            "<http://example#Person> }"
        ),
    }
    support = ["Alice rdf:type Person"]
    expected = ontology_reader_module.infer_owl_boolean_answer(metadata, support)
    actual = runner.default_ontology_reader(metadata, support, runner.MODEL_NAME)
    assert expected is not None
    assert actual["predicted_answer"] == expected["answer"]
    assert actual["explanation"] == expected["explanation"]
    assert actual["answer_source"] == "deterministic_owl_proof"


def test_existing_answer_normalization_and_evaluation_semantics_are_preserved():
    question = "Is this supported?"
    assert text_reader_module.normalize_generated_answer(question, "TRUE") == "yes"
    text_actual = runner._answer_scores("text", "The Eiffel Tower", "Eiffel Tower")
    text_f1, text_precision, text_recall = f1_score("The Eiffel Tower", "Eiffel Tower")
    assert text_actual == (
        float(exact_match_score("The Eiffel Tower", "Eiffel Tower")),
        text_f1,
        text_precision,
        text_recall,
    )
    assert runner._answer_scores("ontology", "Alice; Bob", "Bob and Alice") == answer_set_scores(
        "Alice; Bob", "Bob and Alice"
    )


def test_evaluation_uses_frozen_support_and_does_not_regenerate_it(production_fixture):
    _generate(production_fixture)

    def gold_loader(dataset, _source, ids):
        domain = runner.DATASET_INFO[dataset][1]
        return {example_id: ("TRUE" if domain == "ontology" else "yes") for example_id in ids}

    runner.evaluate_phase(
        production_fixture["retrieval"],
        production_fixture["output"],
        source_root=production_fixture["source_root"],
        gold_loader=gold_loader,
    )
    rows = runner.load_jsonl(production_fixture["output"] / "per_example_end_to_end.jsonl")
    for row in rows:
        assert row["selected_support"] == production_fixture["expected_support"][
            (row["example_id"], row["method"], row["setting"])
        ]
