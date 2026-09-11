import json
import os
import shutil
import uuid
from pathlib import Path

import pytest

from generation import run_final_manuscript_baselines as runner


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture()
def baseline_fixture():
    parent = ROOT / "outputs" / "test_runs"
    parent.mkdir(parents=True, exist_ok=True)
    work = parent / f"final_baselines_{os.getpid()}_{uuid.uuid4().hex}"
    work.mkdir()
    try:
        lexical = work / "lexical" / "per_example_retrieval.jsonl"
        gnn = work / "gnn" / "predictions_frozen.jsonl"
        candidate_root = work / "candidates"
        output = work / "output"
        support_gold = work / "support_gold.jsonl"
        protected_root = work / "protected"
        protected_file = protected_root / "frozen.txt"
        protected_file.parent.mkdir(parents=True)
        protected_file.write_text("do not change\n", encoding="utf-8")
        protected_manifest = protected_root / "artifact_manifest.json"
        _write_json(
            protected_manifest,
            {
                "files": {
                    "frozen.txt": {
                        "size_bytes": protected_file.stat().st_size,
                        "sha256": runner.reader_pipeline.sha256(protected_file),
                    }
                }
            },
        )

        examples = [
            {
                "dataset": "HotpotQA",
                "directory": "HotpotQA",
                "domain": "text",
                "example_id": "HotpotQA__test__fixture",
                "question": "Is this supported?",
                "unit": "SENT::Page::0::This is supported.",
                "full": ["SENT::Page::0::This is supported.", "SENT::Page::1::More context."],
                "metadata": {
                    "dataset": "HotpotQA",
                    "example_id": "HotpotQA__test__fixture",
                    "split": "test",
                    "question": "Is this supported?",
                    "answer_type": "OPEN",
                },
            },
            {
                "dataset": "FamilyOWL_1hop",
                "directory": "FamilyOWL_1hop",
                "domain": "ontology",
                "example_id": "FamilyOWL_1hop__g0__q0__fixture",
                "question": "Is Alice a Person?",
                "unit": "Alice rdf:type Person",
                "full": ["Alice rdf:type Person", "Person SubClassOf Agent"],
                "metadata": {
                    "source_name": "FamilyOWL_1hop",
                    "example_id": "FamilyOWL_1hop__g0__q0__fixture",
                    "split": "test",
                    "question": "Is Alice a Person?",
                    "answer_type": "BIN",
                    "group_index": 0,
                    "qa_index": 0,
                    "sparql_query": "ASK WHERE { <x#Alice> <rdf#type> <x#Person> }",
                },
            },
        ]
        lexical_rows = []
        gnn_rows = []
        support_rows = []
        contexts = {}
        for item in examples:
            lexical_rows.append(
                {
                    "dataset": item["dataset"],
                    "example_id": item["example_id"],
                    "settings": {
                        "k1": {
                            "requested_k": 1,
                            "actual_candidates_aggregated": 1,
                            "retrieved_evidence_units": [item["unit"]],
                        },
                        "k3": {
                            "requested_k": 3,
                            "actual_candidates_aggregated": 1,
                            "retrieved_evidence_units": [item["unit"], "MUST NOT BE USED"],
                        },
                    },
                }
            )
            node = (
                runner.evidence_node(item["unit"])
                if item["domain"] == "text"
                else runner.unit_node(item["unit"])
            )
            gnn_rows.append(
                {
                    "dataset": item["dataset"],
                    "example_id": item["example_id"],
                    "native_ranked_entities": [
                        {"rank": 1, "entity": node, "probability": 0.8},
                        {"rank": 2, "entity": "EVIDENCE::MUST NOT BE USED", "probability": 0.1},
                    ],
                }
            )
            _write_jsonl(
                candidate_root / item["directory"] / "test_subgraph_retrieval.jsonl",
                [{**item["metadata"], "subgraph_units": [item["unit"]]}],
            )
            support_rows.append(
                {"example_id": item["example_id"], "gold_explanations": [[item["unit"]]]}
            )
            contexts[item["example_id"]] = item["full"]
        _write_jsonl(lexical, lexical_rows)
        _write_json(
            lexical.parent / "artifact_manifest.json",
            {
                "files": {
                    lexical.name: {
                        "size_bytes": lexical.stat().st_size,
                        "sha256": runner.reader_pipeline.sha256(lexical),
                    }
                }
            },
        )
        _write_jsonl(gnn, gnn_rows)
        gnn_freeze = gnn.parent / "prediction_freeze_manifest.json"
        _write_json(
            gnn_freeze,
            {
                "status": "predictions_frozen_gold_unopened",
                "prediction_freeze_sha256": runner.reader_pipeline.sha256(gnn),
            },
        )
        _write_jsonl(support_gold, support_rows)

        def context_loader(_source_root, metadata):
            assert runner.FORBIDDEN_INPUT_FIELDS.isdisjoint(
                set().union(*(set(row) for row in metadata.values()))
            )
            return {key: list(contexts[key]) for key in metadata}, {"fixture": {"gold_fields_decoded": []}}

        yield {
            "work": work,
            "lexical": lexical,
            "gnn": gnn,
            "gnn_freeze": gnn_freeze,
            "candidate_root": candidate_root,
            "output": output,
            "support_gold": support_gold,
            "protected_manifest": protected_manifest,
            "protected_file": protected_file,
            "protected_before": protected_file.read_bytes(),
            "context_loader": context_loader,
            "examples": examples,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _freeze(fixture):
    return runner.freeze_inputs_phase(
        fixture["lexical"],
        fixture["gnn"],
        fixture["gnn_freeze"],
        fixture["candidate_root"],
        fixture["work"],
        fixture["output"],
        full_context_loader=fixture["context_loader"],
        protected_manifests=(fixture["protected_manifest"],),
    )


def _text_reader(question, support, model):
    assert model == runner.MODEL_NAME
    return {
        "predicted_answer": "yes",
        "explanation": "fixture",
        "raw_response": "{}",
        "answer_source": "fixture_reader",
    }


def _ontology_reader(metadata, support, model):
    assert runner.FORBIDDEN_INPUT_FIELDS.isdisjoint(metadata)
    return {
        "predicted_answer": "TRUE",
        "explanation": "fixture",
        "raw_response": "{}",
        "answer_source": "fixture_reader",
    }


def _generate(fixture):
    return runner.generate_phase(
        fixture["output"],
        workers=2,
        text_reader=_text_reader,
        ontology_reader=_ontology_reader,
        protected_manifests=(fixture["protected_manifest"],),
    )


def test_freeze_reuses_exact_k1_and_contains_no_adaptive_or_k3(baseline_fixture):
    freeze = _freeze(baseline_fixture)
    rows = runner.reader_pipeline.load_jsonl(
        baseline_fixture["output"] / "frozen_reader_inputs.jsonl"
    )
    assert freeze["configurations"] == [
        {"method": method, "setting": setting} for method, setting in runner.CONFIGURATIONS
    ]
    assert freeze["adaptive_k_used"] is False
    assert len(rows) == len(baseline_fixture["examples"]) * 3
    assert {(row["method"], row["setting"]) for row in rows} == set(runner.CONFIGURATIONS)
    assert all(row["setting"] not in {"k3", "adaptive"} for row in rows)
    for item in baseline_fixture["examples"]:
        selected = {
            (row["method"], row["setting"]): row["selected_support"]
            for row in rows
            if row["example_id"] == item["example_id"]
        }
        assert selected[("lexical_subgraph", "k1")] == [item["unit"]]
        assert selected[("gnn_rag", "k1")] == [item["unit"]]
        assert selected[("full_context", "full")] == item["full"]
        assert all("MUST NOT BE USED" not in unit for units in selected.values() for unit in units)


def test_generate_reads_only_frozen_inputs_and_preserves_protected_artifacts(baseline_fixture):
    _freeze(baseline_fixture)
    # Generation has no source/candidate/retrieval arguments and succeeds after they disappear.
    shutil.rmtree(baseline_fixture["candidate_root"])
    baseline_fixture["lexical"].unlink()
    baseline_fixture["gnn"].unlink()
    freeze = _generate(baseline_fixture)
    assert freeze["prediction_count"] == len(baseline_fixture["examples"]) * 3
    assert freeze["gold_answers_opened"] is False
    assert baseline_fixture["protected_file"].read_bytes() == baseline_fixture["protected_before"]


def test_evaluation_opens_gold_only_after_prediction_hash_freeze_and_full_context_is_answer_only(
    baseline_fixture,
):
    _freeze(baseline_fixture)
    _generate(baseline_fixture)
    accesses = []

    def gold_loader(dataset, source, ids):
        accesses.append(("answer", dataset, set(ids)))
        value = "TRUE" if runner.reader_pipeline.DATASET_INFO[dataset][1] == "ontology" else "yes"
        return {example_id: value for example_id in ids}

    def support_loader(path, ids):
        accesses.append(("support", str(path), set(ids)))
        by_id = {item["example_id"]: [[item["unit"]]] for item in baseline_fixture["examples"]}
        return {example_id: by_id[example_id] for example_id in ids}

    metrics = runner.evaluate_phase(
        baseline_fixture["output"],
        baseline_fixture["support_gold"],
        source_root=baseline_fixture["work"],
        gold_loader=gold_loader,
        support_gold_loader=support_loader,
        protected_manifests=(baseline_fixture["protected_manifest"],),
    )
    assert accesses
    for dataset_metrics in metrics["by_dataset"].values():
        assert set(dataset_metrics["full_context"]) == {"answer_examples", "answer_em", "answer_f1"}
        for method in ("lexical_subgraph", "gnn_rag"):
            assert {"support_em", "support_f1", "joint_em", "joint_f1"} <= set(
                dataset_metrics[method]
            )
    details = runner.reader_pipeline.load_jsonl(
        baseline_fixture["output"] / "per_example_end_to_end.jsonl"
    )
    full_rows = [row for row in details if row["method"] == "full_context"]
    assert all(not any(key.startswith(("support_", "joint_")) for key in row) for row in full_rows)
    assert (baseline_fixture["output"] / "summary.md").is_file()
    assert (baseline_fixture["output"] / "artifact_manifest.json").is_file()
    assert baseline_fixture["protected_file"].read_bytes() == baseline_fixture["protected_before"]


def test_prediction_tamper_blocks_all_gold_access(baseline_fixture):
    _freeze(baseline_fixture)
    _generate(baseline_fixture)
    with (baseline_fixture["output"] / "predictions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    accesses = []
    with pytest.raises(ValueError, match="changed after the generation freeze"):
        runner.evaluate_phase(
            baseline_fixture["output"],
            baseline_fixture["support_gold"],
            gold_loader=lambda *args: accesses.append(args),
            support_gold_loader=lambda *args: accesses.append(args),
            protected_manifests=(baseline_fixture["protected_manifest"],),
        )
    assert accesses == []


def test_selective_ontology_context_extractor_never_decodes_qa_answers(baseline_fixture):
    source = baseline_fixture["work"] / "ontology.json"
    first = [
        {
            "OWL Context": "context one",
            "QAs": [{"NL Question": "q", "Answer": "SECRET GOLD A", "Explanations": ["gold"]}],
        },
        {
            "OWL Context": "context two",
            "QAs": [{"NL Question": "q2", "Answer": "SECRET GOLD B"}],
        },
    ]
    _write_json(source, first)
    before = runner.extract_top_level_string_field(source, "OWL Context")
    first[0]["QAs"][0]["Answer"] = "MUTATED ANSWER"
    first[0]["QAs"][0]["Explanations"] = ["MUTATED SUPPORT"]
    _write_json(source, first)
    after = runner.extract_top_level_string_field(source, "OWL Context")
    assert before == after == ["context one", "context two"]


def test_cli_declares_exactly_three_answer_generation_configurations():
    assert runner.CONFIGURATIONS == (
        ("lexical_subgraph", "k1"),
        ("gnn_rag", "k1"),
        ("full_context", "full"),
    )
    args = runner.build_parser().parse_args(["generate"])
    assert args.workers == 8
