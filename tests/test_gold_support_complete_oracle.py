from generation import run_gold_support_complete_oracle as oracle


def test_ordered_union_uses_all_alternatives_and_deduplicates():
    assert oracle._ordered_union([["a", "b"], ["b", "c"], ["a", "d"]]) == [
        "a", "b", "c", "d"
    ]


def test_corrected_cache_contract_contains_every_inference_field():
    row = {
        "dataset": "OWL2Bench_1hop",
        "example_id": "e1",
        "question": "q",
        "support_sha256": "support",
        "domain": "ontology",
        "metadata": {
            "question": "q",
            "task_type": "boolean",
            "sparql_query": "ASK {}",
            "answer_type": "boolean",
        },
    }
    contract = oracle._cache_contract(row)
    assert tuple(contract) == oracle.CACHE_FIELDS
    assert contract["reader_type"] == "ontology_proof_first_reader_v1"
    assert contract["task_type"] == "boolean"
    assert contract["sparql_query"] == "ASK {}"
    assert contract["answer_type"] == "boolean"


def test_cache_key_changes_with_task_metadata():
    base = {
        "dataset": "OWL2Bench_1hop",
        "example_id": "e1",
        "question": "q",
        "support_sha256": "support",
        "domain": "ontology",
        "metadata": {"task_type": "a", "sparql_query": "ASK {}", "answer_type": "boolean"},
    }
    changed = {**base, "metadata": {**base["metadata"], "task_type": "b"}}
    assert oracle._cache_key(base) != oracle._cache_key(changed)
