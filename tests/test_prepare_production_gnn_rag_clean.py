from data_processing.prepare_production_gnn_rag_clean import (
    attach_supervision,
    inference_sample,
    invariance_check,
)


def test_text_graph_ignores_gold_and_uses_mentions_page():
    rows = [{
        "example_id": "x",
        "question": "Who?",
        "subgraph_units": ["SENT::Page::0::Sentence"],
        "graph_context_units": [],
        "gold_support_units": ["SENT::Gold::1::Injected"],
        "answer": "secret",
        "evidences": [["secret"]],
    }]
    sample, candidates = inference_sample(rows, "text", 5)
    assert candidates == ["SENT::Page::0::Sentence"]
    assert "mentions_page" in {edge[1] for edge in sample["graph"]}
    assert "mentions" not in {edge[1] for edge in sample["graph"]}
    assert invariance_check(rows, "text", 5)["pass"]
    assert attach_supervision(sample, candidates, rows, "text") == 0
    assert sample["answers"] == []


def test_ontology_absent_gold_stays_absent_and_answer_is_not_fallback():
    rows = [{
        "example_id": "o",
        "question": "Is A related?",
        "sparql_query": "ASK WHERE { <http://x#A_1> <http://x#p> <http://x#B_1> }",
        "subgraph_units": ["A_1 q C_1"],
        "gold_units": ["A_1 p B_1"],
        "gold_explanations": [["A_1 p B_1"]],
        "answer": "TRUE",
    }]
    sample, candidates = inference_sample(rows, "ontology", 5)
    assert invariance_check(rows, "ontology", 5)["pass"]
    assert attach_supervision(sample, candidates, rows, "ontology") == 0
    assert sample["answers"] == []
    assert all("ANSWER::" not in entity for entity in sample["subgraph"]["entities"])
