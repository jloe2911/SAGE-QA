"""
Run the type-aware graph-retrieval pipeline (data_processing/graph_retrieval.py)
against a real 2WikiMultiHopQA question and print every stage: KG construction,
question-type classification, relation-chain parsing, anchor extraction, typed
traversal (or the fallback ladder), and the final answer vs. gold.

Usage:
    python3 examples/run_graph_retrieval_pipeline.py --question-contains "Ghost Fever"
    python3 examples/run_graph_retrieval_pipeline.py --row-id 42
    python3 examples/run_graph_retrieval_pipeline.py --question-contains "Metamathics" \
        --split validation

By default the question's own `type` field (compositional / comparison /
bridge_comparison / inference) selects which retrieve_* function runs;
pass --use-classifier to route with classify_question_type() instead, e.g.
to see where the lightweight surface classifier disagrees with the gold type.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.build_2wiki_subgraph_dataset import flatten_context
from data_processing.graph_retrieval import (
    classify_question_type,
    extract_anchor_entities,
    parse_relation_chain,
    retrieve_bridge_comparison,
    retrieve_comparison,
    retrieve_compositional,
    retrieve_inference,
)
from data_processing.text_kg_constructor import KGConstructionConfig, construct_text_kg

RETRIEVE_FN = {
    "compositional": retrieve_compositional,
    "comparison": retrieve_comparison,
    "bridge_comparison": retrieve_bridge_comparison,
    "inference": retrieve_inference,
}

RAW_DATA = {
    "train": "data/raw/2WikiMultihopQA/data/train-00000-of-00002.parquet",
    "validation": "data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet",
}


def load_row(split: str, question_contains: str | None, row_id: int | None) -> pd.Series:
    repo_root = Path(__file__).resolve().parents[1]
    df = pd.read_parquet(repo_root / RAW_DATA[split])
    if row_id is not None:
        return df.iloc[row_id]
    if question_contains:
        matches = df[df["question"].str.contains(question_contains, na=False, regex=False)]
        if matches.empty:
            raise SystemExit(f"No question containing {question_contains!r} found in {split} split.")
        return matches.iloc[0]
    raise SystemExit("Pass --question-contains or --row-id.")


def run(row: pd.Series, use_classifier: bool) -> None:
    question = row["question"]
    gold_type = row["type"]

    print("=" * 100)
    print(f"Question: {question}")
    print(f"Gold type: {gold_type}")
    print(f"Gold answer: {row['answer']}")
    print(f"Gold evidences: {list(row['evidences'])}")
    print("-" * 100)

    pairs = [[t, list(s)] for t, s in zip(row["context"]["title"], row["context"]["sentences"])]
    sentence_records = flatten_context({"context": pairs})
    context_titles = list(row["context"]["title"])

    config = KGConstructionConfig(backend="deterministic", max_triples=64)
    kg_triples, method = construct_text_kg(
        provided_evidences=[],
        sentence_records=sentence_records,
        question=question,
        config=config,
        llm_constructor=None,
    )
    typed = [t for t in kg_triples if t[1] != "mentions_page"]
    print(f"KG construction method: {method}")
    print(f"  {len(typed)} typed triples, {len(kg_triples) - len(typed)} generic bridge triples")

    predicted_type = classify_question_type(question)
    print(f"classify_question_type() -> {predicted_type}" + (" (mismatch with gold)" if predicted_type != gold_type else ""))

    routing_type = predicted_type if use_classifier else gold_type
    print(f"Routing on: {'classifier' if use_classifier else 'gold'} type -> {routing_type}")

    chain = parse_relation_chain(question)
    anchors = extract_anchor_entities(question, context_titles)
    print(f"parse_relation_chain() -> {chain}")
    print(f"extract_anchor_entities() -> {anchors}")
    print("-" * 100)

    retrieve_fn = RETRIEVE_FN[routing_type]
    result = retrieve_fn(question, kg_triples, sentence_records, context_titles)

    if result.get("type") == "compositional":
        print(f"status: {result['status']}")
        for e in result["evidence"]:
            print(f"  {e}")
        if result.get("fallback"):
            print(f"  fallback tier used: {result['fallback']['tier']}")
        print(f"Predicted answer: {result['answer']}")

    elif result.get("type") in ("bridge_comparison", "comparison"):
        for anchor, value, evidence in zip(result["anchors"], result["branch_values"], result["branch_evidence"]):
            print(f"Branch [{anchor}] -> {value}")
            for e in evidence:
                print(f"  {e}")
        print(f"Comparator: {result.get('comparator')}")
        print(f"Predicted answer: {result.get('answer')}  (status: {result.get('status')})")

    elif result.get("type") == "inference":
        for i, hop in enumerate(result["hops"], start=1):
            print(f"Hop {i} [{hop['tier']}]:")
            for e in hop["evidence"]:
                print(f"  {e if isinstance(e, tuple) else e.get('sentence')}")
        print(f"Answer hint: {result['answer_hint']}")

    else:
        print("Result:", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question-contains", type=str, default=None, help="Substring to find a question by.")
    parser.add_argument("--row-id", type=int, default=None, help="Row index to pick directly instead of searching.")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument(
        "--use-classifier",
        action="store_true",
        help="Route with classify_question_type() instead of the gold type field.",
    )
    args = parser.parse_args()

    row = load_row(args.split, args.question_contains, args.row_id)
    run(row, args.use_classifier)


if __name__ == "__main__":
    main()
