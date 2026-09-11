import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.symbolic_composer import (
    SymbolicComposer,
    extract_query_signature,
    parse_axiom,
)


FEATURE_NAMES = [
    "connectivity_bonus",
    "fact_rule_mix_bonus",
    "query_alignment_bonus",
    "bridge_bonus",
    "redundancy_penalty",
]


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ranking_target(row: Dict) -> float:
    if row.get("exact_match_any_gold", False):
        return 1.0
    if row.get("contains_any_gold_explanation", False):
        return 0.9

    partial_f1 = float(row.get("best_set_f1_to_gold", 0.0))
    if partial_f1 > 0:
        return min(0.3, 0.5 * partial_f1)

    return 0.0


def group_rows_by_example(rows: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["example_id"]), []).append(row)
    return grouped


def composer_features(row: Dict, composer: SymbolicComposer) -> List[float]:
    units = row.get("subgraph_units", []) or []
    if not units:
        return [0.0] * len(FEATURE_NAMES)

    nodes = []
    for idx, unit in enumerate(units):
        nodes.append(
            {
                "node_id": f"n{idx}",
                "raw_axiom": unit,
                "candidate_kind": "learned_weight_fit",
                "score": 0.0,
                "label": int(row.get("label", 0)),
                "parsed_obj": parse_axiom(unit),
            }
        )

    edges = composer._build_graph(nodes)
    node_ids = {node["node_id"] for node in nodes}

    qsig = extract_query_signature(
        question=str(row.get("question") or ""),
        sparql_query=str(row.get("sparql_query") or ""),
    )
    query_entities = set(qsig["query_entities"])
    query_properties = set(qsig["query_properties"])

    connected = 1.0 if composer._is_connected(node_ids, edges) else 0.0
    mix = 1.0 if composer._fact_rule_mix(nodes) else 0.0
    q_align = composer._query_alignment(nodes, query_entities, query_properties)
    bridge = composer._bridge_score(nodes, query_entities, query_properties)
    redundancy = composer._redundancy(nodes)

    return [connected, mix, q_align, bridge, -redundancy]


def dot(weights: List[float], features: List[float]) -> float:
    return sum(w * x for w, x in zip(weights, features))


def fit_weights(
    rows: List[Dict],
    epochs: int,
    lr: float,
    margin: float,
    max_pairs_per_example: int,
    seed: int,
    initial_weights: List[float] | None = None,
) -> Tuple[List[float], Dict]:
    rng = random.Random(seed)
    composer = SymbolicComposer()
    grouped = group_rows_by_example(rows)

    weights = list(initial_weights) if initial_weights is not None else [0.0] * 5
    examples = []

    for example_rows in grouped.values():
        prepared = []
        for row in example_rows:
            prepared.append(
                {
                    "target": ranking_target(row),
                    "features": composer_features(row, composer),
                }
            )

        positives = [r for r in prepared if r["target"] >= 0.9]
        negatives = [r for r in prepared if r["target"] < 0.9]
        if positives and negatives:
            examples.append((positives, negatives))

    updates = 0
    violations = 0

    for _ in range(max(1, epochs)):
        rng.shuffle(examples)
        for positives, negatives in examples:
            pairs = [
                (pos, neg)
                for pos in positives
                for neg in negatives
                if pos["target"] > neg["target"]
            ]
            rng.shuffle(pairs)
            for pos, neg in pairs[:max_pairs_per_example]:
                pos_score = dot(weights, pos["features"])
                neg_score = dot(weights, neg["features"])
                if pos_score <= neg_score + margin:
                    violations += 1
                    for i in range(len(weights)):
                        weights[i] += lr * (pos["features"][i] - neg["features"][i])
                        weights[i] = max(0.0, weights[i])
                    updates += 1

    diagnostics = {
        "examples": len(examples),
        "updates": updates,
        "violations": violations,
        "epochs": epochs,
        "lr": lr,
        "margin": margin,
    }
    return weights, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit lightweight SymbolicComposer weights from support rows."
    )
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--output", default="checkpoints/symbolic_composer_weights.json")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--max-pairs-per-example", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--initial-weights",
        default="",
        help=(
            "Optional comma-separated initial weights in FEATURE_NAMES order. "
            "Defaults to all zeros so coefficients are learned from data."
        ),
    )
    parser.add_argument(
        "--initial-weights-json",
        default="",
        help="Optional JSON file containing existing SymbolicComposer weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.train_path)

    initial_weights = None
    if args.initial_weights_json:
        with open(args.initial_weights_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        initial_weights = [float(loaded.get(name, 0.0)) for name in FEATURE_NAMES]
    elif args.initial_weights:
        initial_weights = [
            float(value.strip()) for value in args.initial_weights.split(",") if value.strip()
        ]
        if len(initial_weights) != len(FEATURE_NAMES):
            raise ValueError(f"--initial-weights must contain {len(FEATURE_NAMES)} values")

    weights, diagnostics = fit_weights(
        rows=rows,
        epochs=args.epochs,
        lr=args.lr,
        margin=args.margin,
        max_pairs_per_example=args.max_pairs_per_example,
        seed=args.seed,
        initial_weights=initial_weights,
    )

    output = {name: round(value, 6) for name, value in zip(FEATURE_NAMES, weights)}
    output["diagnostics"] = diagnostics

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
