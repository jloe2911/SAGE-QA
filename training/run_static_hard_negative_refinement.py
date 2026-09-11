"""Frozen one-pass, head-only static hard-negative refinement.

TRAIN is streamed one example at a time.  Candidate metadata for the current
example is used to select the already-audited pair; only those two candidates
are materialized and scored.  The full local evidence graph is reconstructed
because it is an input to the frozen production representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.gnn_subgraph_retriever import (
    GNNSubgraphRetriever,
    build_graph_inputs_for_example,
    compute_subgraph_symbolic_features,
)
from training.train_gnn_subgraph_retriever import (
    _is_text_dataset,
    binary_label_from_target,
    compute_example_loss,
    gold_explanations_from_row,
    gold_support_units_from_row,
    ranking_target,
    reconstruct_candidate_axioms,
    select_reserved_hard_pair,
)
from utils.eval_splits import infer_answer_type, infer_dataset_name, infer_hop


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
DATASETS = (
    ("HotpotQA", "hotpotqa", 2378),
    ("2WikiMultiHopQA", "2wiki", 2278),
    ("FamilyOWL_1hop", "familyowl_1hop", 772),
    ("FamilyOWL_2hop", "familyowl_2hop", 304),
    ("pizza_100_1hop", "pizza_100_1hop", 183),
    ("pizza_100_2hop", "pizza_100_2hop", 108),
    ("pizza_250_1hop", "pizza_250_1hop", 236),
    ("pizza_250_2hop", "pizza_250_2hop", 117),
    ("OWL2Bench_1hop", "OWL2Bench_1hop", 623),
    ("OWL2Bench_2hop", "OWL2Bench_2hop", 605),
)
FROZEN = {
    "learning_rate": 2e-5,
    "ranking_margin": 0.2,
    "ranking_weight": 2.0,
    "bce_weight": 0.2,
    "listwise_weight": 0.0,
    "max_pairs": 512,
    "max_length": 128,
    "seed": SEED,
    "passes": 1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def grouped_jsonl(path: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    """Yield one contiguous example group; never retain the corpus."""
    seen: set[str] = set()
    current_id: str | None = None
    current_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            example_id = str(row.get("example_id") or "")
            if not example_id:
                raise ValueError(f"{path}:{line_number}: missing example_id")
            if current_id is None:
                current_id = example_id
            if example_id != current_id:
                if example_id in seen:
                    raise ValueError(f"{path}: non-contiguous example {example_id!r}")
                seen.add(current_id)
                yield current_id, current_rows
                current_id, current_rows = example_id, []
            current_rows.append(row)
    if current_id is not None:
        yield current_id, current_rows


def row_for_selection(row: dict[str, Any], order: int) -> dict[str, Any]:
    """Minimal metadata view accepted by the audited selector."""
    return {
        "_raw": row,
        "rank_target": ranking_target(row),
        "candidate_pre_rank_score": float(row.get("candidate_pre_rank_score", 0.0)),
        "generation_rank": int(row.get("generation_rank", order)),
        "materialization_order": order,
        "subgraph_units": list(row.get("subgraph_units", []) or []),
    }


def materialize_selected_row(
    raw: dict[str, Any], *, order: int, axiom_to_idx: dict[str, int]
) -> dict[str, Any]:
    units = list(raw.get("subgraph_units", []) or [])
    node_ids = [axiom_to_idx[unit] for unit in units if unit in axiom_to_idx]
    if not node_ids:
        raise ValueError("Selected candidate has no nodes in its production local graph")
    question = str(
        raw.get("question")
        or raw.get("abs_question")
        or raw.get("task_id")
        or raw.get("sparql_query")
        or raw["example_id"]
    )
    sparql = str(raw.get("sparql_query") or "")
    target = ranking_target(raw)
    if _is_text_dataset(raw):
        symbolic = list(raw.get("symbolic_features", []) or [])
        if len(symbolic) != 8:
            symbolic = [0.0] * 8
        bridge = list(raw.get("graph_context_units", []) or [])
    else:
        symbolic = compute_subgraph_symbolic_features(
            question=question,
            sparql_query=sparql,
            subgraph_units=units,
            use_gold_features=False,
            exact_match_any_gold=raw.get("exact_match_any_gold", False),
            contains_any_gold_explanation=raw.get("contains_any_gold_explanation", False),
        )
        bridge = list(raw.get("graph_context_units", []) or raw.get("kg_evidence_units", []) or [])
    return {
        "example_id": str(raw["example_id"]),
        "question": question,
        "sparql_query": sparql,
        "subgraph_units": units,
        "subgraph_node_ids": node_ids,
        "subgraph_size": len(units),
        "graph_context_units": bridge,
        "symbolic_features": symbolic,
        "rank_target": target,
        "label": binary_label_from_target(target),
        "gold_support_units": gold_support_units_from_row(raw),
        "gold_explanations": gold_explanations_from_row(raw),
        "best_set_f1_to_gold": float(raw.get("best_set_f1_to_gold", 0.0)),
        "exact_match_any_gold": bool(raw.get("exact_match_any_gold", False)),
        "contains_any_gold_explanation": bool(raw.get("contains_any_gold_explanation", False)),
        "candidate_pre_rank_score": float(raw.get("candidate_pre_rank_score", 0.0)),
        "generation_rank": int(raw.get("generation_rank", order)),
        "materialization_order": order,
    }


def prepare_static_pair(raw_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    views = [row_for_selection(row, order) for order, row in enumerate(raw_rows)]
    selected = select_reserved_hard_pair(views)
    if selected is None:
        return None
    positive_view, negative_view = selected
    first = raw_rows[0]
    if first.get("gold_available_during_candidate_generation") is True:
        raise ValueError("Candidate artifact reports gold access during generation")
    candidate_axioms = reconstruct_candidate_axioms(raw_rows)
    axiom_to_idx = {axiom: index for index, axiom in enumerate(candidate_axioms)}
    rows = [
        materialize_selected_row(
            view["_raw"], order=int(view["materialization_order"]), axiom_to_idx=axiom_to_idx
        )
        for view in selected
    ]
    example_id = str(first["example_id"])
    question = str(
        first.get("question")
        or first.get("abs_question")
        or first.get("task_id")
        or first.get("sparql_query")
        or example_id
    )
    return {
        "example_id": example_id,
        "dataset": infer_dataset_name(example_id, first),
        "hop": infer_hop(example_id, first),
        "answer_type": infer_answer_type(example_id, first),
        "question": question,
        "sparql_query": str(first.get("sparql_query") or ""),
        "candidate_axioms": candidate_axioms,
        "candidate_rows": rows,
    }


def freeze_except_classifier(model: GNNSubgraphRetriever) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    model.eval()
    model.classifier.train()


def frozen_pair_representations(model, tokenizer, example, device) -> torch.Tensor:
    """Build two fixed representations; no classifier work occurs here."""
    with torch.no_grad():
        inputs = build_graph_inputs_for_example(
            candidate_axioms=example["candidate_axioms"],
            question=example["question"],
            sparql_query=example["sparql_query"],
            tokenizer=tokenizer,
            model_device=device,
            max_length=FROZEN["max_length"],
        )
        query = model.encode_texts(
            inputs["query_input_ids"],
            inputs["query_attention_mask"],
            inputs["query_token_type_ids"],
        ).squeeze(0)
        node_text = model.encode_texts(
            inputs["node_input_ids"], inputs["node_attention_mask"], inputs["node_token_type_ids"]
        )
        nodes = model.encode_graph(
            node_text, inputs["node_symbolic_features"], inputs["edge_index"]
        )
        representations = []
        for row in example["candidate_rows"]:
            ids = torch.tensor(row["subgraph_node_ids"], dtype=torch.long, device=device)
            pooled = model.pool_subgraph(nodes, ids)
            symbolic = torch.tensor(row["symbolic_features"], dtype=torch.float, device=device)
            symbolic_repr = model.subgraph_feature_projection(symbolic.unsqueeze(0)).squeeze(0)
            representations.append(torch.cat([query, pooled, symbolic_repr], dim=-1))
    return torch.stack(representations)


def load_v1(checkpoint_path: Path, device):
    from transformers import AutoTokenizer

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GNNSubgraphRetriever(
        model_name=model_name,
        node_symbolic_dim=int(checkpoint.get("node_symbolic_dim", 8)),
        subgraph_symbolic_dim=int(checkpoint.get("subgraph_symbolic_dim", 8)),
        gnn_hidden_dim=int(checkpoint.get("gnn_hidden_dim", 128)),
        gnn_layers=int(checkpoint.get("gnn_layers", 2)),
        classifier_hidden_dim=int(checkpoint.get("classifier_hidden_dim", 128)),
        dropout=0.1,
        freeze_encoder=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    freeze_except_classifier(model)
    return checkpoint, tokenizer, model


def frozen_state(model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("classifier.")
    }


def train_dataset(
    train_path: Path, v1_path: Path, output_path: Path, expected_eligible: int
) -> dict[str, Any]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, tokenizer, model = load_v1(v1_path, device)
    before = frozen_state(model)
    optimizer = AdamW(model.classifier.parameters(), lr=FROZEN["learning_rate"])
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, expected_eligible // 10),
        num_training_steps=expected_eligible,
    )
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    totals = {"examples": 0, "eligible": 0, "raw_candidate_rows": 0, "scored_candidate_rows": 0}
    loss_sums = {"total_loss": 0.0, "ranking_loss": 0.0, "bce_loss": 0.0, "listwise_loss": 0.0}
    for _, raw_rows in tqdm(grouped_jsonl(train_path), total=None, desc=train_path.parent.name):
        totals["examples"] += 1
        totals["raw_candidate_rows"] += len(raw_rows)
        example = prepare_static_pair(raw_rows)
        if example is None:
            continue
        totals["eligible"] += 1
        optimizer.zero_grad()
        representations = frozen_pair_representations(model, tokenizer, example, device)
        scores = model.classifier(representations).squeeze(-1)
        loss, stats = compute_example_loss(
            scores,
            example["candidate_rows"],
            bce,
            ranking_margin=FROZEN["ranking_margin"],
            ranking_weight=FROZEN["ranking_weight"],
            bce_weight=FROZEN["bce_weight"],
            listwise_weight=FROZEN["listwise_weight"],
            max_pairs=FROZEN["max_pairs"],
            hard_pair_reservation=False,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        totals["scored_candidate_rows"] += 2
        for key in loss_sums:
            loss_sums[key] += stats[key]
        del example, representations, scores, loss
    if totals["eligible"] != expected_eligible:
        raise AssertionError(
            f"Eligible count {totals['eligible']} != frozen audit {expected_eligible}"
        )
    for name, value in model.state_dict().items():
        if not name.startswith("classifier.") and not torch.equal(
            value.detach().cpu(), before[name]
        ):
            raise AssertionError(f"Frozen parameter or buffer changed: {name}")
    result = {
        **totals,
        "mean_losses": {key: value / expected_eligible for key, value in loss_sums.items()},
        "device": str(device),
        "v1_checkpoint": str(v1_path),
        "v1_checkpoint_sha256": sha256(v1_path),
        "only_classifier_trainable": True,
        "frozen_state_unchanged": True,
    }
    refined = dict(checkpoint)
    refined["model_state_dict"] = model.state_dict()
    refined.update(
        {
            "refinement": "one_pass_static_generator_d_hard_pair_head_only",
            "refinement_hyperparameters": FROZEN,
            "training_examples": totals["eligible"],
            "source_checkpoint_sha256": result["v1_checkpoint_sha256"],
            "dev_used_for_training_or_selection": False,
            "test_accessed": False,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=False)
    torch.save(refined, output_path)
    result["checkpoint"] = str(output_path)
    result["checkpoint_sha256"] = sha256(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/production_generator_d_v1")
    parser.add_argument(
        "--v1-root", type=Path, default=ROOT / "checkpoints/production_generator_d_v1"
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "checkpoints/production_generator_d_static_hard_v1",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "outputs/final_model_development/production_generator_d_static_hard_v1",
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=ROOT
        / "outputs/final_model_development/production_generator_d_static_hard_v1_gate/implementation_invariance_gate.json",
    )
    args = parser.parse_args()
    for name in ("data_root", "v1_root", "checkpoint_root", "run_root", "gate"):
        setattr(args, name, getattr(args, name).resolve())
    if args.checkpoint_root.exists() or args.run_root.exists():
        raise FileExistsError("Refusing to resume or overwrite the frozen final refinement")
    implementation_path = Path(__file__).resolve()
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    if gate.get("status") != "passed" or gate.get("implementation_sha256") != sha256(
        implementation_path
    ):
        raise RuntimeError(
            "The static refinement implementation/invariance gate is missing or stale"
        )
    manifest = {
        "schema_version": "sageqa_static_hard_refinement_freeze_v1",
        "status": "frozen_before_training",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": {
            "path": str(implementation_path.relative_to(ROOT)),
            "sha256": sha256(implementation_path),
        },
        "gate": {"path": str(args.gate.relative_to(ROOT)), "sha256": sha256(args.gate)},
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).splitlines(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "dataset_order": [name for name, _, _ in DATASETS],
        "hyperparameters": FROZEN,
        "streaming_contract": {
            "corpus_materialized": False,
            "prepared_candidates_per_eligible_example": 2,
            "gnn_forwards_per_eligible_example": 1,
        },
        "dev_accessed_before_all_training_complete": False,
        "test_accessed": False,
        "answer_generation_run": False,
    }
    write_json(args.run_root / "frozen_protocol.json", manifest)
    frozen_hash = manifest["implementation"]["sha256"]
    results = []
    env = os.environ
    for dataset, slug, expected in DATASETS:
        if sha256(implementation_path) != frozen_hash:
            raise RuntimeError("Frozen implementation changed after training began")
        results.append(
            train_dataset(
                args.data_root / dataset / "train_subgraph_retrieval.jsonl",
                args.v1_root / slug / "best_model.pt",
                args.checkpoint_root / slug / "best_model.pt",
                expected,
            )
        )
        write_json(
            args.run_root / "training_progress.json",
            {"completed": len(results), "datasets": results, "test_accessed": False},
        )
    write_json(
        args.run_root / "all_training_complete.json",
        {
            "schema_version": "sageqa_static_hard_refinement_completion_v1",
            "status": "all_ten_complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "datasets": results,
            "implementation_hash_unchanged": sha256(implementation_path) == frozen_hash,
            "dev_accessed": False,
            "test_accessed": False,
            "answer_generation_run": False,
        },
    )


if __name__ == "__main__":
    main()
