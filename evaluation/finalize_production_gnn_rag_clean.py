"""Verify and freeze the final clean TRAIN/DEV GNN-RAG run report."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.prepare_production_gnn_rag_clean import DATASETS, sha256_file, write_json


CODE_FILES = (
    "data_processing/prepare_production_gnn_rag_clean.py",
    "training/train_production_gnn_rag_clean.py",
    "evaluation/finalize_production_gnn_rag_clean.py",
    "third_party/GNN-RAG/gnn/parsing.py",
    "third_party/GNN-RAG/gnn/dataset_load.py",
    "third_party/GNN-RAG/gnn/evaluate.py",
    "third_party/GNN-RAG/gnn/train_model.py",
    "third_party/GNN-RAG/gnn/modules/question_encoding/tokenizers.py",
    "third_party/GNN-RAG/gnn/models/ReaRev/rearev.py",
    "tests/test_prepare_production_gnn_rag_clean.py",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_root = root / "checkpoints" / "production_generator_d_v1_gnn_rag_clean"
    rows = []
    for dataset in DATASETS:
        dataset_dir = output_root / dataset
        summary = json.loads((dataset_dir / "adapter_summary.json").read_text(encoding="utf-8"))
        frozen = json.loads((dataset_dir / "frozen_checkpoint_manifest.json").read_text(encoding="utf-8"))
        invariance = json.loads((dataset_dir / "leakage_invariance.json").read_text(encoding="utf-8"))
        command = json.loads((dataset_dir / "training_command.json").read_text(encoding="utf-8"))
        log = (dataset_dir / "training.log").read_text(encoding="utf-8")
        checkpoint = dataset_dir / frozen["selected_checkpoint"]
        checks = {
            "checkpoint_hash_matches": sha256_file(checkpoint) == frozen["selected_checkpoint_sha256"],
            "relation_hash_matches": sha256_file(dataset_dir / "adapter" / "relations.txt") == frozen["relation_vocabulary_sha256"],
            "three_dev_epochs": [item["epoch"] for item in frozen["dev_metrics"]] == [1, 2, 3],
            "invariance_pass": invariance.get("pass") is True and all(item.get("pass") for item in invariance["checks"]),
            "no_test_adapter": not (dataset_dir / "adapter" / "test.json").exists(),
            "train_dev_only_flag": "--train_dev_only" in command and command[command.index("--train_dev_only") + 1].lower() == "true",
            "no_legacy_load_argument": "--load_experiment" not in command,
            "no_test_eval_argument": "--is_eval" not in command,
            "no_test_metric_log": "TEST F1" not in log,
            "sealed_completion_log": "TEST remained sealed" in log,
        }
        if not all(checks.values()):
            raise RuntimeError(f"Final freeze audit failed for {dataset}: {checks}")
        rows.append(
            {
                "dataset": dataset,
                "clean_train_count": summary["train_count"],
                "clean_dev_count": summary["dev_count"],
                "relation_vocabulary_size": summary["relation_vocabulary_size"],
                "relation_vocabulary_sha256": summary["relation_vocabulary_sha256"],
                "leakage_invariance": "PASS",
                "dev_metrics": frozen["dev_metrics"],
                "selected_checkpoint": frozen["selected_checkpoint"],
                "checkpoint_sha256": frozen["selected_checkpoint_sha256"],
                "checks": checks,
            }
        )

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "--", *CODE_FILES], cwd=root)
    code_state = {
        "base_git_commit": head,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "note": "Run code includes the listed working-tree files; use their hashes in addition to the base commit.",
        "files": [
            {"path": path, "sha256": sha256_file(root / path)}
            for path in CODE_FILES
        ],
    }
    write_json(output_root / "code_state.json", code_state)
    report = {
        "protocol": "production_generator_d_v1_gnn_rag_clean_v1",
        "checkpoint_selection_rule": "final epoch 3, preserving the historical production runner's explicit final-checkpoint semantics",
        "datasets": rows,
        "global_confirmations": {
            "test_inference_occurred": False,
            "test_metric_inspected": False,
            "production_generator_d_v1_test_file_opened": False,
            "test_used_for_vocabulary_training_selection_or_debugging": False,
            "legacy_checkpoint_reused": False,
            "two_wiki_evidences_used": False,
            "answer_or_explanation_changed_graph_topology": False,
            "mentions_page_has_distinct_vocabulary_entry": True,
            "answer_generation_occurred": False,
            "strict_operator_test_gold_display_seal_satisfied": False,
            "operator_seal_exception": "An initial broad repository source search emitted legacy non-production TEST answer rows. They were not used by adapter construction, vocabulary, mapping, debugging, training, validation, or checkpoint selection.",
        },
        "code_state_file": "code_state.json",
    }
    write_json(output_root / "final_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
