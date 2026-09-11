import json
import shutil
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluation.adaptive_support_aggregation_v2 import (
    CONTINUOUS_FEATURES,
    FEATURE_ORDER,
    AdaptiveV2Policy,
    adaptive_v2_support_aggregate,
    compute_decision_features,
    load_domain_policy,
)
from evaluation.evaluate_adaptive_support_aggregation_v2 import grouped_oof
from evaluation.fit_production_dev_adaptive_k import fit, load_clean_final_records
from evaluation.run_production_test_retrieval import load_policy_bundle, sha256
from evaluation.evaluate_owl_qa_predictions import get_top_support_units as evaluated_support
from generation.generate_owl_answers_with_llm import get_top_support_units as generated_support


def candidate(score: float, units: list[str], rank: int) -> dict[str, Any]:
    return {"rank": rank, "adjusted_score": score, "subgraph_units": units}


def fitted_policy(probability_direction: int = 1, threshold: float = 0.5):
    x = np.asarray(
        [
            [1.0, 0.9, 0.8, 0.1, 1, 1, 1, 1.0, 0.0, 1],
            [1.0, 0.1, 0.01, 2.0, 5, 5, 0, 0.0, 1.0, 2],
            [1.0, 0.8, 0.7, 0.2, 2, 2, 2, 1.0, 0.0, 3],
            [1.0, 0.2, 0.02, 1.5, 7, 3, 0, 0.0, 1.0, 4],
        ],
        dtype=float,
    )
    y = np.asarray([1, 0, 1, 0] if probability_direction == 1 else [0, 1, 0, 1])
    scaler = StandardScaler().fit(x[:, : len(CONTINUOUS_FEATURES)])
    model_x = np.column_stack(
        [scaler.transform(x[:, : len(CONTINUOUS_FEATURES)]), x[:, -1]]
    )
    model = LogisticRegression(random_state=42, solver="liblinear").fit(model_x, y)
    return AdaptiveV2Policy("text", scaler, model, threshold)


def test_feature_computation_uses_ranked_prefix_and_deduplicated_units():
    candidates = [
        candidate(1.0, ["a", "a", "b"], 1),
        candidate(0.8, ["b", "c", "c"], 2),
        candidate(0.5, ["d"], 3),
    ]
    features = compute_decision_features(candidates, decision_rank=1)

    assert tuple(features) == FEATURE_ORDER
    assert features["cumulative_support_size"] == 2
    assert features["next_candidate_size"] == 2
    assert features["new_evidence_units"] == 1
    assert features["novelty"] == 0.5
    assert features["overlap"] == 0.5
    assert features["decision_rank"] == 1
    assert features["relative_to_top_next"] == pytest.approx(
        np.exp((0.8 - 1.0) / np.std([1.0, 0.8, 0.5]))
    )
    assert features["normalized_consecutive_gap"] == pytest.approx(
        (1.0 - 0.8) / np.std([1.0, 0.8, 0.5])
    )


def test_identical_scores_use_epsilon_scale_without_instability():
    candidates = [candidate(0.5, ["a"], 1), candidate(0.5, ["b"], 2)]
    features = compute_decision_features(candidates, decision_rank=1)

    assert features["relative_to_top_next"] == 1.0
    assert features["normalized_consecutive_gap"] == 0.0


def test_policy_serialization_round_trip_and_explicit_domain_selection():
    text_policy = fitted_policy()
    scaler_buffer = BytesIO()
    model_buffer = BytesIO()
    joblib.dump(text_policy.scaler, scaler_buffer)
    joblib.dump(text_policy.model, model_buffer)
    scaler_buffer.seek(0)
    model_buffer.seek(0)
    loaded = AdaptiveV2Policy(
        "text", joblib.load(scaler_buffer), joblib.load(model_buffer), 0.4
    )
    features = {
        name: float(index + 1) for index, name in enumerate(FEATURE_ORDER)
    }
    assert loaded.continue_probability(features) == pytest.approx(
        AdaptiveV2Policy(
            "text", text_policy.scaler, text_policy.model, 0.4
        ).continue_probability(features)
    )
    with pytest.raises(ValueError, match="domain"):
        load_domain_policy("unused", domain="HotpotQA")


def test_grouped_oof_keeps_all_rows_from_an_example_in_one_validation_fold():
    rows = []
    for example_index in range(12):
        has_continue = example_index < 6
        for rank in (1, 2):
            features = {name: float(example_index + rank) for name in FEATURE_ORDER}
            features["decision_rank"] = float(rank)
            rows.append(
                {
                    "example_id": f"example-{example_index}",
                    "group_id": f"example-{example_index}",
                    "condition": "synthetic",
                    "domain": "text",
                    "decision_rank": rank,
                    "label_continue": int(has_continue and rank == 1),
                    "features": features,
                }
            )

    oof, info = grouped_oof(rows)
    assert len(oof) == len(rows)
    assert all(0.0 <= row["oof_continue_probability"] <= 1.0 for row in oof)
    validation_membership = {}
    for fold in info["folds"]:
        for example_id in fold["validation_example_ids"]:
            assert example_id not in validation_membership
            validation_membership[example_id] = fold["fold"]
    assert set(validation_membership) == {f"example-{index}" for index in range(12)}


def test_sequential_stopping_is_deterministic_and_deduplicated():
    policy = fitted_policy(threshold=0.5)
    candidates = [
        candidate(1.0, ["a", "b"], 1),
        candidate(0.9, ["b", "c"], 2),
        candidate(0.1, ["c"], 3),
    ]
    first = adaptive_v2_support_aggregate(candidates, policy=policy)
    second = adaptive_v2_support_aggregate(candidates, policy=policy)

    assert first == second
    assert 1 <= first["selected_k"] <= 3
    assert first["support_units"] == list(dict.fromkeys(first["support_units"]))


def test_always_selects_one_and_handles_no_second_candidate():
    policy = AdaptiveV2Policy("text", None, None, 0.5)
    result = adaptive_v2_support_aggregate([candidate(1.0, ["a", "a"], 1)], policy=policy)

    assert result["selected_k"] == 1
    assert result["support_units"] == ["a"]
    assert result["decisions"] == []


def test_never_selects_more_than_five():
    policy = fitted_policy(threshold=0.0)
    candidates = [candidate(1.0, [str(index)], index) for index in range(1, 7)]
    result = adaptive_v2_support_aggregate(candidates, policy=policy)

    assert result["selected_k"] == 5
    assert result["support_units"] == ["1", "2", "3", "4", "5"]
    assert result["decisions"][-1]["decision_rank"] == 4


class GoldGuardCandidate(Mapping[str, Any]):
    def __init__(self, allowed: dict[str, Any]):
        self.allowed = allowed

    def __getitem__(self, key: str) -> Any:
        if key not in self.allowed:
            raise AssertionError(f"Adaptive-v2 accessed forbidden field: {key}")
        return self.allowed[key]

    def __iter__(self):
        return iter(self.allowed)

    def __len__(self) -> int:
        return len(self.allowed)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in {"rank", "adjusted_score", "subgraph_units"}:
            raise AssertionError(f"Adaptive-v2 accessed forbidden field: {key}")
        return self.allowed.get(key, default)


def test_inference_policy_is_independent_of_gold_example_fields():
    candidates = [
        GoldGuardCandidate(
            {"rank": 1, "adjusted_score": 1.0, "subgraph_units": ["a"]}
        ),
        GoldGuardCandidate(
            {"rank": 2, "adjusted_score": 0.9, "subgraph_units": ["b"]}
        ),
    ]
    result = adaptive_v2_support_aggregate(
        candidates, policy=fitted_policy(threshold=0.0)
    )

    assert result["selected_k"] == 2
    assert result["support_units"] == ["a", "b"]


@pytest.mark.parametrize("support_function", [generated_support, evaluated_support])
def test_pipeline_requires_explicit_v2_mode_and_policy(support_function):
    item = {
        "top5": [candidate(1.0, ["a"], 1), candidate(0.9, ["b"], 2)]
    }

    assert support_function(item, top_k=1) == ["a"]
    assert support_function(
        item,
        top_k=1,
        aggregation_mode="adaptive_v2",
        adaptive_v2_policy=fitted_policy(threshold=0.0),
    ) == ["a", "b"]
    with pytest.raises(ValueError, match="adaptive_v2_policy"):
        support_function(item, top_k=1, aggregation_mode="adaptive_v2")


@pytest.fixture
def adaptive_policy_work_dir():
    root = Path("outputs") / ".test_scratch_adaptive_k" / uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def ranking_record(method: str, domain: str, index: int, *, positive: bool) -> dict[str, Any]:
    prefix = f"{method}-{domain}-{index}"
    ranked = []
    for rank, score in enumerate((0.9, 0.8, 0.7, 0.6, 0.5), start=1):
        if rank == 1:
            unit = f"{prefix}-gold-0"
        elif rank == 2 and positive:
            unit = f"{prefix}-gold-1"
        else:
            unit = f"{prefix}-noise-{rank}"
        ranked.append(
            {
                "rank": rank,
                "score": score - 0.2,
                "adjusted_score": score,
                "subgraph_size": 1,
                "subgraph_units": [unit],
            }
        )
    gold = [f"{prefix}-gold-0"]
    if positive:
        gold.append(f"{prefix}-gold-1")
    return {
        "dataset": f"dataset-{domain}",
        "domain": domain,
        "split": "dev",
        "example_id": prefix,
        "method": method,
        "score_mode": "neural" if method == "gnn_only" else "persisted_symbolic",
        "gold_explanations": [gold],
        "ranked_candidates": ranked,
        "prefix_evaluation": [],
    }


def write_rankings(root: Path, rows: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "per_example_rankings.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


@pytest.mark.parametrize("ranking_method", ["gnn_only", "sageqa_final"])
def test_production_fitter_loads_only_requested_persisted_ranking(
    adaptive_policy_work_dir: Path, ranking_method: str
):
    other_method = "sageqa_final" if ranking_method == "gnn_only" else "gnn_only"
    selected = [
        ranking_record(ranking_method, "text", 0, positive=True),
        ranking_record(ranking_method, "ontology", 0, positive=False),
    ]
    ignored = {
        "split": "dev",
        "method": other_method,
        "example_id": selected[0]["example_id"],
        "ranked_candidates": "must not be inspected",
    }
    write_rankings(adaptive_policy_work_dir, [selected[0], ignored, selected[1]])

    loaded = load_clean_final_records(adaptive_policy_work_dir, ranking_method)

    assert [row["example_id"] for row in loaded] == [row["example_id"] for row in selected]
    assert loaded[0]["candidates"][0]["adjusted_score"] == selected[0]["ranked_candidates"][0][
        "adjusted_score"
    ]
    assert loaded[0]["candidates"][0]["score"] == selected[0]["ranked_candidates"][0][
        "score"
    ]


def write_policy_bundle(
    root: Path,
    *,
    ranking_method: str = "sageqa_final",
    config_thresholds: dict[str, float] | None = None,
    joblib_thresholds: dict[str, float] | None = None,
) -> None:
    config_thresholds = config_thresholds or {"text": 0.13, "ontology": 0.87}
    joblib_thresholds = joblib_thresholds or dict(config_thresholds)
    descriptions = {}
    model_config = None
    for domain in ("text", "ontology"):
        policy = fitted_policy()
        joblib.dump(policy.scaler, root / f"{domain}_scaler.joblib")
        joblib.dump(policy.model, root / f"{domain}_model.joblib")
        descriptions[domain] = {
            "domain": domain,
            "feature_order": list(FEATURE_ORDER),
            "scaler_mean": policy.scaler.mean_.tolist(),
            "scaler_scale": policy.scaler.scale_.tolist(),
            "coefficients_in_model_feature_order": policy.model.coef_[0].tolist(),
            "coefficient_by_feature": dict(zip(FEATURE_ORDER, policy.model.coef_[0].tolist())),
            "intercept": float(policy.model.intercept_[0]),
            "classes": policy.model.classes_.astype(int).tolist(),
        }
        params = policy.model.get_params()
        model_config = {
            key: params[key] for key in ("class_weight", "solver", "max_iter", "random_state")
        }
    joblib.dump(joblib_thresholds, root / "chosen_thresholds.joblib")
    fitted_names = [
        "chosen_thresholds.joblib",
        "text_scaler.joblib",
        "text_model.joblib",
        "ontology_scaler.joblib",
        "ontology_model.joblib",
    ]
    config = {
        "ranking_method": ranking_method,
        "allowed_k": [1, 2, 3, 5],
        "domains": ["text", "ontology"],
        "dataset_specific_thresholds": False,
        "transitions": [
            {"current_k": current, "continue_to_k": following}
            for current, following in ((1, 2), (2, 3), (3, 5))
        ],
        "thresholds": config_thresholds,
        "feature_order": list(FEATURE_ORDER),
        "continuous_features_standardized": list(CONTINUOUS_FEATURES),
        "model": model_config,
        "fitted_artifact_sha256": {name: sha256(root / name) for name in fitted_names},
        "full_dev_model_parameters": descriptions,
    }
    (root / "adaptive_k_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    manifest_names = ["adaptive_k_config.json", *fitted_names]
    (root / "artifact_manifest.json").write_text(
        json.dumps({"files": {name: sha256(root / name) for name in manifest_names}}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def test_test_loader_accepts_arbitrary_consistent_dev_thresholds(adaptive_policy_work_dir: Path):
    write_policy_bundle(adaptive_policy_work_dir)

    policies, config, _ = load_policy_bundle(adaptive_policy_work_dir, "sageqa_final")

    assert config["thresholds"] == {"text": 0.13, "ontology": 0.87}
    assert policies["text"].threshold == 0.13
    assert policies["ontology"].threshold == 0.87


def test_test_loader_rejects_inconsistent_threshold_artifacts(adaptive_policy_work_dir: Path):
    write_policy_bundle(
        adaptive_policy_work_dir,
        config_thresholds={"text": 0.13, "ontology": 0.87},
        joblib_thresholds={"text": 0.14, "ontology": 0.87},
    )

    with pytest.raises(ValueError, match="threshold artifact disagrees"):
        load_policy_bundle(adaptive_policy_work_dir, "sageqa_final")


def test_test_loader_rejects_policy_ranking_method_mismatch(adaptive_policy_work_dir: Path):
    write_policy_bundle(adaptive_policy_work_dir, ranking_method="gnn_only")

    with pytest.raises(ValueError, match="Wrong policy ranking method"):
        load_policy_bundle(adaptive_policy_work_dir, "sageqa_final")


def test_production_fitting_opens_no_test_file(
    adaptive_policy_work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    input_dir = adaptive_policy_work_dir / "dev_rankings"
    rows = [
        ranking_record("gnn_only", domain, index, positive=index < 6)
        for domain in ("text", "ontology")
        for index in range(12)
    ]
    write_rankings(input_dir, rows)
    (input_dir / "metrics.json").write_text(
        json.dumps({"k_values": [1, 2, 3, 5]}) + "\n", encoding="utf-8"
    )
    (input_dir / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "test_rows_read": 0,
                "test_gold_accessed": False,
                "adaptive_k_ready": True,
                "code_commit_hash": "synthetic",
                "datasets": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    opened_test_files = []
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.name.lower().startswith("test"):
            opened_test_files.append(path)
            raise AssertionError(f"Fitter opened TEST file: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(
        "evaluation.fit_production_dev_adaptive_k.git_output", lambda *args: "synthetic"
    )
    fit(
        SimpleNamespace(
            input_dir=input_dir,
            output_dir=adaptive_policy_work_dir / "policy",
            ranking_method="gnn_only",
            overwrite=False,
        )
    )

    assert opened_test_files == []
    config = json.loads(
        (adaptive_policy_work_dir / "policy" / "adaptive_k_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["ranking_method"] == "gnn_only"
