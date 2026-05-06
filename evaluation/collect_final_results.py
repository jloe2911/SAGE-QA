import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, List


MODEL_FILES = {
    "Random Subgraph": [
        "random_hybrid_metrics.json",
        "random_nl_metrics.json",
    ],
    "Lexical Subgraph": [
        "lexical_hybrid_metrics.json",
        "lexical_nl_metrics.json",
    ],
    "Random Axiom": [
        "axiom_random_hybrid_min_gold_metrics.json",
        "axiom_random_nl_min_gold_metrics.json",
    ],
    "Lexical Axiom": [
        "axiom_lexical_hybrid_min_gold_metrics.json",
        "axiom_lexical_nl_min_gold_metrics.json",
    ],
    "GNN Subgraph": [
        "gnn_neural/test_split_metrics.json",
        "gnn_neural/metrics.json",
        "gnn_neural/test_metrics.json",
        "gnn_neural/eval_metrics.json",
    ],
    "NeSyQA": [
        "gnn_nesyqa_compact/test_split_metrics.json",
        "gnn_nesyqa_compact/metrics.json",
        "gnn_nesyqa_compact/test_metrics.json",
        "gnn_nesyqa_compact/eval_metrics.json",
    ],
}


SPLIT_ORDER = ["ALL", "BIN", "OPEN"]
ALL_COLUMNS = [
    "Dataset",
    "Type",
    "Model",
    "N",
    "EM@1",
    "Gold-Contained@1",
    "Support-Jaccard@1",
    "Support-F1@1",
    "EM@3",
    "Gold-Contained@3",
    "Support-Jaccard@3",
    "Support-F1@3",
    "EM@5",
    "Gold-Contained@5",
    "Support-Jaccard@5",
    "Support-F1@5",
]


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_metrics_file(dataset_dir: Path, candidates: List[str]) -> Optional[Path]:
    """
    Tries exact candidate paths first. If not found, searches subdirectories.
    """
    for rel in candidates:
        path = dataset_dir / rel
        if path.exists():
            return path

    # fallback: recursively search likely metric files
    all_jsons = list(dataset_dir.rglob("*metrics*.json"))

    for rel in candidates:
        wanted_name = Path(rel).name
        for path in all_jsons:
            if path.name == wanted_name:
                return path

    return None


def get_metric(m: Dict, *keys: str, default: float = 0.0) -> float:
    """
    Reads metric keys across different evaluator naming conventions.
    """
    for key in keys:
        if key in m:
            try:
                return float(m[key])
            except Exception:
                return default
    return default


def normalize_metric_row(dataset: str, model: str, split_name: str, m: Dict) -> Dict:
    """
    Converts all metric naming styles into final paper metric names.
    """

    return {
        "Dataset": dataset,
        "Type": split_name,
        "Model": model,
        "N": int(m.get("examples", 0)),
        # Paper names
        "EM@1": get_metric(
            m,
            "em@1",
            "exact@1",
            "exact_hit@1",
            default=0.0,
        ),
        "Gold-Contained@1": get_metric(
            m,
            "gold_contained@1",
            "contains@1",
            "contains_gold_hit@1",
            default=0.0,
        ),
        "Support-Jaccard@1": get_metric(
            m,
            "support_jaccard@1",
            "jaccard@1",
            "best_jaccard@1",
            default=0.0,
        ),
        "Support-F1@1": get_metric(
            m,
            "support_f1@1",
            "set_f1@1",
            "best_set_f1@1",
            default=0.0,
        ),
        "EM@3": get_metric(
            m,
            "em@3",
            "exact@3",
            "exact_hit@3",
            default=0.0,
        ),
        "Gold-Contained@3": get_metric(
            m,
            "gold_contained@3",
            "contains@3",
            "contains_gold_hit@3",
            default=0.0,
        ),
        "Support-Jaccard@3": get_metric(
            m,
            "support_jaccard@3",
            "jaccard@3",
            "best_jaccard@3",
            default=0.0,
        ),
        "Support-F1@3": get_metric(
            m,
            "support_f1@3",
            "set_f1@3",
            "best_set_f1@3",
            default=0.0,
        ),
        "EM@5": get_metric(
            m,
            "em@5",
            "exact@5",
            "exact_hit@5",
            default=0.0,
        ),
        "Gold-Contained@5": get_metric(
            m,
            "gold_contained@5",
            "contains@5",
            "contains_gold_hit@5",
            default=0.0,
        ),
        "Support-Jaccard@5": get_metric(
            m,
            "support_jaccard@5",
            "jaccard@5",
            "best_jaccard@5",
            default=0.0,
        ),
        "Support-F1@5": get_metric(
            m,
            "support_f1@5",
            "set_f1@5",
            "best_set_f1@5",
            default=0.0,
        ),
    }


def infer_summary_split(raw_split_name: str) -> Optional[str]:
    """
    Normalizes split names from both baseline and GNN evaluators.

    Baselines use ALL/BIN/OPEN directly. GNN split metrics use names such as:
      FamilyOWL_1hop | 1hop | BIN
    """
    split = str(raw_split_name).strip()
    if split in SPLIT_ORDER:
        return split

    tail = split.rsplit("|", 1)[-1].strip().upper()
    if tail in {"BIN", "OPEN"}:
        return tail

    return None


def aggregate_rows(rows: List[Dict]) -> Dict:
    total_n = sum(int(row.get("examples", 0)) for row in rows)
    if total_n <= 0:
        return {"examples": 0}

    out = {"examples": total_n}
    keys = set().union(*(row.keys() for row in rows))

    for key in keys:
        if key == "examples":
            continue
        weighted_sum = 0.0
        seen = False
        for row in rows:
            n = int(row.get("examples", 0))
            if key in row:
                weighted_sum += float(row[key]) * n
                seen = True
        if seen:
            out[key] = weighted_sum / total_n

    return out


def normalize_metric_splits(metrics: Dict) -> Dict[str, Dict]:
    out = {}
    grouped = {"BIN": [], "OPEN": []}

    for raw_split_name, values in metrics.items():
        if not isinstance(values, dict):
            continue

        raw_split = str(raw_split_name).strip()
        split_name = infer_summary_split(raw_split)
        if split_name is None:
            continue

        if raw_split in SPLIT_ORDER:
            out[raw_split] = values
        elif split_name in grouped:
            grouped[split_name].append(values)

    for split_name, rows in grouped.items():
        if rows:
            out[split_name] = aggregate_rows(rows)

    if "ALL" not in out:
        aggregate_inputs = []
        aggregate_inputs.extend(grouped["BIN"])
        aggregate_inputs.extend(grouped["OPEN"])
        if aggregate_inputs:
            out["ALL"] = aggregate_rows(aggregate_inputs)

    return out


def collect_dataset(dataset_name: str, dataset_dir: Path) -> List[Dict]:
    rows = []

    for model_name, candidate_paths in MODEL_FILES.items():
        metrics_path = find_metrics_file(dataset_dir, candidate_paths)

        if metrics_path is None:
            print(f"[WARN] Missing {model_name} metrics for {dataset_name}")
            continue

        metrics = load_json(metrics_path)

        normalized_splits = normalize_metric_splits(metrics)

        for split_name in SPLIT_ORDER:
            if split_name not in normalized_splits:
                continue

            rows.append(
                normalize_metric_row(
                    dataset=dataset_name,
                    model=model_name,
                    split_name=split_name,
                    m=normalized_splits[split_name],
                )
            )

        print(f"[OK] {dataset_name} | {model_name}: {metrics_path}")

    return rows


def format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def escape_latex(value) -> str:
    text = format_value(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_csv(path: Path, rows: List[Dict], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_latex(path: Path, rows: List[Dict], columns: List[str]) -> None:
    alignment = "l" * len(columns)
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(escape_latex(col) for col in columns) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            " & ".join(escape_latex(row.get(col, "")) for col in columns) + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def select_columns(rows: List[Dict], columns: List[str]) -> List[Dict]:
    return [{col: row.get(col, "") for col in columns} for row in rows]


def write_main_tables(rows: List[Dict], output_dir: Path) -> None:
    """
    Writes smaller paper-ready tables.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    main_cols = [
        "Dataset",
        "Type",
        "Model",
        "N",
        "EM@1",
        "Gold-Contained@1",
        "Support-Jaccard@1",
        "Support-F1@1",
    ]

    topk_cols = [
        "Dataset",
        "Type",
        "Model",
        "N",
        "Gold-Contained@1",
        "Gold-Contained@3",
        "Gold-Contained@5",
        "Support-F1@5",
    ]

    main_rows = select_columns(rows, main_cols)
    topk_rows = select_columns(rows, topk_cols)

    write_csv(output_dir / "final_summary_main.csv", main_rows, main_cols)
    write_csv(output_dir / "final_summary_topk.csv", topk_rows, topk_cols)

    write_latex(output_dir / "final_summary_main.tex", main_rows, main_cols)
    write_latex(output_dir / "final_summary_topk.tex", topk_rows, topk_cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="outputs/final_results")
    parser.add_argument(
        "--output-csv", default="outputs/final_results/final_summary.csv"
    )
    parser.add_argument(
        "--output-latex", default="outputs/final_results/final_summary.tex"
    )
    args = parser.parse_args()

    root = Path(args.results_root)

    all_rows = []

    for dataset_name in ["FamilyOWL_1hop", "FamilyOWL_2hop"]:
        dataset_dir = root / dataset_name

        if not dataset_dir.exists():
            print(f"[WARN] Missing dataset directory: {dataset_dir}")
            continue

        all_rows.extend(collect_dataset(dataset_name, dataset_dir))

    if not all_rows:
        raise RuntimeError(f"No metric rows collected from {root}")

    # Stable ordering
    model_order = {
        "Random Subgraph": 0,
        "Random Axiom": 1,
        "Lexical Axiom": 2,
        "Lexical Subgraph": 3,
        "GNN Subgraph": 4,
        "NeSyQA": 5,
    }

    dataset_order = {
        "FamilyOWL_1hop": 0,
        "FamilyOWL_2hop": 1,
    }

    type_order = {
        "ALL": 0,
        "BIN": 1,
        "OPEN": 2,
    }

    all_rows = sorted(
        all_rows,
        key=lambda row: (
            dataset_order.get(row["Dataset"], 99),
            type_order.get(row["Type"], 99),
            model_order.get(row["Model"], 99),
        ),
    )

    output_csv = Path(args.output_csv)
    output_latex = Path(args.output_latex)

    write_csv(output_csv, all_rows, ALL_COLUMNS)
    write_latex(output_latex, all_rows, ALL_COLUMNS)

    write_main_tables(all_rows, output_csv.parent)

    for row in all_rows:
        print(row)
    print(f"\nSaved full CSV to: {output_csv}")
    print(f"Saved full LaTeX to: {output_latex}")
    print(f"Saved paper-ready tables to: {output_csv.parent}")


if __name__ == "__main__":
    main()
