"""
Batch evaluation: GNN-only vs SAGE-QA Text-Chain vs SAGE-QA Proof reranking.

Runs all 200 test examples for 2WikiMultiHopQA and HotpotQA (text benchmarks)
and optionally FamilyOWL_2hop (OWL benchmark).

Metrics per benchmark and question_type:
  - Hit@1 (J≥0.5)
  - Mean Jaccard
  - EM@1 (exact match)

Run with:  .venv/bin/python scripts/batch_eval_sageqa.py [--benchmarks owl text all]
"""
import sys, json, argparse, time, torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.eval_gnn_subgraph_retriever import load_gnn_checkpoint
from models.gnn_subgraph_retriever import GNNSubgraphRetriever
from training.train_gnn_subgraph_retriever import (
    load_jsonl, prepare_examples,
    encode_example_graph, score_candidate_rows,
)
from models.symbolic_composer import SymbolicComposer
from transformers import AutoTokenizer

device = torch.device("cpu")

BENCHMARKS = {
    "FamilyOWL_2hop": (
        "checkpoints/familyowl_2hop/best_model.pt",
        "data/FamilyOWL_2hop/test_subgraph_retrieval.jsonl",
        "owl",
    ),
    "2WikiMultiHopQA": (
        "checkpoints/2wiki/best_model.pt",
        "data/2WikiMultiHopQA/test_subgraph_retrieval.jsonl",
        "text",
    ),
    "HotpotQA": (
        "checkpoints/hotpotqa/best_model.pt",
        "data/HotpotQA/test_subgraph_retrieval.jsonl",
        "text",
    ),
}

composer = SymbolicComposer()


# ── Metrics ────────────────────────────────────────────────────────────────────

def jaccard(a, b):
    s, t = set(a), set(b)
    return len(s & t) / len(s | t) if s | t else 1.0

def best_j(pred, golds):
    return max((jaccard(pred, g) for g in golds), default=0.0)

def exact_match(pred, golds):
    return any(set(pred) == set(g) for g in golds)


class Metrics:
    def __init__(self):
        self.j_sum = 0.0
        self.hit  = 0
        self.em   = 0
        self.n    = 0

    def add(self, pred, golds):
        j = best_j(pred, golds)
        self.j_sum += j
        self.hit   += int(j >= 0.5)
        self.em    += int(exact_match(pred, golds))
        self.n     += 1

    def summary(self):
        if self.n == 0:
            return {"n": 0, "hit@1": 0.0, "mean_j": 0.0, "em@1": 0.0}
        return {
            "n":      self.n,
            "hit@1":  round(self.hit / self.n, 4),
            "mean_j": round(self.j_sum / self.n, 4),
            "em@1":   round(self.em / self.n, 4),
        }


# ── Per-benchmark runner ───────────────────────────────────────────────────────

def run_benchmark(bench_name, ckpt_path, data_path, bench_type, max_examples=None):
    print(f"\n{'='*70}")
    print(f"Benchmark: {bench_name}  [{bench_type}]")
    print(f"{'='*70}")

    ckpt = load_gnn_checkpoint(ckpt_path, device)
    tok  = AutoTokenizer.from_pretrained(ckpt["model_name"])
    model = GNNSubgraphRetriever(
        model_name=ckpt["model_name"],
        node_symbolic_dim=ckpt.get("node_symbolic_dim", 8),
        subgraph_symbolic_dim=ckpt.get("subgraph_symbolic_dim", 8),
        gnn_hidden_dim=ckpt.get("gnn_hidden_dim", 128),
        gnn_layers=ckpt.get("gnn_layers", 2),
        classifier_hidden_dim=ckpt.get("classifier_hidden_dim", 128),
        freeze_encoder=ckpt.get("freeze_encoder", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    raw = load_jsonl(data_path)
    # Build question_type lookup from raw rows (field is dropped by prepare_examples)
    qtype_by_id = {r["example_id"]: r.get("question_type", "unknown") for r in raw}
    examples = prepare_examples(raw)
    if max_examples:
        examples = examples[:max_examples]

    print(f"Running {len(examples)} examples …")

    gnn_all   = Metrics()
    sage_all  = Metrics()
    # per question_type buckets
    gnn_by_qt  = defaultdict(Metrics)
    sage_by_qt = defaultdict(Metrics)

    per_example = []

    with torch.no_grad():
        for idx, ex in enumerate(examples):
            question     = ex["question"]
            sparql_query = ex.get("sparql_query", "")
            answer       = ex.get("answer", "")
            cand_rows    = ex["candidate_rows"]
            gold_exps    = cand_rows[0].get("gold_explanations", []) if cand_rows else []
            qtype        = qtype_by_id.get(ex.get("example_id", ""), "unknown")

            # GNN scoring
            encoded = encode_example_graph(
                model=model, tokenizer=tok,
                example=ex, device=device, max_length=128,
            )
            scored = []
            for start in range(0, len(cand_rows), 64):
                batch = cand_rows[start:start+64]
                out   = score_candidate_rows(
                    model=model, encoded_graph=encoded,
                    candidate_rows=batch, device=device,
                )
                probs = out["probs"].tolist()
                for row, p in zip(batch, probs):
                    scored.append((row["subgraph_units"], p, row.get("label", 0)))

            scored.sort(key=lambda x: x[1], reverse=True)
            gnn_top1 = scored[0][0] if scored else []

            # SAGE-QA reranking
            axiom_scores = {}
            for units, sc, _ in scored:
                for u in units:
                    if u not in axiom_scores or sc > axiom_scores[u]:
                        axiom_scores[u] = sc

            retrieved_units = [
                {"axiom": ax, "score": sc, "label": 0}
                for ax, sc in axiom_scores.items()
            ]
            composed  = composer.compose(
                question=question,
                retrieved_units=retrieved_units,
                sparql_query=sparql_query,
            )
            sage_top1 = [
                n["raw_axiom"]
                for n in composed.get("best_subgraph", {}).get("nodes", [])
            ]

            gnn_j   = best_j(gnn_top1,  gold_exps)
            sage_j  = best_j(sage_top1, gold_exps)
            gnn_em  = exact_match(gnn_top1,  gold_exps)
            sage_em = exact_match(sage_top1, gold_exps)

            gnn_all.add(gnn_top1,  gold_exps)
            sage_all.add(sage_top1, gold_exps)
            gnn_by_qt[qtype].add(gnn_top1,  gold_exps)
            sage_by_qt[qtype].add(sage_top1, gold_exps)

            delta_j = sage_j - gnn_j
            per_example.append({
                "example_id": ex.get("example_id", ""),
                "qtype": qtype,
                "question": question[:80],
                "gnn_j":  round(gnn_j,  3),
                "sage_j": round(sage_j, 3),
                "delta_j": round(delta_j, 3),
                "gnn_em":  gnn_em,
                "sage_em": sage_em,
            })

            if (idx + 1) % 25 == 0:
                gs = gnn_all.summary()
                ss = sage_all.summary()
                print(f"  [{idx+1:3d}/{len(examples)}]"
                      f"  GNN hit@1={gs['hit@1']:.3f} em={gs['em@1']:.3f}"
                      f"  SAGE hit@1={ss['hit@1']:.3f} em={ss['em@1']:.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    gs = gnn_all.summary()
    ss = sage_all.summary()
    print(f"\n{'─'*60}")
    print(f"{'Metric':<18}  {'GNN':>8}  {'SAGE-QA':>8}  {'Δ':>7}")
    print(f"{'─'*60}")
    for key in ("hit@1", "mean_j", "em@1"):
        delta = ss[key] - gs[key]
        sign  = "+" if delta >= 0 else ""
        print(f"  {key:<16}  {gs[key]:>8.4f}  {ss[key]:>8.4f}  {sign}{delta:>6.4f}")

    # Per question-type
    all_qtypes = sorted(set(list(gnn_by_qt.keys()) + list(sage_by_qt.keys())))
    if len(all_qtypes) > 1:
        print(f"\n  By question_type:")
        for qt in all_qtypes:
            g = gnn_by_qt[qt].summary()
            s = sage_by_qt[qt].summary()
            d_em = s["em@1"] - g["em@1"]
            sign = "+" if d_em >= 0 else ""
            print(f"    {qt:<30}  n={g['n']:3d}"
                  f"  GNN em={g['em@1']:.3f}  SAGE em={s['em@1']:.3f}"
                  f"  Δ={sign}{d_em:.3f}")

    # Top improvements and regressions
    improved   = sorted([e for e in per_example if e["delta_j"] > 0],
                        key=lambda x: x["delta_j"], reverse=True)
    regressed  = sorted([e for e in per_example if e["delta_j"] < 0],
                        key=lambda x: x["delta_j"])
    unchanged  = [e for e in per_example if e["delta_j"] == 0]

    print(f"\n  Improved : {len(improved):3d}   Unchanged: {len(unchanged):3d}"
          f"   Regressed: {len(regressed):3d}")

    if improved[:5]:
        print(f"\n  Top-5 improvements (Δ J):")
        for e in improved[:5]:
            print(f"    [{e['qtype']:<20}] {e['delta_j']:+.3f}"
                  f"  {e['gnn_j']:.2f}→{e['sage_j']:.2f}  {e['question']}")

    if regressed[:5]:
        print(f"\n  Top-5 regressions (Δ J):")
        for e in regressed[:5]:
            print(f"    [{e['qtype']:<20}] {e['delta_j']:+.3f}"
                  f"  {e['gnn_j']:.2f}→{e['sage_j']:.2f}  {e['question']}")

    return {
        "bench": bench_name,
        "gnn": gs,
        "sage": ss,
        "per_example": per_example,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--benchmarks", nargs="+",
        choices=["all", "text", "owl", "FamilyOWL_2hop", "2WikiMultiHopQA", "HotpotQA"],
        default=["text"],
        help="Which benchmarks to run (default: text = 2Wiki + HotpotQA)",
    )
    p.add_argument(
        "--max-examples", type=int, default=None,
        help="Limit number of examples per benchmark (for quick tests)",
    )
    p.add_argument(
        "--output", default="outputs/batch_eval_sageqa.json",
        help="JSON output path",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve benchmark list
    selected = set()
    for b in args.benchmarks:
        if b == "all":
            selected.update(BENCHMARKS.keys())
        elif b == "text":
            selected.update(k for k, v in BENCHMARKS.items() if v[2] == "text")
        elif b == "owl":
            selected.update(k for k, v in BENCHMARKS.items() if v[2] == "owl")
        else:
            selected.add(b)

    t0 = time.time()
    all_results = []

    for bench_name in ["FamilyOWL_2hop", "2WikiMultiHopQA", "HotpotQA"]:
        if bench_name not in selected:
            continue
        ckpt_path, data_path, bench_type = BENCHMARKS[bench_name]
        try:
            result = run_benchmark(
                bench_name, ckpt_path, data_path, bench_type,
                max_examples=args.max_examples,
            )
            all_results.append(result)
        except FileNotFoundError as e:
            print(f"\n[{bench_name}] skipped — {e}")

    # ── Cross-benchmark summary ────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"CROSS-BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Benchmark':<20}  {'GNN hit@1':>10}  {'SAGE hit@1':>10}"
              f"  {'GNN EM':>8}  {'SAGE EM':>8}  {'Δ EM':>7}")
        for r in all_results:
            g, s = r["gnn"], r["sage"]
            d_em = s["em@1"] - g["em@1"]
            sign = "+" if d_em >= 0 else ""
            print(f"  {r['bench']:<20}  {g['hit@1']:>10.4f}  {s['hit@1']:>10.4f}"
                  f"  {g['em@1']:>8.4f}  {s['em@1']:>8.4f}  {sign}{d_em:>6.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
