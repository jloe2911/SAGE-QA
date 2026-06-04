"""
Step-by-step pipeline debug for selected examples across all benchmarks.
Shows: GNN candidate scores → SymbolicComposer breakdown → LLM answer.

Run with: .venv/bin/python scripts/debug_pipeline.py
"""
import sys, json, time, requests, torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.eval_gnn_subgraph_retriever import load_gnn_checkpoint
from models.gnn_subgraph_retriever import GNNSubgraphRetriever
from training.train_gnn_subgraph_retriever import (
    load_jsonl, prepare_examples,
    encode_example_graph, score_candidate_rows,
)
from models.symbolic_composer import SymbolicComposer, extract_query_signature, parse_axiom
from docx.oxml.ns import qn
from transformers import AutoTokenizer

device = torch.device("cpu")

API_KEY = "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"
MODEL   = "google/gemma-4-31b-it:free"

# ── Selected examples ──────────────────────────────────────────────────────────
EXAMPLES = [
    # FamilyOWL_2hop — three distinct OWL axiom patterns
    {"bench": "FamilyOWL_2hop", "cat": "isSpouseOf owl:inverseOf hasSpouse",
     "eid": "FamilyOWL_2hop__g293__q0__1hop-Thing_alice_whitfield_1859_james_whitfield_1821-james_whitfield_1821-isSpouseOf-BIN__Is James Whitfield married to Harriet Ann Young?__ASK WHERE { <http://www.example.com/genealogy.owl#james_whitfield_1821> <http://www.example.com/genealogy.owl#isSpouseOf> <http://www.example.com/genealogy.owl#harriet_ann_young_1825> }"},
    {"bench": "FamilyOWL_2hop", "cat": "hasAncestor owl:inverseOf isAncestorOf",
     "eid": "FamilyOWL_2hop__g294__q0__1hop-Thing_alice_whitfield_1859_alice_whitfield_1859-alice_whitfield_1859-hasAncestor-BIN__Does Alice Whitfield have Edward Young as an ancestor?__ASK WHERE { <http://www.example.com/genealogy.owl#alice_whitfield_1859> <http://www.example.com/genealogy.owl#hasAncestor> <http://www.example.com/genealogy.owl#edward_young_1795> }"},
    {"bench": "FamilyOWL_2hop", "cat": "isSiblingOf rdfs:subPropertyOf isBloodrelationOf",
     "eid": "FamilyOWL_2hop__g398__q0__1hop-Thing_ann_green_1806_mary_green_1803-mary_green_1803-isBloodrelationOf-BIN__Is Mary Green a blood relative of Rebecca Green?__ASK WHERE { <http://www.example.com/genealogy.owl#mary_green_1803> <http://www.example.com/genealogy.owl#isBloodrelationOf> <http://www.example.com/genealogy.owl#rebecca_green_1800> }"},
    # 2WikiMultiHopQA — comparison, film→composer→death, person→father→birthplace
    {"bench": "2WikiMultiHopQA", "cat": "comparison — who lived longer",
     "eid": "2WikiMultiHopQA__test__3c736a9508fc11ebbdadac1f6bf848b6"},
    {"bench": "2WikiMultiHopQA", "cat": "film → composer → place of death",
     "eid": "2WikiMultiHopQA__test__cfad61500bdd11eba7f7acde48001122"},
    {"bench": "2WikiMultiHopQA", "cat": "person → father → birthplace",
     "eid": "2WikiMultiHopQA__test__9be20dbc0bdd11eba7f7acde48001122"},
    # HotpotQA — comparison, formation date, same-article chain
    {"bench": "HotpotQA", "cat": "comparison — who is older",
     "eid": "HotpotQA__test__5a78f753554299078472776e"},
    {"bench": "HotpotQA", "cat": "which band formed first",
     "eid": "HotpotQA__test__5a8e276b5542995a26add468"},
    {"bench": "HotpotQA", "cat": "same-article multi-sentence chain",
     "eid": "HotpotQA__test__5ade66dd55429975fa854ebe"},
]

BENCH_CONFIG = {
    "FamilyOWL_2hop":  ("checkpoints/familyowl_2hop/best_model.pt",
                        "data/FamilyOWL_2hop/test_subgraph_retrieval.jsonl"),
    "2WikiMultiHopQA": ("checkpoints/2wiki/best_model.pt",
                        "data/2WikiMultiHopQA/test_subgraph_retrieval.jsonl"),
    "HotpotQA":        ("checkpoints/hotpotqa/best_model.pt",
                        "data/HotpotQA/test_subgraph_retrieval.jsonl"),
}

composer = SymbolicComposer()


# ── Helpers ────────────────────────────────────────────────────────────────────
def jaccard(a, b):
    s, t = set(a), set(b)
    return len(s & t) / len(s | t) if s | t else 1.0

def best_j(pred, golds):
    return max((jaccard(pred, g) for g in golds), default=0.0)

def exact_match(pred, golds):
    return any(set(pred) == set(g) for g in golds)

def llm(prompt):
    for attempt in range(5):
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 64, "temperature": 0.0},
            timeout=30,
        )
        if r.status_code in (429, 503):
            time.sleep(2 ** attempt); continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    return "ERROR"

def build_llm_prompt(question, units, bench):
    if bench == "FamilyOWL_2hop":
        ctx = "OWL facts and axioms:\n" + "\n".join(f"  {u}" for u in units)
        note = "Apply OWL semantics (SymmetricObjectProperty, inverseOf, subPropertyOf). Answer Yes or No."
    else:
        ctx = "Retrieved evidence:\n" + "\n".join(f"  {u}" for u in units)
        note = "Answer concisely based only on the evidence above."
    return f"{note}\n\n{ctx}\n\nQuestion: {question}\nAnswer:"


# ── Load models once per benchmark ────────────────────────────────────────────
loaded_models = {}
loaded_data   = {}

def get_model(bench):
    if bench not in loaded_models:
        ckpt_path, data_path = BENCH_CONFIG[bench]
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
        loaded_models[bench] = (model, tok)

        raw = load_jsonl(data_path)
        by_id = defaultdict(list)
        for row in raw:
            by_id[row["example_id"]].append(row)
        loaded_data[bench] = by_id

    return loaded_models[bench], loaded_data[bench]


# ── Per-example debug ──────────────────────────────────────────────────────────
def debug_example(spec):
    bench = spec["bench"]
    (model, tok), by_id = get_model(bench)

    # Resolve example ID
    eid = spec.get("eid")
    if eid is None:
        q_needle = spec.get("q", "").lower()
        eid = next((k for k, rows in by_id.items()
                    if q_needle in rows[0].get("question", "").lower()), None)
        if eid is None:
            print(f"  [NOT FOUND: {spec.get('q')}]"); return

    rows = by_id.get(eid, [])
    if not rows:
        print(f"  [NO ROWS for {eid[-40:]}]"); return

    examples = prepare_examples(rows)
    if not examples:
        print(f"  [prepare_examples returned empty]"); return
    ex = examples[0]

    question     = ex["question"]
    sparql_query = ex.get("sparql_query", "")
    answer       = ex.get("answer", "")
    cand_rows    = ex["candidate_rows"]
    gold_exps    = cand_rows[0].get("gold_explanations", []) if cand_rows else []

    print(f"\n{'='*72}")
    print(f"[{bench}] [{spec['cat']}]")
    print(f"Q : {question}")
    print(f"A : {answer}")
    if gold_exps:
        print(f"Gold: {gold_exps[0]}")

    # ── Step 1: GNN scoring ───────────────────────────────────────────────────
    print(f"\n── Step 1: GNN scores (top-5 candidates) ──")
    encoded = encode_example_graph(model=model, tokenizer=tok,
                                   example=ex, device=device, max_length=128)
    scored = []
    for start in range(0, len(cand_rows), 64):
        batch = cand_rows[start:start+64]
        out   = score_candidate_rows(model=model, encoded_graph=encoded,
                                     candidate_rows=batch, device=device)
        probs = out["probs"].tolist()
        for row, p in zip(batch, probs):
            scored.append((row["subgraph_units"], p, row.get("label", 0)))

    scored.sort(key=lambda x: x[1], reverse=True)
    gnn_top1 = scored[0][0]

    for rank, (units, prob, lbl) in enumerate(scored[:5], 1):
        j    = best_j(units, gold_exps)
        mark = "★" if lbl else " "
        short = [u[:60]+"…" if len(u)>60 else u for u in units]
        print(f"  [{rank}]{mark} prob={prob:.4f}  J={j:.2f}  {short}")

    gnn_j  = best_j(gnn_top1, gold_exps)
    gnn_em = exact_match(gnn_top1, gold_exps)
    print(f"\n  GNN top-1: J={gnn_j:.2f}, EM={gnn_em}")

    # ── Step 2: SymbolicComposer breakdown ────────────────────────────────────
    print(f"\n── Step 2: SymbolicComposer reranking ──")
    axiom_scores = {}
    for units, prob, _ in scored:
        for u in units:
            if u not in axiom_scores or prob > axiom_scores[u]:
                axiom_scores[u] = prob

    retrieved_units = [{"axiom": ax, "score": sc, "label": 0}
                       for ax, sc in axiom_scores.items()]

    # Run composer internals manually to show breakdown
    parsed_nodes = []
    for idx, item in enumerate(retrieved_units):
        parsed = parse_axiom(item["axiom"])
        parsed_nodes.append({"node_id": f"n{idx}", "raw_axiom": item["axiom"],
                              "candidate_kind": "unknown", "score": float(item["score"]),
                              "label": 0, "parsed_obj": parsed})

    qsig = extract_query_signature(question=question, sparql_query=sparql_query)
    query_entities   = set(qsig["query_entities"])
    query_properties = set(qsig["query_properties"])
    edges = composer._build_graph(parsed_nodes)

    print(f"  Axiom pool: {len(parsed_nodes)} units  |  Edges: {len(edges)}")
    print(f"  Query entities: {query_entities}  |  Properties: {query_properties}")

    composed  = composer.compose(question=question, retrieved_units=retrieved_units,
                                 sparql_query=sparql_query)
    best_sg   = composed["best_subgraph"]
    sage_top1 = [n["raw_axiom"] for n in best_sg.get("nodes", [])]

    print(f"\n  Best subgraph score: {best_sg['score']:.4f}")
    print(f"    connected={best_sg['connected']}  "
          f"fact_rule_mix={best_sg['fact_rule_mix']}  "
          f"query_alignment={best_sg['query_alignment']:.3f}  "
          f"bridge={best_sg['bridge_score']:.3f}")
    print(f"  Selected units:")
    for n in best_sg.get("nodes", []):
        short = n["raw_axiom"][:70] + ("…" if len(n["raw_axiom"]) > 70 else "")
        print(f"    [{n['axiom_type']}] score={n['score']:.4f}  {short}")

    sage_j  = best_j(sage_top1, gold_exps)
    sage_em = exact_match(sage_top1, gold_exps)
    print(f"\n  SAGE-QA top-1: J={sage_j:.2f}, EM={sage_em}")

    # ── Step 3: LLM answer generation ─────────────────────────────────────────
    print(f"\n── Step 3: LLM answer (Gemma-4-31B, context=SAGE-QA top-1) ──")
    time.sleep(4)
    prompt  = build_llm_prompt(question, sage_top1, bench)
    llm_ans = llm(prompt)
    gold_str = str(answer).lower()
    pred_str = llm_ans.lower()
    llm_ok   = gold_str in pred_str or pred_str.strip(".") in gold_str
    print(f"  LLM answer : {llm_ans!r}")
    print(f"  Gold answer: {answer!r}")
    print(f"  Correct    : {llm_ok}")


# ── Run all ───────────────────────────────────────────────────────────────────
for spec in EXAMPLES:
    debug_example(spec)

print(f"\n{'='*72}")
print("Done.")
