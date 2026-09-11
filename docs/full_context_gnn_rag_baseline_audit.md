# Read-only scientific audit: Full Context and GNN-RAG TEST baselines

Date: 2026-09-10  
Scope: frozen TEST artifacts only; no generation, API call, retrieval, proof, prompt, prediction, or metric was changed or rerun.

## Verdict

**Full Context: B — valid but NOT directly comparable; report it only as a reference / upper-context condition.**

The executed Full Context path is clean: text datasets use only the original benchmark `id` and `context` columns, and ontology datasets use only the top-level `OWL Context`. It does not use Generator-D candidates to construct Full Context, and generation was frozen before answer/support gold was opened. However, its mean context is roughly one to three orders of magnitude larger than the retrieval-based reader contexts, depending on dataset and comparator. More importantly, ontology Full Context is not GPT-4.1-mini alone: it runs the existing deterministic proof-first procedure over the complete parsed ontology context, then calls GPT only when that procedure does not prove the query. The deterministic branch resolves 26.6%–36.6% of ontology Full Context predictions and is 100% EM/F1 on every such subset. This is a legitimate no-retrieval upper-context condition, but not a like-for-like reader baseline.

**GNN-RAG: valid weak baseline**, provided the manuscript calls it a **GNN-RAG top-1 EvidenceUnit ablation** rather than native full GNN-RAG reader context.

All 4,249 frozen rank-1 identities map uniquely and exactly to the EvidenceUnit supplied to the reader; there are no empty reader contexts, prediction/input mismatches, or local truncation/serialization mismatches. The weakness is therefore real for the declared top-1 condition. The native GNN-RAG artifact also stores a different, larger context: shortest-path EvidenceUnits to every entity admitted by its cumulative-probability `eps=0.95` policy. That all-`eps` context must not be confused with k1.

## 1. Exact Full Context path

The construction is implemented in `generation/run_final_manuscript_baselines.py`:

Common frozen/join sources are:

- cohort and Lexical identity: `outputs/final_results/production_generator_d_v1_test_baselines/lexical_subgraph/per_example_retrieval.jsonl`;
- GNN-RAG cohort identity: `outputs/final_results/production_generator_d_v1_test_baselines/gnn_rag/predictions_frozen.jsonl`, validated against its `prediction_freeze_manifest.json`;
- question and ontology group/QA metadata: `data/production_generator_d_v1/<dataset>/test_subgraph_retrieval.jsonl`; retained fields are the metadata whitelist, while `subgraph_units` are used only for the separate GNN-RAG identity map; and
- actual Full Context evidence: the per-dataset raw benchmark source listed in the table below.

1. Frozen Lexical and GNN-RAG files establish the 4,249-example cohort.
2. Generator-D TEST candidate rows provide only whitelisted question metadata and the `group_index`/`qa_index` join for ontology. Their candidates do **not** form Full Context.
3. Text Full Context reads the raw Parquet with column projection `columns=["id", "context"]`, normalizes the context, and flattens every non-empty sentence to `SENT::<title>::<index>::<sentence>`.
4. Ontology Full Context uses a selective JSON scanner that decodes only the top-level `OWL Context` string and skips all other values, including `QAs`. `parse_owl_context` parses that RDF graph and emits a deduplicated EvidenceUnit projection of URI-to-URI assertions and supported OWL/RDFS axioms. Literals, comments, RDF list scaffolding, and unsupported blank-node structures are not reader units; property chains are separately normalized.
5. Each resulting list is hash-frozen in `frozen_reader_inputs.jsonl`. Generation subsequently reads only this frozen file.

Thus “complete ontology” below means the **complete parsed EvidenceUnit projection of the benchmark group’s `OWL Context`**, not every raw RDF serialization statement.

| Dataset | Raw source | Fields decoded | Current scientific category | Mean units | Query-conditioned expansion? |
|---|---|---|---|---:|---|
| HotpotQA | `data/raw/hotpot_qa/distractor/validation-00000-of-00001.parquet` | `id`, `context` | A: original benchmark distractor context, sentence-flattened | 40.845 | No; only the benchmark's question-associated context |
| 2WikiMultiHopQA | `data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet` | `id`, `context` | A: original benchmark context, sentence-flattened | 32.274 | No; `evidences` and `supporting_facts` are not projected |
| FamilyOWL_1hop | `data/raw/family/FamilyOWL_1hop.json` | top-level `OWL Context` | D/A-equivalent: complete parsed benchmark ontology context for the group | 99.039 | No |
| FamilyOWL_2hop | `data/raw/family/FamilyOWL_2hop.json` | top-level `OWL Context` | D/A-equivalent | 1,433.000 | No |
| pizza_100_1hop | `data/raw/pizza_100/pizza_100_1hop.json` | top-level `OWL Context` | D/A-equivalent | 126.235 | No |
| pizza_100_2hop | `data/raw/pizza_100/pizza_100_2hop.json` | top-level `OWL Context` | D/A-equivalent | 536.944 | No |
| pizza_250_1hop | `data/raw/pizza_250/pizza_250_1hop.json` | top-level `OWL Context` | D/A-equivalent | 127.255 | No |
| pizza_250_2hop | `data/raw/pizza_250/pizza_250_2hop.json` | top-level `OWL Context` | D/A-equivalent | 718.248 | No |
| OWL2Bench_1hop | `data/raw/owl2bench/OWL2Bench_1hop.json` | top-level `OWL Context` | D/A-equivalent | 386.484 | No |
| OWL2Bench_2hop | `data/raw/owl2bench/OWL2Bench_2hop.json` | top-level `OWL Context` | D/A-equivalent | 718.895 | No |

It is never B (a complete Generator-D `EvidenceGraph`) or C (a union of Generator-D candidates). For text, A is the scientifically natural no-retrieval baseline. For ontology, D is the benchmark analogue of A, but it should be described as an entire-ontology/reference condition because it supplies the proof engine with the whole parsed ontology context.

### Gold-free concrete reader-input diagnostics

No gold answer is printed below.

```text
dataset: HotpotQA
example_id: HotpotQA__test__5ae69cfe55429908198fa65f
question: Reeves Plains Power Station is a proposal from a company owned by who?
context source: data/raw/hotpot_qa/distractor/validation-00000-of-00001.parquet [id, context]
context size: 53 EvidenceUnits
first units:
  1. SENT::Condamine Power Station::0::Condamine Power Station is a 140 MW combined cycle power station near Miles on the western Darling Downs in Queensland, Australia.
  2. SENT::Condamine Power Station::1::The station is located 8 km east of Miles on the south side of the Warrego Highway.
  3. SENT::Condamine Power Station::2::The Condamine Power Station is owned by QGC Limited, a subsidiary of BG Group.
reader mode: text_reader_v1 / GPT-4.1-mini
```

```text
dataset: 2WikiMultiHopQA
example_id: 2WikiMultiHopQA__test__772cba800bb011ebab90acde48001122
question: Who is Hugh Cholmondeley, 1St Earl Of Cholmondeley's paternal grandmother?
context source: data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet [id, context]
context size: 41 EvidenceUnits
first units:
  1. SENT::Robert Cholmondeley, 1st Viscount Cholmondeley::0::Robert Cholmondeley, 1st Viscount Cholmondeley (died 22 May 1681) was an English peer.
  2. SENT::Robert Cholmondeley, 1st Viscount Cholmondeley::1::Lord Cholmondeley was the son of Hugh Cholmondeley and Mary Bodvile.
  3. SENT::Robert Cholmondeley, 1st Viscount Cholmondeley::2::Sir Hugh Cholmondeley of Cholmondeley was his grandfather and Robert Cholmondeley, 1st Earl of Leinster, his uncle.
reader mode: text_reader_v1 / GPT-4.1-mini
```

```text
dataset: FamilyOWL_2hop
example_id: FamilyOWL_2hop__g0__q0__2hop-Thing_alan_john_dowse_1936_alan_john_dowse_1936-alan_john_dowse_1936-rdf:type-Man-BIN__Is Alan John Dowse a man?__ASK WHERE { <http://www.example.com/genealogy.owl#alan_john_dowse_1936> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.example.com/genealogy.owl#Man> }
question: Is Alan John Dowse a man?
context source: data/raw/family/FamilyOWL_2hop.json [group 0, top-level OWL Context]
context size: 1,433 EvidenceUnits
first units:
  1. henrietta_sarah_green_1873 hasSister mary_kate_green_1865
  2. rebecca_cotton_1845 hasSister susanna_cotton_1836
  3. john_archer_1804 isFatherOf james_archer_1840
reader mode: ontology_proof_first_reader_v1; deterministic proof, then GPT-4.1-mini fallback
```

```text
dataset: pizza_100_2hop
example_id: pizza_100_2hop__g1__q0__2hop-CajunSpiceTopping_CajunSpiceTopping_dynamic_1_CajunSpiceTopping_dynamic_1-CajunSpiceTopping_dynamic_1-rdf:type-American-NEG-BIN__Is Cajun Spice Topping considered American?__ASK WHERE { <http://www.example.com/genealogy.owl#CajunSpiceTopping_dynamic_1> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.example.com/genealogy.owl#American> }
question: Is Cajun Spice Topping considered American?
context source: data/raw/pizza_100/pizza_100_2hop.json [group 1, top-level OWL Context]
context size: 192 EvidenceUnits
first units:
  1. PolloAdAstra SubClassOf NamedPizza
  2. PizzaTopping SubClassOf Food
  3. FunctionalObjectProperty(isBaseOf)
reader mode: ontology_proof_first_reader_v1; deterministic proof, then GPT-4.1-mini fallback
```

```text
dataset: OWL2Bench_2hop
example_id: OWL2Bench_2hop__g3__q0__2hop-Article_P23_P23-P23-rdf:type-Publication-BIN__Is there a publication identified as P23 in the genealogy records?__ASK WHERE { <http://www.example.com/genealogy.owl#P23> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.example.com/genealogy.owl#Publication> }
question: Is there a publication identified as P23 in the genealogy records?
context source: data/raw/owl2bench/OWL2Bench_2hop.json [group 3, top-level OWL Context]
context size: 465 EvidenceUnits
first units:
  1. OtherStaff SubClassOf SupportingStaff
  2. InverseObjectProperties(hasFullProfessor,isFullProfessorOf)
  3. College SubClassOf Organization
reader mode: ontology_proof_first_reader_v1; deterministic proof, then GPT-4.1-mini fallback
```

## 2. Ontology proof-first audit

The answer is **B**: every ontology configuration, including Full Context, calls `infer_owl_boolean_answer(metadata, selected_support)` first. Only a `None` result reaches GPT-4.1-mini. The deterministic procedure proves positive simple ASK queries using direct assertions plus its implemented subclass, subproperty, domain/range, inverse, and symmetric-property machinery. It does not produce deterministic FALSE; unresolved cases fall back to GPT.

The following figures are recomputed from the frozen prediction `answer_source` and the already-stored post-freeze per-example answer scores. “Proof” is `deterministic_owl_proof`; “GPT” is `gpt-4.1-mini_fallback`.

| Dataset | Condition | Proof n (%) | Proof EM/F1 | GPT n (%) | GPT EM/F1 |
|---|---|---:|---:|---:|---:|
| FamilyOWL_1hop | GNN k1 | 112 (24.2%) | 1.000/1.000 | 350 (75.8%) | 0.697/0.734 |
| FamilyOWL_1hop | GNN adaptive | 112 (24.2%) | 1.000/1.000 | 350 (75.8%) | 0.697/0.734 |
| FamilyOWL_1hop | SAGE k1 | 168 (36.4%) | 1.000/1.000 | 294 (63.6%) | 0.724/0.765 |
| FamilyOWL_1hop | SAGE adaptive | 169 (36.6%) | 1.000/1.000 | 293 (63.4%) | 0.730/0.771 |
| FamilyOWL_1hop | Full Context | 169 (36.6%) | 1.000/1.000 | 293 (63.4%) | 0.829/0.884 |
| FamilyOWL_2hop | GNN k1 | 32 (6.9%) | 1.000/1.000 | 430 (93.1%) | 0.460/0.472 |
| FamilyOWL_2hop | GNN adaptive | 52 (11.3%) | 1.000/1.000 | 410 (88.7%) | 0.541/0.558 |
| FamilyOWL_2hop | SAGE k1 | 60 (13.0%) | 1.000/1.000 | 402 (87.0%) | 0.470/0.483 |
| FamilyOWL_2hop | SAGE adaptive | 75 (16.2%) | 1.000/1.000 | 387 (83.8%) | 0.587/0.602 |
| FamilyOWL_2hop | Full Context | 169 (36.6%) | 1.000/1.000 | 293 (63.4%) | 0.594/0.670 |
| pizza_100_1hop | GNN k1 | 4 (3.4%) | 1.000/1.000 | 115 (96.6%) | 0.426/0.531 |
| pizza_100_1hop | GNN adaptive | 4 (3.4%) | 1.000/1.000 | 115 (96.6%) | 0.426/0.531 |
| pizza_100_1hop | SAGE k1 | 23 (19.3%) | 1.000/1.000 | 96 (80.7%) | 0.479/0.632 |
| pizza_100_1hop | SAGE adaptive | 23 (19.3%) | 1.000/1.000 | 96 (80.7%) | 0.479/0.632 |
| pizza_100_1hop | Full Context | 41 (34.5%) | 1.000/1.000 | 78 (65.5%) | 0.692/0.758 |
| pizza_100_2hop | GNN k1 | 13 (10.4%) | 1.000/1.000 | 112 (89.6%) | 0.446/0.492 |
| pizza_100_2hop | GNN adaptive | 13 (10.4%) | 1.000/1.000 | 112 (89.6%) | 0.464/0.525 |
| pizza_100_2hop | SAGE k1 | 15 (12.0%) | 1.000/1.000 | 110 (88.0%) | 0.445/0.482 |
| pizza_100_2hop | SAGE adaptive | 15 (12.0%) | 1.000/1.000 | 110 (88.0%) | 0.445/0.482 |
| pizza_100_2hop | Full Context | 39 (31.2%) | 1.000/1.000 | 86 (68.8%) | 0.802/0.832 |
| pizza_250_1hop | GNN k1 | 8 (5.4%) | 1.000/1.000 | 141 (94.6%) | 0.383/0.526 |
| pizza_250_1hop | GNN adaptive | 8 (5.4%) | 1.000/1.000 | 141 (94.6%) | 0.383/0.526 |
| pizza_250_1hop | SAGE k1 | 25 (16.8%) | 1.000/1.000 | 124 (83.2%) | 0.460/0.630 |
| pizza_250_1hop | SAGE adaptive | 25 (16.8%) | 1.000/1.000 | 124 (83.2%) | 0.460/0.630 |
| pizza_250_1hop | Full Context | 49 (32.9%) | 1.000/1.000 | 100 (67.1%) | 0.740/0.808 |
| pizza_250_2hop | GNN k1 | 7 (4.5%) | 1.000/1.000 | 150 (95.5%) | 0.360/0.393 |
| pizza_250_2hop | GNN adaptive | 7 (4.5%) | 1.000/1.000 | 150 (95.5%) | 0.367/0.421 |
| pizza_250_2hop | SAGE k1 | 11 (7.0%) | 1.000/1.000 | 146 (93.0%) | 0.377/0.407 |
| pizza_250_2hop | SAGE adaptive | 11 (7.0%) | 1.000/1.000 | 146 (93.0%) | 0.384/0.419 |
| pizza_250_2hop | Full Context | 50 (31.8%) | 1.000/1.000 | 107 (68.2%) | 0.729/0.789 |
| OWL2Bench_1hop | GNN k1 | 91 (24.2%) | 1.000/1.000 | 285 (75.8%) | 0.663/0.728 |
| OWL2Bench_1hop | GNN adaptive | 91 (24.2%) | 1.000/1.000 | 285 (75.8%) | 0.663/0.728 |
| OWL2Bench_1hop | SAGE k1 | 121 (32.2%) | 1.000/1.000 | 255 (67.8%) | 0.620/0.689 |
| OWL2Bench_1hop | SAGE adaptive | 121 (32.2%) | 1.000/1.000 | 255 (67.8%) | 0.620/0.691 |
| OWL2Bench_1hop | Full Context | 121 (32.2%) | 1.000/1.000 | 255 (67.8%) | 0.514/0.600 |
| OWL2Bench_2hop | GNN k1 | 65 (16.3%) | 1.000/1.000 | 334 (83.7%) | 0.509/0.558 |
| OWL2Bench_2hop | GNN adaptive | 65 (16.3%) | 1.000/1.000 | 334 (83.7%) | 0.509/0.560 |
| OWL2Bench_2hop | SAGE k1 | 103 (25.8%) | 1.000/1.000 | 296 (74.2%) | 0.561/0.610 |
| OWL2Bench_2hop | SAGE adaptive | 103 (25.8%) | 1.000/1.000 | 296 (74.2%) | 0.561/0.610 |
| OWL2Bench_2hop | Full Context | 106 (26.6%) | 1.000/1.000 | 293 (73.4%) | 0.594/0.707 |

Interpretation: the Full Context advantage is not solely a larger GPT prompt. On most ontology datasets it also raises the deterministic resolution rate substantially. The 100% proof-subset result is expected from the one-sided procedure: it emits only when it finds a positive entailment and otherwise delegates; it is not evidence that arbitrary positive and negative ontology questions were perfectly classified by a complete reasoner.

## 3. Leakage audit

### Static trace

- Text Full Context uses a Parquet column projection of only `id` and `context`. `answer`, `supporting_facts`, and 2Wiki `evidences` are not decoded.
- Ontology Full Context's selective scanner decodes only `OWL Context`. The raw object also contains `QAs` with answer and explanation fields, but that whole value is skipped without JSON decoding by the Full Context loader.
- Candidate rows are not used to assemble Full Context. They provide a whitelisted metadata projection: `example_id`, dataset/split, question, hop/answer type, SPARQL query/task type, source, and group/QA indexes.
- Generation validates the frozen input hash, copies `selected_support` without reconstruction, and sends only question plus EvidenceUnits to the text prompt. The ontology predictor receives the same support plus the formal query metadata for proof parsing; the GPT fallback prompt still contains only question plus EvidenceUnits.
- Evaluation opens answer/support gold only after it validates the complete prediction count and frozen prediction-file SHA-256.

### Dynamic checks over the frozen artifacts

- All 14 manifest-covered members of the baseline and GNN/SAGE end-to-end directories currently match both their recorded byte sizes and SHA-256 hashes.
- All 12,747 frozen baseline reader inputs were scanned. No top-level or nested metadata key matched gold/answer/label/target, `best_set_f1_to_gold`, `rank_target`, `supporting_facts`, or `evidences`.
- The 4,249-generation cohort is complete for all three baseline conditions; no selected context is empty.
- Every ontology frozen input has a populated `sparql_query`; therefore the proof parser's legacy fallback that can parse a query from `example_id` is never taken.

### Residual boundary notes

No effective gold leakage was found, but two defense-in-depth issues should be documented:

1. Some ontology `example_id` strings encode `NEG` (the polarity label): 148/462 in each FamilyOWL split, 37/119 and 44/125 in pizza_100, 44/149 and 52/157 in pizza_250, and 126/376 and 141/399 in OWL2Bench. The ID is used for joining/persistence and is present in metadata, but it is not included in either GPT prompt. The deterministic reader uses the populated `sparql_query` and never consults the ID fallback in these artifacts. Thus the marker did not influence the executed predictions, but future runners should not expose label-bearing IDs to predictor code.
2. `FORBIDDEN_GENERATION_FIELDS` explicitly names the main answer/support fields but does not itself enumerate every requested synonym such as `rank_target`, `best_set_f1_to_gold`, or generic `label`. The metadata whitelist and the actual frozen input eliminate those fields here, so this is not current leakage; it is a guard-coverage limitation.

## 4. Context-size comparison

Cells are **mean / median / p90 EvidenceUnits**, measured from the exact frozen reader support lists. No condition has an empty context.

| Dataset | Lexical k1 | GNN-RAG k1 | GNN k1 | GNN adaptive | SAGE k1 | SAGE adaptive | Full Context |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 2.770/3/3 | 1.000/1/1 | 2.940/3/3 | 3.238/3/5 | 2.954/3/3 | 3.214/3/5 | 40.845/40/54 |
| 2WikiMultiHopQA | 2.677/3/3 | 1.000/1/1 | 2.782/3/3 | 3.062/3/4 | 2.789/3/3 | 3.014/3/4 | 32.274/28/51 |
| FamilyOWL_1hop | 1.154/1/2 | 1.000/1/1 | 2.710/3/3 | 2.710/3/3 | 2.236/2/3 | 2.255/2/3 | 99.039/97/105 |
| FamilyOWL_2hop | 1.043/1/1 | 1.000/1/1 | 1.725/1/3 | 2.576/2/3 | 1.706/2/3 | 3.361/3/5 | 1433.000/1433/1433 |
| pizza_100_1hop | 1.538/2/2 | 1.000/1/1 | 2.857/3/3 | 2.857/3/3 | 2.176/2/3 | 2.176/2/3 | 126.235/125/128 |
| pizza_100_2hop | 1.456/1/2 | 1.000/1/1 | 1.816/1/3 | 2.144/2/3 | 1.528/1/2 | 1.656/2/2 | 536.944/571/675 |
| pizza_250_1hop | 1.651/2/2 | 1.000/1/1 | 2.805/3/3 | 2.812/3/3 | 2.221/2/3 | 2.221/2/3 | 127.255/125/128 |
| pizza_250_2hop | 1.420/1/2 | 1.000/1/1 | 1.529/1/3 | 2.229/2/3 | 1.446/1/2 | 1.637/2/2 | 718.248/714/897 |
| OWL2Bench_1hop | 2.024/2/3 | 1.000/1/1 | 2.077/2/3 | 2.088/2/3 | 2.253/2/3 | 2.258/2/3 | 386.484/387/391 |
| OWL2Bench_2hop | 1.474/1/2 | 1.000/1/1 | 2.393/3/3 | 2.429/3/3 | 2.206/2/3 | 2.221/2/3 | 718.895/762/870 |

This scale difference is sufficient by itself to make Full Context an upper-context reference rather than a directly matched competitor. The proof-first difference further strengthens that conclusion.

## 5. GNN-RAG handoff audit

### Frozen identity and reader input

The baseline runner does the following for each example:

1. reads `native_ranked_entities` from the clean, globally frozen GNN-RAG prediction file;
2. requires the first record to have rank 1;
3. maps its normalized evidence-node identity back to the original Generator-D `subgraph_units` string;
4. supplies exactly `[mapped_rank_1_unit]` to the common reader; and
5. hash-locks that one-element list and checks it again at prediction/evaluation time.

Dynamic results:

- Rank-1 node identity equals the supplied EvidenceUnit identity: **4,249/4,249**.
- Empty rank-1 reader contexts: **0/4,249**.
- Frozen-input versus prediction support mismatches: **0/4,249**.
- For the three requested focus datasets, candidate-map verification found no missing or ambiguous normalized rank-1 identity: HotpotQA 1,000/1,000 exact, 2Wiki 1,000/1,000 exact, pizza_100_2hop 125/125 exact.
- The reader prompt builders enumerate the entire supplied list and contain no OpenAI-path truncation or reordering. Stored supports are JSON-serialized in order and canonical-hashed. No local serialization/truncation defect was found.

### Why it differs from the native saved context

The upstream frozen GNN-RAG file is not itself a k1 file. Native inference retains ranked non-seed entities until cumulative probability exceeds `eps=0.95`, then constructs shortest paths from question seed nodes to **all** retained entities. Its `retrieved_evidence_units` therefore has these mean sizes:

| Dataset | Current top-1 reader units | Native all-`eps` path units | Top-1 equals full native list |
|---|---:|---:|---:|
| HotpotQA | 1.000 | 3.359 | 66/1,000 |
| 2WikiMultiHopQA | 1.000 | 3.002 | 200/1,000 |
| pizza_100_2hop | 1.000 | 162.168 | 0/125 |

In every one of these examples, the top-1 unit is a subset of the native all-`eps` context. The current runner does not accidentally pick the wrong unit; it intentionally collapses native GNN-RAG to rank 1. A concrete Hotpot example illustrates why a multihop answer can then fail:

```text
example_id: HotpotQA__test__5ae69cfe55429908198fa65f
top-1 reader unit: Reeves Plains Power Station ... is a proposal from Alinta Energy ...
native all-eps path also contains: Alinta Energy ... is owned by ... Chow Tai Fook Enterprises.
```

The one-unit k1 input identifies the intermediate company but omits the ownership sentence needed to answer the question. That is poor top-1 evidence sufficiency, not an ID/text handoff failure.

For the focused support results:

| Dataset | Current top-1 Support F1 | Native all-`eps` path Support F1 | Native all-`eps` recall |
|---|---:|---:|---:|
| HotpotQA | 0.310436 | 0.358439 | 0.427402 |
| 2WikiMultiHopQA | 0.280486 | 0.338560 | 0.356650 |
| pizza_100_2hop | 0.000000 | 0.036475 | 0.972840 |

The pizza result is especially diagnostic. Native GNN-RAG admits about 162 EvidenceUnits on average and nearly always includes the gold support somewhere, but with extremely low precision. The top-ranked single axiom alone has zero support F1 on the 81 support-evaluable examples. This is a genuine rank-1 ranking/sufficiency weakness. It is not evidence that the native all-`eps` retriever has zero recall, nor is it a reader serialization bug.

### Reporting constraint

The saved baseline is scientifically interpretable if named “GNN-RAG top-1 EvidenceUnit” or equivalent. Calling it simply “GNN-RAG” without the k1/top-1 qualifier would imply the upstream native shortest-path context and would be misleading. No replacement is implemented or recommended within this audit's read-only scope.

## 6. Final conclusions

1. Full Context is built from legitimate inference-time benchmark context, not candidate unions or gold-labeled artifacts.
2. No executed gold-answer/support leakage was detected. The label-bearing ontology IDs are a dormant boundary concern but do not reach prompts or influence proof parsing in the frozen cohort.
3. Ontology Full Context is proof-first over the full parsed ontology, not GPT-only. Its results therefore combine an increased evidence budget with increased deterministic proof coverage.
4. Full Context should be reported as a reference/upper-context condition and kept out of claims of directly matched retrieval-system superiority.
5. GNN-RAG k1 has a correct frozen identity handoff. Its weak results are valid for top-1 EvidenceUnit retrieval, while the larger native all-`eps` GNN-RAG context is a separate condition that was not used by this reader run.
