import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import re
import string
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.build_subgraph_training_data import (
    beam_connected_subgraphs,
    materialize_retrieval_rows,
)
from data_processing.text_kg_constructor import (
    KGConstructionConfig,
    LLMKGConstructor,
    construct_text_kg,
    normalize_evidence_triples,
    normalize_relation,
    select_context_sentences,
)


# ============================================================
# Text utilities
# ============================================================

_ARTICLES = {"a", "an", "the"}


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return [t for t in normalize_text(text).split() if t and t not in _ARTICLES]


def token_set(text: str) -> Set[str]:
    return set(tokenize(text))


def safe_title(title: str) -> str:
    title = str(title).replace("\n", " ").replace("\t", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title


def clean_sentence(sent: str) -> str:
    sent = str(sent).replace("\n", " ").replace("\t", " ")
    sent = re.sub(r"\s+", " ", sent).strip()
    return sent


def make_kg_triple_unit(subject: str, predicate: str, obj: str) -> str:
    """
    Structured KG fact used by both text-derived and ontology-native graphs.
    """
    return (
        f"KG::{clean_sentence(subject)}::"
        f"{normalize_relation(predicate)}::{clean_sentence(obj)}"
    )


def evidence_triple_units(evidences: List[List[str]]) -> List[str]:
    units: List[str] = []
    seen = set()
    for ev in evidences:
        if not isinstance(ev, list) or len(ev) < 3:
            continue
        unit = make_kg_triple_unit(ev[0], ev[1], ev[2])
        if unit not in seen:
            seen.add(unit)
            units.append(unit)
    return units


def parse_kg_triple_unit(unit: str) -> Tuple[str, str, str] | None:
    parts = str(unit).split("::", 3)
    if len(parts) == 4 and parts[0] == "KG":
        return parts[1], parts[2], parts[3]
    return None


def kg_triple_key(unit: str) -> Tuple[str, str, str] | None:
    triple = parse_kg_triple_unit(unit)
    if triple is None:
        return None
    subject, predicate, obj = triple
    return (
        normalize_text(subject),
        normalize_relation(predicate),
        normalize_text(obj),
    )


def get_first_present(mapping: Dict[str, Any], keys: List[str], default=None):
    """
    Safely returns the first existing non-None value from a dict.

    Avoids using `or` on numpy arrays, which raises:
    ValueError: The truth value of an array with more than one element is ambiguous.
    """
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def to_python_list(x: Any) -> List[Any]:
    """
    Converts numpy arrays / pandas objects / tuples into normal Python lists.
    """
    if x is None:
        return []

    if isinstance(x, list):
        return x

    if isinstance(x, tuple):
        return list(x)

    # numpy array or pandas array
    if hasattr(x, "tolist"):
        y = x.tolist()
        if isinstance(y, list):
            return y
        return [y]

    return [x]


# ============================================================
# Loading and normalizing 2Wiki records
# ============================================================


def load_2wiki_file(path: str) -> List[Dict[str, Any]]:
    """
    Loads 2WikiMultiHopQA from:
      - a .parquet file
      - a .json file
      - a .jsonl file
      - a directory containing parquet files

    If a directory is given, all parquet files under it are loaded.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"2Wiki path not found: {p}")

    if p.is_dir():
        parquet_files = sorted(p.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under directory: {p}")

        all_rows: List[Dict[str, Any]] = []
        for pq in parquet_files:
            # Avoid accidentally mixing validation/test files into train
            # when user passes a broad directory. We still load all parquet
            # files in the directory intentionally.
            all_rows.extend(load_2wiki_file(str(pq)))
        return all_rows

    suffix = p.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(p)
        return [normalize_2wiki_record(r) for r in df.to_dict(orient="records")]

    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {p}, got {type(data)}")
        return [normalize_2wiki_record(r) for r in data]

    if suffix == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(normalize_2wiki_record(json.loads(line)))
        return rows

    raise ValueError(f"Unsupported file type: {p}")


def normalize_2wiki_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes HuggingFace 2WikiMultiHopQA rows.

    Expected input columns include:
      id, question, answer, type, evidences, context

    context usually looks like:
      {
        "title": array([...]),
        "sentences": array([array([...]), ...])
      }
    """
    out = dict(row)

    if "_id" not in out:
        out["_id"] = str(out.get("id", ""))

    # Normalize context to:
    #   [[title, [sent0, sent1, ...]], ...]
    context = out.get("context", {})

    if isinstance(context, dict):
        titles = to_python_list(get_first_present(context, ["title", "titles"], []))
        sentences = to_python_list(
            get_first_present(context, ["sentences", "sentence"], [])
        )

        normalized_context = []
        for title, sents in zip(titles, sentences):
            normalized_context.append(
                [safe_title(title), [clean_sentence(s) for s in to_python_list(sents)]]
            )

        out["context"] = normalized_context

    elif isinstance(context, list):
        normalized_context = []
        for item in context:
            if isinstance(item, dict):
                title = item.get("title", "")
                sents = item.get("sentences", item.get("sentence", []))
                normalized_context.append(
                    [
                        safe_title(title),
                        [clean_sentence(s) for s in to_python_list(sents)],
                    ]
                )
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                title, sents = item
                normalized_context.append(
                    [
                        safe_title(title),
                        [clean_sentence(s) for s in to_python_list(sents)],
                    ]
                )
        out["context"] = normalized_context

    else:
        out["context"] = []

    # Sentence-level supporting-fact annotations are deliberately excluded.
    # 2Wiki evidence triples are the gold reasoning target for this pipeline.
    out.pop("supporting_facts", None)
    out["evidences"] = normalize_evidence_triples(out.get("evidences", []))

    return out


# ============================================================
# Context / support extraction
# ============================================================


def flatten_context(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for page in example.get("context", []):
        if not isinstance(page, list) or len(page) != 2:
            continue

        title, sentences = page
        title = safe_title(title)

        for idx, sent in enumerate(to_python_list(sentences)):
            sent_clean = clean_sentence(sent)
            if not sent_clean:
                continue

            records.append(
                {
                    "title": title,
                    "sent_idx": idx,
                    "sentence": sent_clean,
                }
            )

    return records


# ============================================================
# Candidate generation
# ============================================================


def lexical_overlap_score(query: str, text: str) -> float:
    q = token_set(query)
    t = token_set(text)
    if not q:
        return 0.0
    return len(q & t) / max(len(q), 1)


def select_kg_pool(
    question: str,
    kg_units: List[str],
    max_candidate_units: int,
) -> List[str]:
    """Select inference-time KG facts using question relevance and connectivity."""
    if max_candidate_units <= 0 or len(kg_units) <= max_candidate_units:
        return list(kg_units)

    signatures = []
    for unit in kg_units:
        triple = parse_kg_triple_unit(unit)
        entities = set()
        if triple is not None:
            entities = {normalize_text(triple[0]), normalize_text(triple[2])}
        signatures.append(entities)

    degrees = []
    for i, entities in enumerate(signatures):
        degree = sum(
            bool(entities & other) for j, other in enumerate(signatures) if i != j
        )
        degrees.append(degree)

    scored = [
        (
            lexical_overlap_score(question, unit) + 0.02 * min(degrees[i], 10),
            unit,
        )
        for i, unit in enumerate(kg_units)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [unit for _, unit in scored[:max_candidate_units]]


def align_gold_evidence_to_candidate_kg(
    gold_units: List[str],
    candidate_units: List[str],
) -> Tuple[List[str], float]:
    """
    Align gold annotations to inference-time KG spelling without adding facts.

    An unmatched gold fact remains in the evaluation target, making extraction
    coverage visible instead of silently inserting the target into the graph.
    """
    by_key = {
        key: unit
        for unit in candidate_units
        if (key := kg_triple_key(unit)) is not None
    }
    aligned = []
    matched = 0
    for gold in gold_units:
        key = kg_triple_key(gold)
        if key is not None and key in by_key:
            aligned.append(by_key[key])
            matched += 1
        else:
            aligned.append(gold)
    coverage = matched / max(len(gold_units), 1)
    return aligned, coverage


def infer_answer_type(answer: str) -> str:
    ans = normalize_text(answer)
    if ans in {"yes", "no"}:
        return "BIN"
    return "OPEN"


# ============================================================
# Row construction
# ============================================================


def build_rows_for_example(
    example: Dict[str, Any],
    split_name: str,
    max_sentences_per_example: int,
    max_subgraph_size: int,
    max_candidates_per_question: int,
    kg_config: KGConstructionConfig,
    llm_kg_constructor: LLMKGConstructor | None,
    kg_cache: Dict[str, Dict[str, Any]] | None,
    kg_cache_path: Path | None,
    kg_cache_lock: Any,
    seed: int,
    candidate_beam_width: int = 96,
) -> List[Dict[str, Any]]:
    raw_id = str(get_first_present(example, ["_id", "id"], f"no_id_{seed}"))
    example_id = f"2WikiMultiHopQA__{split_name}__{raw_id}"
    question = str(example.get("question", ""))
    answer = str(example.get("answer", ""))
    raw_evidences = normalize_evidence_triples(example.get("evidences", []))

    sentence_records = flatten_context(example)
    cached_kg = kg_cache.get(example_id) if kg_cache is not None else None
    if cached_kg and cached_kg.get("cache_signature") != kg_config.cache_signature:
        cached_kg = None
    cached_method = str(cached_kg.get("construction_method", "")) if cached_kg else ""
    if cached_kg and "provided" not in cached_method:
        kg_triples = cached_kg.get("kg_triples", [])
        kg_construction_method = cached_method or "cache"
    else:
        kg_triples, kg_construction_method = construct_text_kg(
            sentence_records=sentence_records,
            question=question,
            config=kg_config,
            llm_constructor=llm_kg_constructor,
        )
        if llm_kg_constructor is not None and kg_cache is not None:
            cache_row = {
                "example_id": example_id,
                "kg_triples": kg_triples,
                "construction_method": kg_construction_method,
                "cache_signature": kg_config.cache_signature,
            }
            if kg_cache_lock is not None:
                with kg_cache_lock:
                    kg_cache[example_id] = cache_row
                    append_kg_cache_row(kg_cache_path, cache_row)
            else:
                kg_cache[example_id] = cache_row
                append_kg_cache_row(kg_cache_path, cache_row)
    constructed_kg_units = evidence_triple_units(kg_triples)
    candidate_pool = select_kg_pool(
        question=question,
        kg_units=constructed_kg_units,
        max_candidate_units=max_sentences_per_example,
    )
    if not candidate_pool:
        return []

    raw_gold_evidence_units = evidence_triple_units(raw_evidences)
    gold_units, gold_kg_coverage = align_gold_evidence_to_candidate_kg(
        raw_gold_evidence_units,
        candidate_pool,
    )
    candidate_keys = {
        key for unit in candidate_pool if (key := kg_triple_key(unit)) is not None
    }
    matched_gold_units = [
        unit
        for unit in raw_gold_evidence_units
        if kg_triple_key(unit) in candidate_keys
    ]
    missing_gold_units = [
        unit
        for unit in raw_gold_evidence_units
        if kg_triple_key(unit) not in candidate_keys
    ]
    selected_context_records = select_context_sentences(
        question=question,
        sentence_records=sentence_records,
        limit=kg_config.max_context_sentences,
    )
    gold_reference_sets = [gold_units] if gold_units else []

    candidate_paths = beam_connected_subgraphs(
        candidate_units=candidate_pool,
        question=question,
        sparql_query="",
        min_subgraph_size=1,
        max_subgraph_size=max_subgraph_size,
        beam_width=candidate_beam_width,
        max_candidate_subgraphs=max_candidates_per_question,
    )
    candidates = [list(candidate) for candidate in sorted(candidate_paths)]

    answer_type = infer_answer_type(answer)
    rows = materialize_retrieval_rows(
        candidate_units=candidate_pool,
        candidate_subgraphs=candidates,
        gold_explanations=gold_reference_sets,
        base_row={
            "example_id": example_id,
            "dataset": "2WikiMultiHopQA",
            "hop": "2hop",
            "answer_type": answer_type,
            "question": question,
            "answer": answer,
            "evidence_unit_type": "kg_triple",
            "kg_construction_method": kg_construction_method,
            "candidate_kg_units": candidate_pool,
            # Candidate reasoning path.
            "graph_context_units": [],
            # Triple-level gold reasoning supervision.
            "gold_explanations": gold_reference_sets,
            "gold_units": gold_units,
            "raw_gold_evidence_units": raw_gold_evidence_units,
            "gold_kg_coverage": float(gold_kg_coverage),
            "matched_gold_kg_units": matched_gold_units,
            "missing_gold_kg_units": missing_gold_units,
            "context_sentence_count": len(sentence_records),
            "selected_context_sentence_count": len(selected_context_records),
            "context_page_count": len(
                {
                    normalize_text(record.get("title", ""))
                    for record in sentence_records
                    if normalize_text(record.get("title", ""))
                }
            ),
            "kg_cache_signature": kg_config.cache_signature,
        },
    )
    for candidate_index, row in enumerate(rows):
        row["row_id"] = f"{example_id}__cand{candidate_index}"

    return rows


# ============================================================
# Splitting / writing
# ============================================================


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_kg_cache(path: Path | None) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if path is None or not path.exists():
        return cache
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                example_id = row.get("example_id")
                if example_id:
                    cache[str(example_id)] = row
    return cache


def append_kg_cache_row(path: Path | None, row: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def limit_examples(
    examples: List[Dict[str, Any]], max_examples: int, seed: int
) -> List[Dict[str, Any]]:
    if not max_examples or max_examples <= 0:
        return examples
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    return shuffled[:max_examples]


def load_selection_manifest(path: Path | None) -> Dict[str, List[str]] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    splits = manifest.get("splits", {})
    selected = {}
    for split in ("train", "dev", "test"):
        details = splits.get(split, {})
        ids = details.get("selected_example_ids")
        if not isinstance(ids, list):
            raise ValueError(
                f"Selection manifest {path} has no selected IDs for {split}"
            )
        selected[split] = [str(example_id) for example_id in ids]
    return selected


def select_manifest_examples(
    examples: List[Dict[str, Any]],
    split_name: str,
    selected_example_ids: List[str],
) -> List[Dict[str, Any]]:
    by_id = {}
    for example in examples:
        raw_id = str(get_first_present(example, ["_id", "id"], ""))
        example_id = f"2WikiMultiHopQA__{split_name}__{raw_id}"
        if example_id in by_id:
            raise ValueError(f"Duplicate 2Wiki source ID: {example_id}")
        by_id[example_id] = example
    missing = [
        example_id for example_id in selected_example_ids if example_id not in by_id
    ]
    if missing:
        raise ValueError(
            f"Selection manifest contains {len(missing)} IDs absent from the "
            f"{split_name} source, including: {missing[:3]}"
        )
    return [by_id[example_id] for example_id in selected_example_ids]


def split_train_dev_from_train(
    examples: List[Dict[str, Any]],
    dev_ratio: float,
    seed: int,
    max_train_examples: int = 0,
    max_dev_examples: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    total_requested = None
    if max_train_examples and max_train_examples > 0:
        dev_guess = (
            max_dev_examples
            if max_dev_examples > 0
            else int(max_train_examples * dev_ratio)
        )
        total_requested = max_train_examples + dev_guess

    if total_requested:
        shuffled = shuffled[: min(total_requested, len(shuffled))]

    n_dev = int(round(len(shuffled) * dev_ratio))

    if max_dev_examples and max_dev_examples > 0:
        n_dev = min(n_dev, max_dev_examples)

    dev = shuffled[:n_dev]
    train = shuffled[n_dev:]

    if max_train_examples and max_train_examples > 0:
        train = train[:max_train_examples]

    return train, dev


def build_split_rows(
    examples: List[Dict[str, Any]],
    split_name: str,
    max_sentences_per_example: int,
    max_subgraph_size: int,
    max_candidates_per_question: int,
    kg_config: KGConstructionConfig,
    llm_kg_constructor: LLMKGConstructor | None,
    kg_construction_workers: int,
    kg_cache_path: Path | None,
    seed: int,
    candidate_beam_width: int = 96,
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    start_time = time.time()
    uses_llm = llm_kg_constructor is not None
    kg_cache = load_kg_cache(kg_cache_path) if uses_llm else None
    kg_cache_lock = threading.Lock() if uses_llm else None
    print(
        f"  {split_name}: starting {len(examples)} examples "
        f"(kg_backend={kg_config.backend}, "
        f"workers={kg_construction_workers if uses_llm else 1})",
        flush=True,
    )
    if uses_llm and kg_cache_path is not None:
        print(
            f"  {split_name}: KG cache has {len(kg_cache or {})} examples at {kg_cache_path}",
            flush=True,
        )

    def print_progress(done_count: int, total_count: int) -> None:
        elapsed = max(time.time() - start_time, 1e-6)
        rate = done_count / elapsed
        remaining = max(total_count - done_count, 0)
        eta_seconds = remaining / rate if rate > 0 else 0.0
        print(
            f"  {split_name}: built {done_count}/{total_count} examples "
            f"({rate:.2f} ex/s, elapsed {elapsed / 60:.1f} min, "
            f"ETA {eta_seconds / 60:.1f} min)",
            flush=True,
        )

    if llm_kg_constructor is not None and kg_construction_workers > 1:
        indexed_results: List[Tuple[int, List[Dict[str, Any]]]] = []

        def build_one(
            idx_ex: Tuple[int, Dict[str, Any]],
        ) -> Tuple[int, List[Dict[str, Any]]]:
            idx, ex = idx_ex
            local_constructor = LLMKGConstructor(kg_config)
            return idx, build_rows_for_example(
                example=ex,
                split_name=split_name,
                max_sentences_per_example=max_sentences_per_example,
                max_subgraph_size=max_subgraph_size,
                max_candidates_per_question=max_candidates_per_question,
                kg_config=kg_config,
                llm_kg_constructor=local_constructor,
                kg_cache=kg_cache,
                kg_cache_path=kg_cache_path,
                kg_cache_lock=kg_cache_lock,
                seed=seed + idx,
                candidate_beam_width=candidate_beam_width,
            )

        with ThreadPoolExecutor(max_workers=kg_construction_workers) as pool:
            futures = [
                pool.submit(build_one, (idx, ex)) for idx, ex in enumerate(examples)
            ]
            for done_count, future in enumerate(as_completed(futures), start=1):
                indexed_results.append(future.result())
                if done_count % 5 == 0 or done_count == len(futures):
                    print_progress(done_count, len(futures))

        for _, rows in sorted(indexed_results, key=lambda item: item[0]):
            all_rows.extend(rows)
        return all_rows

    for idx, ex in enumerate(examples):
        rows = build_rows_for_example(
            example=ex,
            split_name=split_name,
            max_sentences_per_example=max_sentences_per_example,
            max_subgraph_size=max_subgraph_size,
            max_candidates_per_question=max_candidates_per_question,
            kg_config=kg_config,
            llm_kg_constructor=llm_kg_constructor,
            kg_cache=kg_cache,
            kg_cache_path=kg_cache_path,
            kg_cache_lock=kg_cache_lock,
            seed=seed + idx,
            candidate_beam_width=candidate_beam_width,
        )
        all_rows.extend(rows)
        progress_every = 5 if uses_llm else 25
        if (idx + 1) % progress_every == 0 or idx + 1 == len(examples):
            print_progress(idx + 1, len(examples))

    return all_rows


def summarize_rows(name: str, rows: List[Dict[str, Any]]) -> None:
    example_ids = {r["example_id"] for r in rows}
    positives = sum(int(r.get("label", 0)) for r in rows)
    total = len(rows)
    bin_count = len({r["example_id"] for r in rows if r.get("answer_type") == "BIN"})
    open_count = len({r["example_id"] for r in rows if r.get("answer_type") == "OPEN"})
    coverage_by_example = {}
    for row in rows:
        coverage_by_example.setdefault(
            row["example_id"], float(row.get("gold_kg_coverage", 0.0))
        )

    print(f"{name}:")
    print(f"  rows:       {total}")
    print(f"  examples:   {len(example_ids)}")
    print(f"  BIN ex:     {bin_count}")
    print(f"  OPEN ex:    {open_count}")
    print(f"  positives:  {positives}")
    print(f"  negatives:  {total - positives}")
    if total:
        print(f"  pos rate:   {positives / total:.4f}")
    if coverage_by_example:
        mean_coverage = sum(coverage_by_example.values()) / len(coverage_by_example)
        complete_coverage = sum(v >= 1.0 for v in coverage_by_example.values())
        print(f"  gold KG coverage: {mean_coverage:.4f} mean")
        print(
            f"  complete KG extraction: {complete_coverage}/{len(coverage_by_example)}"
        )


def count_labeled_examples(examples: List[Dict[str, Any]]) -> int:
    return sum(
        1 for example in examples if example.get("answer") and example.get("evidences")
    )


def require_nonempty_split(
    split_name: str,
    rows: List[Dict[str, Any]],
    selected_examples: List[Dict[str, Any]],
    source_files: List[str],
) -> None:
    if rows:
        return

    labeled_count = count_labeled_examples(selected_examples)
    source_text = ", ".join(source_files)
    message = (
        f"No rows were built for 2Wiki {split_name} split from {source_text}. "
        f"Selected examples: {len(selected_examples)}; examples with answer and "
        f"gold evidence triples: {labeled_count}."
    )
    if split_name == "test" and labeled_count == 0:
        message += (
            " The public 2Wiki test parquet is unlabeled in this distribution. "
            "It can still be used for inference when context-to-KG extraction "
            "produces candidate triples; use validation for supervised metrics."
        )
    raise ValueError(message)


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-file", type=str, nargs="+", required=True)
    parser.add_argument("--dev-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="data/2WikiMultiHopQA")

    parser.add_argument("--dev-ratio-from-train", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-train-examples", type=int, default=1000)
    parser.add_argument("--max-dev-examples", type=int, default=200)
    parser.add_argument("--max-test-examples", type=int, default=300)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=None,
        help=(
            "A gnn_dev_sample_v1 manifest whose exact example IDs override "
            "--max-*-examples. Candidate rows are rebuilt from the raw records."
        ),
    )

    parser.add_argument(
        "--max-kg-candidate-triples",
        "--max-sentences-per-example",
        dest="max_kg_candidate_triples",
        type=int,
        default=30,
        help=(
            "Maximum context-derived KG facts retained per example. The old "
            "--max-sentences-per-example name remains as a deprecated alias."
        ),
    )
    parser.add_argument("--max-subgraph-size", type=int, default=4)
    parser.add_argument("--max-candidates-per-question", type=int, default=512)
    parser.add_argument(
        "--candidate-beam-width",
        type=int,
        default=96,
        help=("Beam width for the shared FamilyOWL/2Wiki connected-subgraph composer."),
    )
    parser.add_argument(
        "--max-kg-bridge-triples",
        type=int,
        default=64,
        help=("Maximum context-derived KG triples constructed per example."),
    )
    parser.add_argument(
        "--kg-construction-backend",
        choices=[
            "deterministic",
            "llm",
            "llm_with_title_bridges",
        ],
        default="deterministic",
        help=(
            "How to build candidate KG triples from question/context only. "
            "llm extracts triples from the example context. "
            "llm_with_title_bridges combines context-only LLM triples with "
            "deterministic title co-mention bridges."
        ),
    )
    parser.add_argument("--kg-construction-model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--kg-max-context-sentences", type=int, default=20)
    parser.add_argument("--kg-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--kg-request-timeout", type=float, default=90.0)
    parser.add_argument("--kg-max-retries", type=int, default=4)
    parser.add_argument("--kg-retry-initial-sleep", type=float, default=5.0)
    parser.add_argument(
        "--kg-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-split LLM KG cache JSONL files. Defaults to "
            "<output-dir>/kg_cache for LLM KG backends."
        ),
    )
    parser.add_argument(
        "--kg-construction-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel LLM KG extraction workers. Use 1 for serial "
            "execution; 4-8 can speed up large builds if rate limits allow."
        ),
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kg_cache_dir = args.kg_cache_dir or (out_dir / "kg_cache")
    kg_config = KGConstructionConfig(
        backend=args.kg_construction_backend,
        model=args.kg_construction_model,
        max_triples=args.max_kg_bridge_triples,
        max_context_sentences=args.kg_max_context_sentences,
        sleep_seconds=args.kg_sleep_seconds,
        request_timeout=args.kg_request_timeout,
        max_retries=args.kg_max_retries,
        retry_initial_sleep=args.kg_retry_initial_sleep,
    )
    llm_kg_constructor = (
        LLMKGConstructor(kg_config)
        if args.kg_construction_backend in {"llm", "llm_with_title_bridges"}
        else None
    )

    print("Loading 2WikiMultiHopQA...")
    train_raw_all = []
    for train_file in args.train_file:
        train_raw_all.extend(load_2wiki_file(train_file))
    dev_raw_all = load_2wiki_file(args.dev_file)
    test_files = args.test_file if args.test_file is not None else [args.dev_file]
    test_raw_all = []
    for test_file in test_files:
        test_raw_all.extend(load_2wiki_file(test_file))

    print(f"Raw train examples: {len(train_raw_all)}")
    print(f"Raw dev examples:   {len(dev_raw_all)}")
    print(f"Raw test examples:  {len(test_raw_all)}")

    selection = load_selection_manifest(args.selection_manifest)
    if selection is not None:
        train_examples = select_manifest_examples(
            train_raw_all, "train", selection["train"]
        )
        dev_examples = select_manifest_examples(dev_raw_all, "dev", selection["dev"])
        test_examples = select_manifest_examples(
            test_raw_all, "test", selection["test"]
        )
    else:
        # Train, dev, and test are sampled from their respective source files. If
        # --test-file is omitted, the provided dev/validation file is used as test.
        train_examples = limit_examples(
            examples=train_raw_all,
            max_examples=args.max_train_examples,
            seed=args.seed,
        )
        dev_examples = limit_examples(
            examples=dev_raw_all,
            max_examples=args.max_dev_examples,
            seed=args.seed + 100000,
        )
        test_examples = limit_examples(
            examples=test_raw_all,
            max_examples=args.max_test_examples,
            seed=args.seed + 200000,
        )

    print(f"Selected train examples: {len(train_examples)}")
    print(f"Selected dev examples:   {len(dev_examples)}")
    print(f"Selected test examples:  {len(test_examples)}")

    print("\nBuilding train rows...")
    train_rows = build_split_rows(
        examples=train_examples,
        split_name="train",
        max_sentences_per_example=args.max_kg_candidate_triples,
        max_subgraph_size=args.max_subgraph_size,
        max_candidates_per_question=args.max_candidates_per_question,
        kg_config=kg_config,
        llm_kg_constructor=llm_kg_constructor,
        kg_construction_workers=max(1, args.kg_construction_workers),
        kg_cache_path=(kg_cache_dir / "train_kg_cache.jsonl")
        if llm_kg_constructor is not None
        else None,
        seed=args.seed,
        candidate_beam_width=args.candidate_beam_width,
    )

    print("Building dev rows...")
    dev_rows = build_split_rows(
        examples=dev_examples,
        split_name="dev",
        max_sentences_per_example=args.max_kg_candidate_triples,
        max_subgraph_size=args.max_subgraph_size,
        max_candidates_per_question=args.max_candidates_per_question,
        kg_config=kg_config,
        llm_kg_constructor=llm_kg_constructor,
        kg_construction_workers=max(1, args.kg_construction_workers),
        kg_cache_path=(kg_cache_dir / "dev_kg_cache.jsonl")
        if llm_kg_constructor is not None
        else None,
        seed=args.seed + 100000,
        candidate_beam_width=args.candidate_beam_width,
    )

    print("Building test rows...")
    test_rows = build_split_rows(
        examples=test_examples,
        split_name="test",
        max_sentences_per_example=args.max_kg_candidate_triples,
        max_subgraph_size=args.max_subgraph_size,
        max_candidates_per_question=args.max_candidates_per_question,
        kg_config=kg_config,
        llm_kg_constructor=llm_kg_constructor,
        kg_construction_workers=max(1, args.kg_construction_workers),
        kg_cache_path=(kg_cache_dir / "test_kg_cache.jsonl")
        if llm_kg_constructor is not None
        else None,
        seed=args.seed + 200000,
        candidate_beam_width=args.candidate_beam_width,
    )

    summarize_rows("TRAIN", train_rows)
    summarize_rows("DEV", dev_rows)
    summarize_rows("TEST", test_rows)

    require_nonempty_split("train", train_rows, train_examples, args.train_file)
    require_nonempty_split("dev", dev_rows, dev_examples, [args.dev_file])
    require_nonempty_split("test", test_rows, test_examples, test_files)

    write_jsonl(out_dir / "train_subgraph_retrieval.jsonl", train_rows)
    write_jsonl(out_dir / "dev_subgraph_retrieval.jsonl", dev_rows)
    write_jsonl(out_dir / "test_subgraph_retrieval.jsonl", test_rows)

    metadata = {
        "dataset": "2WikiMultiHopQA",
        "train_file": args.train_file,
        "dev_file": args.dev_file,
        "test_file": args.test_file,
        "effective_test_file": test_files,
        "output_dir": str(out_dir),
        "seed": args.seed,
        "selection_manifest": (
            str(args.selection_manifest) if args.selection_manifest else None
        ),
        "max_train_examples": args.max_train_examples,
        "max_dev_examples": args.max_dev_examples,
        "max_test_examples": args.max_test_examples,
        "max_kg_candidate_triples": args.max_kg_candidate_triples,
        "max_subgraph_size": args.max_subgraph_size,
        "max_candidates_per_question": args.max_candidates_per_question,
        "candidate_beam_width": args.candidate_beam_width,
        "max_kg_bridge_triples": args.max_kg_bridge_triples,
        "kg_construction_backend": args.kg_construction_backend,
        "kg_construction_model": args.kg_construction_model,
        "kg_extraction_version": kg_config.extraction_version,
        "kg_cache_signature": kg_config.cache_signature,
        "kg_max_context_sentences": args.kg_max_context_sentences,
        "kg_construction_workers": args.kg_construction_workers,
        "kg_request_timeout": args.kg_request_timeout,
        "kg_max_retries": args.kg_max_retries,
        "kg_retry_initial_sleep": args.kg_retry_initial_sleep,
        "kg_cache_dir": str(kg_cache_dir) if llm_kg_constructor is not None else None,
        "schema_version": "unified_kg_reasoning_v3",
        "evidence_unit_type": "kg_triple",
        "gold_label_fields": ["raw_gold_evidence_units", "gold_explanations"],
        "candidate_field": "subgraph_units",
        "kg_source": "question_and_context_only",
        "candidate_composer": "beam_connected_subgraphs",
        "row_materializer": "materialize_retrieval_rows",
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "test_rows": len(test_rows),
        "train_examples": len({r["example_id"] for r in train_rows}),
        "dev_examples": len({r["example_id"] for r in dev_rows}),
        "test_examples": len({r["example_id"] for r in test_rows}),
    }

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nSaved 2WikiMultiHopQA subgraph retrieval data to: {out_dir}")


if __name__ == "__main__":
    main()
