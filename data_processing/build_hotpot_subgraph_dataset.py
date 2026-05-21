import argparse
import itertools
import json
import random
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Iterable


# ============================================================
# Basic text utilities
# ============================================================

_ARTICLES = {"a", "an", "the"}


def normalize_text(text: str) -> str:
    """
    Lightweight normalization for lexical scoring.
    """
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return [t for t in text.split() if t and t not in _ARTICLES]


def token_set(text: str) -> Set[str]:
    return set(tokenize(text))


def safe_title(title: str) -> str:
    """
    Keep page title readable but stable inside the SENT::... format.
    """
    title = str(title).replace("\n", " ").replace("\t", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title


def clean_sentence(sent: str) -> str:
    sent = str(sent).replace("\n", " ").replace("\t", " ")
    sent = re.sub(r"\s+", " ", sent).strip()
    return sent


def make_sentence_unit(title: str, sent_idx: int, sent: str) -> str:
    """
    Main HotpotQA evidence unit format.

    Example:
    SENT::Albert Einstein::0::Albert Einstein was a German-born physicist.
    """
    return f"SENT::{safe_title(title)}::{int(sent_idx)}::{clean_sentence(sent)}"


def parse_sentence_unit(unit: str) -> Tuple[str, int, str]:
    """
    Returns title, sentence index, sentence text.
    """
    parts = str(unit).split("::", 3)
    if len(parts) != 4 or parts[0] != "SENT":
        return "", -1, str(unit)
    title = parts[1]
    try:
        idx = int(parts[2])
    except ValueError:
        idx = -1
    sent = parts[3]
    return title, idx, sent


# ============================================================
# HotpotQA parsing
# ============================================================


def normalize_hotpot_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes Hugging Face HotpotQA parquet rows into the format expected
    by the builder.

    Expected output:
      {
        "_id": ...,
        "question": ...,
        "answer": ...,
        "type": ...,
        "level": ...,
        "context": [[title, [sentences...]], ...],
        "supporting_facts": [[title, sent_idx], ...]
      }
    """
    out = dict(row)

    # Some HF versions use "id" instead of "_id".
    if "_id" not in out and "id" in out:
        out["_id"] = out["id"]

    # Normalize context.
    context = out.get("context")

    # Common HF parquet format:
    # context = {"title": [...], "sentences": [[...], [...]]}
    if isinstance(context, dict):
        titles = first_present(context, "title", "titles", default=[])
        sentences = first_present(context, "sentences", "sentence", default=[])

        normalized_context = []
        for title, sents in zip(titles, sentences):
            normalized_context.append([title, list(sents)])

        out["context"] = normalized_context

    # Sometimes context is already list-like but numpy arrays appear inside.
    elif isinstance(context, list):
        normalized_context = []
        for item in context:
            if isinstance(item, dict):
                title = item.get("title", "")
                sents = item.get("sentences", item.get("sentence", []))
                normalized_context.append([title, list(sents)])
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                title, sents = item
                normalized_context.append([title, list(sents)])
        out["context"] = normalized_context

    # Normalize supporting_facts.
    sf = out.get("supporting_facts")

    # Common HF parquet format:
    # supporting_facts = {"title": [...], "sent_id": [...]}
    if isinstance(sf, dict):
        titles = first_present(sf, "title", "titles", default=[])
        sent_ids = first_present(
            sf,
            "sent_id",
            "sent_ids",
            "sentence_id",
            "sent_idx",
            default=[],
        )

        normalized_sf = []
        for title, idx in zip(titles, sent_ids):
            normalized_sf.append([title, int(idx)])

        out["supporting_facts"] = normalized_sf

    elif isinstance(sf, list):
        normalized_sf = []
        for item in sf:
            if isinstance(item, dict):
                title = item.get("title", "")
                idx = (
                    item.get("sent_id")
                    if "sent_id" in item
                    else item.get("sentence_id", item.get("sent_idx", 0))
                )
                normalized_sf.append([title, int(idx)])
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                title, idx = item
                normalized_sf.append([title, int(idx)])
        out["supporting_facts"] = normalized_sf

    return out


def first_present(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Return the first non-None value for keys without truth-testing arrays.
    """
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def load_hotpot_file(path: str) -> List[Dict[str, Any]]:
    """
    Loads HotpotQA from either:
      - .json
      - .jsonl
      - .parquet
      - a directory containing parquet files

    Hugging Face downloads usually use parquet.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"HotpotQA path not found: {p}")

    # Directory: load all parquet files inside it.
    if p.is_dir():
        parquet_files = sorted(p.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under directory: {p}")

        all_rows = []
        for pq in parquet_files:
            all_rows.extend(load_hotpot_file(str(pq)))
        return all_rows

    suffix = p.suffix.lower()

    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list, got {type(data)} from {p}")
        return data

    if suffix == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(p)
        records = df.to_dict(orient="records")

        # Normalize HF-style rows if needed.
        return [normalize_hotpot_record(r) for r in records]

    raise ValueError(f"Unsupported HotpotQA file type: {p}")


def flatten_context(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts HotpotQA context into sentence records.

    HotpotQA context format:
      "context": [
          ["Page Title", ["sent0", "sent1", ...]],
          ...
      ]

    Returns list of:
      {
        "title": ...,
        "sent_idx": ...,
        "sentence": ...,
        "unit": ...
      }
    """
    records: List[Dict[str, Any]] = []

    context = example.get("context", [])
    for page in context:
        if not isinstance(page, list) or len(page) != 2:
            continue

        title, sentences = page
        title = safe_title(title)

        if not isinstance(sentences, list):
            continue

        for idx, sent in enumerate(sentences):
            sent_clean = clean_sentence(sent)
            if not sent_clean:
                continue

            unit = make_sentence_unit(title, idx, sent_clean)
            records.append(
                {
                    "title": title,
                    "sent_idx": idx,
                    "sentence": sent_clean,
                    "unit": unit,
                }
            )

    return records


def get_gold_support_units(
    example: Dict[str, Any], sent_lookup: Dict[Tuple[str, int], str]
) -> List[str]:
    """
    Extract gold supporting facts from HotpotQA.

    HotpotQA supporting_facts format:
      [["Page Title", sentence_index], ...]

    Returns sentence units.
    """
    gold_units: List[str] = []

    for item in example.get("supporting_facts", []):
        if not isinstance(item, list) or len(item) != 2:
            continue

        title, idx = item
        title = safe_title(title)

        try:
            idx = int(idx)
        except Exception:
            continue

        unit = sent_lookup.get((title, idx))
        if unit is not None:
            gold_units.append(unit)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in gold_units:
        if u not in seen:
            seen.add(u)
            deduped.append(u)

    return deduped


# ============================================================
# Candidate generation
# ============================================================


def lexical_overlap_score(query: str, text: str) -> float:
    q = token_set(query)
    t = token_set(text)
    if not q:
        return 0.0
    return len(q & t) / max(len(q), 1)


def answer_overlap_score(answer: str, text: str) -> float:
    answer = str(answer or "")
    if not answer:
        return 0.0

    answer_norm = normalize_text(answer)
    text_norm = normalize_text(text)

    if not answer_norm:
        return 0.0

    # Exact substring match is useful for HotpotQA answer grounding.
    if answer_norm in text_norm:
        return 1.0

    a = token_set(answer_norm)
    t = token_set(text_norm)
    if not a:
        return 0.0
    return len(a & t) / max(len(a), 1)


def title_overlap_score(question: str, units: List[str]) -> float:
    q = token_set(question)
    if not q:
        return 0.0

    title_tokens: Set[str] = set()
    for u in units:
        title, _, _ = parse_sentence_unit(u)
        title_tokens |= token_set(title)

    return len(q & title_tokens) / max(len(q), 1)


def select_sentence_pool(
    example: Dict[str, Any],
    sentence_records: List[Dict[str, Any]],
    gold_units: List[str],
    max_sentences_per_example: int,
) -> List[str]:
    """
    Selects a limited pool of sentence units for candidate generation.

    Always keeps gold units so positives can be generated.
    Then adds top lexical and answer-containing sentences.
    """
    question = example.get("question", "")
    answer = example.get("answer", "")

    scored = []
    for rec in sentence_records:
        unit = rec["unit"]
        sent = rec["sentence"]
        title = rec["title"]

        q_score = lexical_overlap_score(question, sent + " " + title)
        a_score = answer_overlap_score(answer, sent)
        score = q_score + 0.5 * a_score

        scored.append((score, unit))

    scored.sort(key=lambda x: x[0], reverse=True)

    pool: List[str] = []
    seen = set()

    # Always include gold units.
    for u in gold_units:
        if u not in seen:
            seen.add(u)
            pool.append(u)

    # Include answer-containing sentences.
    for rec in sentence_records:
        u = rec["unit"]
        if u in seen:
            continue
        if answer_overlap_score(answer, rec["sentence"]) > 0:
            seen.add(u)
            pool.append(u)

    # Include top lexical sentences.
    for _, u in scored:
        if len(pool) >= max_sentences_per_example:
            break
        if u not in seen:
            seen.add(u)
            pool.append(u)

    return pool[:max_sentences_per_example]


def support_f1(candidate: Iterable[str], gold: Iterable[str]) -> float:
    c = set(candidate)
    g = set(gold)
    if not c and not g:
        return 1.0
    if not c or not g:
        return 0.0
    return 2.0 * len(c & g) / (len(c) + len(g))


def support_jaccard(candidate: Iterable[str], gold: Iterable[str]) -> float:
    c = set(candidate)
    g = set(gold)
    if not c and not g:
        return 1.0
    if not c or not g:
        return 0.0
    return len(c & g) / len(c | g)


def label_candidate(
    candidate: List[str], gold_explanations: List[List[str]]
) -> Tuple[int, float, float, bool, bool]:
    """
    Returns:
      label,
      best_f1,
      best_jaccard,
      exact_match_any_gold,
      contains_any_gold_explanation

    label = 1 iff candidate contains a complete gold explanation.
    """
    cset = set(candidate)

    best_f1 = 0.0
    best_j = 0.0
    exact = False
    contains = False

    for gold in gold_explanations:
        gset = set(gold)

        f1 = support_f1(cset, gset)
        jac = support_jaccard(cset, gset)

        best_f1 = max(best_f1, f1)
        best_j = max(best_j, jac)

        if cset == gset:
            exact = True

        if gset and gset.issubset(cset):
            contains = True

    label = 1 if contains else 0
    return label, best_f1, best_j, exact, contains


def generate_candidate_subgraphs(
    example: Dict[str, Any],
    sentence_pool: List[str],
    gold_units: List[str],
    max_subgraph_size: int,
    max_candidates_per_question: int,
    seed: int,
) -> List[List[str]]:
    """
    Generates candidate sentence subgraphs.

    Candidate = list of sentence units.

    We include:
      - all singletons
      - combinations up to max_subgraph_size
      - explicitly ensure gold support is included
      - cap total number of candidates
    """
    rng = random.Random(seed)

    candidates: List[Tuple[str, ...]] = []
    seen = set()

    def add_candidate(units: Iterable[str]):
        cand = tuple(sorted(set(units)))
        if not cand:
            return
        if len(cand) > max_subgraph_size:
            return
        if cand not in seen:
            seen.add(cand)
            candidates.append(cand)

    # Always add gold explanation if within size.
    if gold_units and len(set(gold_units)) <= max_subgraph_size:
        add_candidate(gold_units)

    # Add gold + one distractor supersets to train compactness.
    if gold_units and len(set(gold_units)) < max_subgraph_size:
        for u in sentence_pool:
            if u not in gold_units:
                add_candidate(list(gold_units) + [u])

    # Add singletons.
    for u in sentence_pool:
        add_candidate([u])

    # Add pairs/triples.
    for size in range(2, max_subgraph_size + 1):
        combos = list(itertools.combinations(sentence_pool, size))

        # If too many, sample but keep deterministic.
        if len(combos) > max_candidates_per_question * 4:
            combos = rng.sample(combos, max_candidates_per_question * 4)

        for combo in combos:
            add_candidate(combo)
            if len(candidates) >= max_candidates_per_question:
                break

        if len(candidates) >= max_candidates_per_question:
            break

    # If we still have too many, keep candidates with gold overlap first.
    if len(candidates) > max_candidates_per_question:
        gold_set = set(gold_units)

        def priority(c: Tuple[str, ...]):
            cset = set(c)
            overlap = len(cset & gold_set)
            return (overlap, -len(cset))

        candidates.sort(key=priority, reverse=True)
        candidates = candidates[:max_candidates_per_question]

    return [list(c) for c in candidates]


# ============================================================
# Symbolic/text features
# ============================================================


def make_hotpot_symbolic_features(
    question: str,
    answer: str,
    candidate_units: List[str],
    max_subgraph_size: int,
) -> List[float]:
    """
    8-dimensional gold-free symbolic/text feature vector.

    Keep dimension compatible with existing GNN.
    """
    candidate_text_parts = []
    titles = []
    sent_positions = []

    for u in candidate_units:
        title, idx, sent = parse_sentence_unit(u)
        candidate_text_parts.append(title)
        candidate_text_parts.append(sent)
        if title:
            titles.append(title)
        if idx >= 0:
            sent_positions.append(idx)

    candidate_text = " ".join(candidate_text_parts)

    q_overlap = lexical_overlap_score(question, candidate_text)
    a_overlap = answer_overlap_score(answer, candidate_text)

    unique_titles = len(set(titles))
    size = len(candidate_units)

    unique_page_ratio = unique_titles / max(size, 1)
    cross_page = 1.0 if unique_titles >= 2 else 0.0
    multi_sentence = 1.0 if size >= 2 else 0.0
    size_norm = min(size / max(max_subgraph_size, 1), 1.0)

    if sent_positions:
        # Lower sentence indices are often more salient.
        avg_pos = sum(sent_positions) / len(sent_positions)
        avg_sentence_position = min(avg_pos / 10.0, 1.0)
    else:
        avg_sentence_position = 0.0

    title_overlap = title_overlap_score(question, candidate_units)

    return [
        float(q_overlap),
        float(a_overlap),
        float(unique_page_ratio),
        float(cross_page),
        float(multi_sentence),
        float(size_norm),
        float(avg_sentence_position),
        float(title_overlap),
    ]


# ============================================================
# Row construction
# ============================================================


def build_rows_for_example(
    example: Dict[str, Any],
    split_name: str,
    max_sentences_per_example: int,
    max_subgraph_size: int,
    max_candidates_per_question: int,
    seed: int,
) -> List[Dict[str, Any]]:
    ex_id = str(example.get("_id", f"no_id_{seed}"))
    question = str(example.get("question", ""))
    answer = str(example.get("answer", ""))
    q_type = str(example.get("type", "unknown"))
    level = str(example.get("level", "unknown"))

    sentence_records = flatten_context(example)
    sent_lookup = {
        (rec["title"], int(rec["sent_idx"])): rec["unit"] for rec in sentence_records
    }

    gold_units = get_gold_support_units(example, sent_lookup)

    # Skip examples without gold support.
    if not gold_units:
        return []

    gold_explanations = [gold_units]

    sentence_pool = select_sentence_pool(
        example=example,
        sentence_records=sentence_records,
        gold_units=gold_units,
        max_sentences_per_example=max_sentences_per_example,
    )

    if not sentence_pool:
        return []

    candidates = generate_candidate_subgraphs(
        example=example,
        sentence_pool=sentence_pool,
        gold_units=gold_units,
        max_subgraph_size=max_subgraph_size,
        max_candidates_per_question=max_candidates_per_question,
        seed=seed,
    )

    rows: List[Dict[str, Any]] = []

    for cand_idx, candidate_units in enumerate(candidates):
        label, best_f1, best_jaccard, exact, contains = label_candidate(
            candidate=candidate_units,
            gold_explanations=gold_explanations,
        )

        symbolic_features = make_hotpot_symbolic_features(
            question=question,
            answer=answer,
            candidate_units=candidate_units,
            max_subgraph_size=max_subgraph_size,
        )

        row_id = f"HotpotQA__{split_name}__{ex_id}__cand{cand_idx}"

        row = {
            "row_id": row_id,
            "example_id": f"HotpotQA__{split_name}__{ex_id}",
            "dataset": "HotpotQA",
            "source_dataset": "HotpotQA",
            "hop": "2hop",
            "answer_type": "OPEN",
            "question": question,
            "answer": answer,
            "hotpot_type": q_type,
            "level": level,
            # Compatibility with OWL scripts.
            "sparql_query": "",
            "query": "",
            # Candidate support.
            "subgraph_units": candidate_units,
            "subgraph_size": len(candidate_units),
            # Gold support.
            "gold_units": gold_units,
            "gold_explanations": gold_explanations,
            # Supervision.
            "label": int(label),
            "rank_target": float(best_f1),
            # Diagnostics.
            "best_set_f1_to_gold": float(best_f1),
            "best_jaccard_to_gold": float(best_jaccard),
            "exact_match_any_gold": bool(exact),
            "contains_any_gold_explanation": bool(contains),
            # Model features.
            "symbolic_features": symbolic_features,
        }

        rows.append(row)

    return rows


# ============================================================
# Dataset writing
# ============================================================


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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

    if max_train_examples and max_train_examples > 0:
        # Keep enough examples for train + dev if possible.
        total = max_train_examples + (
            max_dev_examples
            if max_dev_examples > 0
            else int(max_train_examples * dev_ratio)
        )
        shuffled = shuffled[: min(total, len(shuffled))]

    n_dev = int(round(len(shuffled) * dev_ratio))

    if max_dev_examples and max_dev_examples > 0:
        n_dev = min(n_dev, max_dev_examples)

    dev = shuffled[:n_dev]
    train = shuffled[n_dev:]

    if max_train_examples and max_train_examples > 0:
        train = train[:max_train_examples]

    return train, dev


def limit_examples(
    examples: List[Dict[str, Any]], max_examples: int, seed: int
) -> List[Dict[str, Any]]:
    if not max_examples or max_examples <= 0:
        return examples
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    return shuffled[:max_examples]


def build_split_rows(
    examples: List[Dict[str, Any]],
    split_name: str,
    max_sentences_per_example: int,
    max_subgraph_size: int,
    max_candidates_per_question: int,
    seed: int,
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []

    for idx, ex in enumerate(examples):
        rows = build_rows_for_example(
            example=ex,
            split_name=split_name,
            max_sentences_per_example=max_sentences_per_example,
            max_subgraph_size=max_subgraph_size,
            max_candidates_per_question=max_candidates_per_question,
            seed=seed + idx,
        )
        all_rows.extend(rows)

    return all_rows


def summarize_rows(name: str, rows: List[Dict[str, Any]]) -> None:
    example_ids = {r["example_id"] for r in rows}
    positives = sum(int(r.get("label", 0)) for r in rows)
    total = len(rows)
    print(f"{name}:")
    print(f"  rows:      {total}")
    print(f"  examples:  {len(example_ids)}")
    print(f"  positives: {positives}")
    print(f"  negatives: {total - positives}")
    if total:
        print(f"  pos rate:  {positives / total:.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-file", type=str, nargs="+", required=True)
    parser.add_argument("--dev-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="data/HotpotQA")

    parser.add_argument("--dev-ratio-from-train", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-train-examples", type=int, default=1000)
    parser.add_argument("--max-dev-examples", type=int, default=200)
    parser.add_argument("--max-test-examples", type=int, default=300)

    parser.add_argument("--max-sentences-per-example", type=int, default=30)
    parser.add_argument("--max-subgraph-size", type=int, default=3)
    parser.add_argument("--max-candidates-per-question", type=int, default=512)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HotpotQA...")
    train_raw_all = []
    for train_file in args.train_file:
        train_raw_all.extend(load_hotpot_file(train_file))
    dev_raw_all = load_hotpot_file(args.dev_file)
    test_files = args.test_file if args.test_file is not None else [args.dev_file]
    test_raw_all = []
    for test_file in test_files:
        test_raw_all.extend(load_hotpot_file(test_file))

    print(f"Raw train examples: {len(train_raw_all)}")
    print(f"Raw dev examples:   {len(dev_raw_all)}")
    print(f"Raw test examples:  {len(test_raw_all)}")

    # Train and dev are sampled from their respective source files. If
    # --test-file is omitted, the official dev file is also used as test.
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
        max_sentences_per_example=args.max_sentences_per_example,
        max_subgraph_size=args.max_subgraph_size,
        max_candidates_per_question=args.max_candidates_per_question,
        seed=args.seed,
    )

    print("Building dev rows...")
    dev_rows = build_split_rows(
        examples=dev_examples,
        split_name="dev",
        max_sentences_per_example=args.max_sentences_per_example,
        max_subgraph_size=args.max_subgraph_size,
        max_candidates_per_question=args.max_candidates_per_question,
        seed=args.seed + 100000,
    )

    print("Building test rows...")
    test_rows = build_split_rows(
        examples=test_examples,
        split_name="test",
        max_sentences_per_example=args.max_sentences_per_example,
        max_subgraph_size=args.max_subgraph_size,
        max_candidates_per_question=args.max_candidates_per_question,
        seed=args.seed + 200000,
    )

    summarize_rows("TRAIN", train_rows)
    summarize_rows("DEV", dev_rows)
    summarize_rows("TEST", test_rows)

    write_jsonl(out_dir / "train_subgraph_retrieval.jsonl", train_rows)
    write_jsonl(out_dir / "dev_subgraph_retrieval.jsonl", dev_rows)
    write_jsonl(out_dir / "test_subgraph_retrieval.jsonl", test_rows)

    metadata = {
        "dataset": "HotpotQA",
        "train_file": args.train_file,
        "dev_file": args.dev_file,
        "test_file": args.test_file,
        "effective_test_file": test_files,
        "output_dir": str(out_dir),
        "seed": args.seed,
        "max_train_examples": args.max_train_examples,
        "max_dev_examples": args.max_dev_examples,
        "max_test_examples": args.max_test_examples,
        "max_sentences_per_example": args.max_sentences_per_example,
        "max_subgraph_size": args.max_subgraph_size,
        "max_candidates_per_question": args.max_candidates_per_question,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "test_rows": len(test_rows),
        "train_examples": len({r["example_id"] for r in train_rows}),
        "dev_examples": len({r["example_id"] for r in dev_rows}),
        "test_examples": len({r["example_id"] for r in test_rows}),
    }

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nSaved HotpotQA subgraph retrieval data to: {out_dir}")


if __name__ == "__main__":
    main()
