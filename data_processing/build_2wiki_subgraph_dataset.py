import argparse
import itertools
import json
import random
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import pandas as pd


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


def make_sentence_unit(title: str, sent_idx: int, sent: str) -> str:
    """
    Main 2Wiki evidence unit format.

    Example:
    SENT::Move (1970 film)::0::Move is a 1970 American comedy film...
    """
    return f"SENT::{safe_title(title)}::{int(sent_idx)}::{clean_sentence(sent)}"


def parse_sentence_unit(unit: str) -> Tuple[str, int, str]:
    parts = str(unit).split("::", 3)
    if len(parts) == 4 and parts[0] == "SENT":
        title = parts[1]
        try:
            idx = int(parts[2])
        except Exception:
            idx = -1
        sent = parts[3]
        return title, idx, sent
    return "", -1, str(unit)


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
      id, question, answer, type, evidences, supporting_facts, context

    supporting_facts usually looks like:
      {
        "title": array([...]),
        "sent_id": array([...])
      }

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

    # Normalize supporting_facts to:
    #   [[title, sent_idx], ...]
    sf = out.get("supporting_facts", {})

    if isinstance(sf, dict):
        titles = to_python_list(get_first_present(sf, ["title", "titles"], []))
        sent_ids = to_python_list(
            get_first_present(
                sf,
                ["sent_id", "sent_ids", "sentence_id", "sent_idx"],
                [],
            )
        )

        normalized_sf = []
        for title, idx in zip(titles, sent_ids):
            try:
                idx = int(idx)
            except Exception:
                continue
            normalized_sf.append([safe_title(title), idx])

        out["supporting_facts"] = normalized_sf

    elif isinstance(sf, list):
        normalized_sf = []
        for item in sf:
            if isinstance(item, dict):
                title = item.get("title", "")
                idx = item.get(
                    "sent_id", item.get("sentence_id", item.get("sent_idx", 0))
                )
                try:
                    idx = int(idx)
                except Exception:
                    continue
                normalized_sf.append([safe_title(title), idx])
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                title, idx = item
                try:
                    idx = int(idx)
                except Exception:
                    continue
                normalized_sf.append([safe_title(title), idx])
        out["supporting_facts"] = normalized_sf

    else:
        out["supporting_facts"] = []

    # Normalize evidences into readable triples, if present.
    evidences = out.get("evidences", [])
    normalized_evidences = []
    for ev in to_python_list(evidences):
        ev_list = to_python_list(ev)
        if ev_list:
            normalized_evidences.append([str(x) for x in ev_list])
    out["evidences"] = normalized_evidences

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

    # Deduplicate while preserving order.
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

    if answer_norm in {"yes", "no"}:
        # For comparison yes/no questions, answer string often does not appear
        # literally in the evidence, so do not reward yes/no occurrence.
        return 0.0

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
    Select a sentence pool for candidate construction.

    Always keeps gold sentences, then adds:
      - answer-containing sentences
      - top lexical-overlap sentences
      - sentences from pages whose titles overlap with the question
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
        title_score = lexical_overlap_score(question, title)

        score = q_score + 0.50 * a_score + 0.25 * title_score
        scored.append((score, unit))

    scored.sort(key=lambda x: x[0], reverse=True)

    pool: List[str] = []
    seen = set()

    # Always include gold units so positives are possible.
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

    # Include top lexical/title overlap sentences.
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
    sentence_pool: List[str],
    gold_units: List[str],
    max_subgraph_size: int,
    max_candidates_per_question: int,
    seed: int,
) -> List[List[str]]:
    """
    Generates candidate support subgraphs.

    Candidate = set/list of sentence units.

    For 2Wiki, gold support may have size 4, so max_subgraph_size should
    usually be 4.
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

    # Exact gold support candidate.
    if gold_units and len(set(gold_units)) <= max_subgraph_size:
        add_candidate(gold_units)

    # Gold supersets with one distractor if possible.
    if gold_units and len(set(gold_units)) < max_subgraph_size:
        for u in sentence_pool:
            if u not in gold_units:
                add_candidate(list(gold_units) + [u])

    # Gold partials: useful ranking targets for multi-hop chains.
    for size in range(1, min(len(gold_units), max_subgraph_size) + 1):
        for combo in itertools.combinations(gold_units, size):
            add_candidate(combo)

    # Singletons.
    for u in sentence_pool:
        add_candidate([u])

    # Pairs/triples/quads.
    for size in range(2, max_subgraph_size + 1):
        combos = list(itertools.combinations(sentence_pool, size))

        # Avoid exploding combinations.
        sample_limit = max_candidates_per_question * 4
        if len(combos) > sample_limit:
            combos = rng.sample(combos, sample_limit)

        for combo in combos:
            add_candidate(combo)
            if len(candidates) >= max_candidates_per_question:
                break

        if len(candidates) >= max_candidates_per_question:
            break

    # If too many, prefer candidates with more gold overlap and smaller size.
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


def make_2wiki_symbolic_features(
    question: str,
    answer: str,
    candidate_units: List[str],
    max_subgraph_size: int,
) -> List[float]:
    """
    8-dimensional gold-free text feature vector.

    0 question-token overlap
    1 answer-token overlap
    2 unique page ratio
    3 cross-page indicator
    4 multi-sentence indicator
    5 candidate size normalized
    6 average sentence position
    7 title-question overlap
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
    seed: int,
) -> List[Dict[str, Any]]:
    raw_id = str(get_first_present(example, ["_id", "id"], f"no_id_{seed}"))
    question = str(example.get("question", ""))
    answer = str(example.get("answer", ""))
    q_type = str(example.get("type", "unknown"))
    evidences = example.get("evidences", [])

    sentence_records = flatten_context(example)
    sent_lookup = {
        (rec["title"], int(rec["sent_idx"])): rec["unit"] for rec in sentence_records
    }

    gold_units = get_gold_support_units(example, sent_lookup)

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
        sentence_pool=sentence_pool,
        gold_units=gold_units,
        max_subgraph_size=max_subgraph_size,
        max_candidates_per_question=max_candidates_per_question,
        seed=seed,
    )

    rows: List[Dict[str, Any]] = []

    example_id = f"2WikiMultiHopQA__{split_name}__{raw_id}"
    answer_type = infer_answer_type(answer)

    for cand_idx, candidate_units in enumerate(candidates):
        label, best_f1, best_jaccard, exact, contains = label_candidate(
            candidate=candidate_units,
            gold_explanations=gold_explanations,
        )

        symbolic_features = make_2wiki_symbolic_features(
            question=question,
            answer=answer,
            candidate_units=candidate_units,
            max_subgraph_size=max_subgraph_size,
        )

        row_id = f"{example_id}__cand{cand_idx}"

        row = {
            "row_id": row_id,
            "example_id": example_id,
            "dataset": "2WikiMultiHopQA",
            "source_dataset": "2WikiMultiHopQA",
            "hop": "2hop",
            "answer_type": answer_type,
            "question": question,
            "answer": answer,
            "question_type": q_type,
            "evidences": evidences,
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
# Splitting / writing
# ============================================================


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
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
    bin_count = len({r["example_id"] for r in rows if r.get("answer_type") == "BIN"})
    open_count = len({r["example_id"] for r in rows if r.get("answer_type") == "OPEN"})

    print(f"{name}:")
    print(f"  rows:       {total}")
    print(f"  examples:   {len(example_ids)}")
    print(f"  BIN ex:     {bin_count}")
    print(f"  OPEN ex:    {open_count}")
    print(f"  positives:  {positives}")
    print(f"  negatives:  {total - positives}")
    if total:
        print(f"  pos rate:   {positives / total:.4f}")


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

    parser.add_argument("--max-sentences-per-example", type=int, default=30)
    parser.add_argument("--max-subgraph-size", type=int, default=4)
    parser.add_argument("--max-candidates-per-question", type=int, default=512)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        "dataset": "2WikiMultiHopQA",
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

    print(f"\nSaved 2WikiMultiHopQA subgraph retrieval data to: {out_dir}")


if __name__ == "__main__":
    main()
