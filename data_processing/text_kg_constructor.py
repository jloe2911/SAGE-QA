import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from utils.llm_client import load_local_env, openai_compatible_client, parse_model_ref


SUPPORTED_KG_CONSTRUCTION_BACKENDS = frozenset(
    {
        "deterministic",
        "llm",
        "llm_with_title_bridges",
    }
)


def clean_text(text: Any) -> str:
    text = str(text or "").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def phrase_in_text(phrase: str, text: str) -> bool:
    phrase_norm = normalize_text(phrase)
    text_norm = normalize_text(text)
    if len(phrase_norm) < 3 or not text_norm:
        return False
    return f" {phrase_norm} " in f" {text_norm} "


def to_python_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return [value]


def normalize_evidence_triples(raw_evidences: Any) -> List[List[str]]:
    triples: List[List[str]] = []
    for ev in to_python_list(raw_evidences):
        if isinstance(ev, dict):
            subject = ev.get("subject", ev.get("head", ev.get("s")))
            predicate = ev.get("predicate", ev.get("relation", ev.get("p")))
            obj = ev.get("object", ev.get("tail", ev.get("o")))
            ev_list = [subject, predicate, obj]
        else:
            ev_list = to_python_list(ev)

        if len(ev_list) >= 3 and all(x is not None for x in ev_list[:3]):
            triples.append([clean_text(x) for x in ev_list[:3]])
    return dedupe_triples(triples)


def normalize_relation(predicate: Any) -> str:
    predicate = clean_text(predicate).lower()
    predicate = predicate.replace("-", "_")
    predicate = re.sub(r"\s+", "_", predicate)
    predicate = re.sub(r"[^a-z0-9_]", "", predicate)
    predicate = re.sub(r"_+", "_", predicate).strip("_")
    relation_map = {
        "birthplace": "born_in",
        "place_of_birth": "born_in",
        "was_born_in": "born_in",
        "death_place": "died_in",
        "place_of_death": "died_in",
        "located_at": "located_in",
        "is_located_in": "located_in",
        "belongs_to": "member_of",
        "is_member_of": "member_of",
        "spouse": "spouse_of",
        "married_to": "spouse_of",
        "creator": "created_by",
        "made_by": "created_by",
        "director": "directed_by",
        "writer": "written_by",
        "author": "written_by",
        "authored_by": "written_by",
        "producer": "produced_by",
        "founder": "founded_by",
        "owner": "owned_by",
        "capital": "capital_of",
        "country": "country_of",
        "nationality": "country_of",
        "profession": "occupation",
        "education": "educated_at",
        "alma_mater": "educated_at",
        "employed_by": "employer",
        "works_for": "employer",
        "published_on": "publication_date",
        "released_on": "release_date",
        "date_of_birth": "birth_date",
        "date_of_death": "death_date",
    }
    return relation_map.get(predicate, predicate)


def dedupe_triples(triples: Iterable[Iterable[Any]]) -> List[List[str]]:
    out: List[List[str]] = []
    seen = set()
    for triple in triples:
        items = list(triple)
        if len(items) < 3:
            continue
        subject, predicate, obj = [clean_text(x) for x in items[:3]]
        if not subject or not predicate or not obj or subject == obj:
            continue
        key = (subject.lower(), predicate.lower(), obj.lower())
        if key not in seen:
            seen.add(key)
            out.append([subject, predicate, obj])
    return out


def construct_wiki_bridge_triples(
    sentence_records: List[Dict[str, Any]],
    max_triples: int,
) -> List[List[str]]:
    """Build context-only title co-mention bridges between Wikipedia pages."""
    if max_triples <= 0:
        return []

    titles = []
    seen_titles = set()
    for rec in sentence_records:
        title = clean_text(rec.get("title", ""))
        if title and title not in seen_titles:
            seen_titles.add(title)
            titles.append(title)

    triples: List[List[str]] = []
    seen = set()

    def add(subject: str, predicate: str, obj: str) -> None:
        if len(triples) >= max_triples:
            return
        subject = clean_text(subject)
        predicate = clean_text(predicate)
        obj = clean_text(obj)
        if not subject or not predicate or not obj or subject == obj:
            return
        key = (subject.lower(), predicate.lower(), obj.lower())
        if key not in seen:
            seen.add(key)
            triples.append([subject, predicate, obj])

    for rec in sentence_records:
        source_title = rec.get("title", "")
        sentence = rec.get("sentence", "")

        for target_title in titles:
            if target_title == source_title:
                continue
            if phrase_in_text(target_title, sentence):
                add(source_title, "mentions_page", target_title)

        if len(triples) >= max_triples:
            break

    return triples


@dataclass
class KGConstructionConfig:
    backend: str = "deterministic"
    model: str = "gpt-4.1-mini"
    max_triples: int = 64
    max_context_sentences: int = 20
    sleep_seconds: float = 0.0
    request_timeout: float = 90.0
    max_retries: int = 4
    retry_initial_sleep: float = 5.0
    extraction_version: str = "context_kg_v2"

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_KG_CONSTRUCTION_BACKENDS:
            supported = ", ".join(sorted(SUPPORTED_KG_CONSTRUCTION_BACKENDS))
            raise ValueError(
                f"Unsupported KG construction backend {self.backend!r}. "
                f"Choose one of: {supported}."
            )

    @property
    def cache_signature(self) -> str:
        """Identify cached output that is safe to reuse for this extractor."""
        return "|".join(
            [
                self.extraction_version,
                self.backend,
                self.model,
                str(self.max_triples),
                str(self.max_context_sentences),
            ]
        )


class LLMKGConstructor:
    def __init__(self, config: KGConstructionConfig):
        self.config = config
        self._client = None
        load_local_env()

    def _openai_client(self):
        if self._client is None:
            self._client, _, _ = openai_compatible_client(self.config.model)
        return self._client

    def construct(
        self,
        question: str,
        sentence_records: List[Dict[str, Any]],
    ) -> List[List[str]]:
        selected_records = select_context_sentences(
            question=question,
            sentence_records=sentence_records,
            limit=self.config.max_context_sentences,
        )
        prompt = build_llm_kg_prompt(
            question=question,
            sentence_records=selected_records,
            max_triples=self.config.max_triples,
        )
        client = self._openai_client()
        model_name = parse_model_ref(self.config.model).model
        kwargs = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract concise knowledge-graph triples from "
                        "multi-hop QA evidence. The answer is unknown. "
                        "Use only the question and evidence. Return JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        if not model_name.startswith("gpt-5"):
            kwargs["temperature"] = 0.0
        response = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    **kwargs,
                    timeout=self.config.request_timeout,
                )
                break
            except Exception as exc:
                if attempt >= self.config.max_retries:
                    raise
                message = str(exc).lower()
                is_rate_limit = "429" in message or "rate" in message
                sleep_for = self.config.retry_initial_sleep * (2**attempt)
                if not is_rate_limit:
                    sleep_for = min(sleep_for, self.config.retry_initial_sleep)
                print(
                    f"  LLM KG extraction retry {attempt + 1}/"
                    f"{self.config.max_retries} after error: {exc!r}; "
                    f"sleeping {sleep_for:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_for)
        if self.config.sleep_seconds > 0:
            time.sleep(self.config.sleep_seconds)
        if response is None:
            return []
        return parse_llm_triples(response.choices[0].message.content)


def select_context_sentences(
    question: str,
    sentence_records: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Select context without dropping whole pages needed by a multi-hop chain.

    Pure question-overlap ranking tends to keep several sentences from the
    first-hop page and discard the bridge/answer page. We first retain the
    earliest sentence from every page (normally its defining sentence), then
    fill the remaining budget with question-relevant and cross-page-linking
    sentences. This remains gold- and answer-free.
    """
    if limit <= 0 or len(sentence_records) <= limit:
        return list(sentence_records)

    question_tokens = set(normalize_text(question).split())
    titles = []
    seen_titles = set()
    for record in sentence_records:
        title = clean_text(record.get("title", ""))
        title_key = normalize_text(title)
        if title_key and title_key not in seen_titles:
            seen_titles.add(title_key)
            titles.append((title_key, title))

    scored = []
    for index, record in enumerate(sentence_records):
        title = clean_text(record.get("title", ""))
        sentence = clean_text(record.get("sentence", ""))
        title_tokens = set(normalize_text(title).split())
        sentence_tokens = set(normalize_text(sentence).split())
        overlap = len(question_tokens & (title_tokens | sentence_tokens))
        title_overlap = len(question_tokens & title_tokens)
        linked_pages = sum(
            1
            for target_key, target_title in titles
            if target_key != normalize_text(title)
            and phrase_in_text(target_title, sentence)
        )
        score = overlap + 0.5 * title_overlap + 1.5 * linked_pages
        scored.append((score, linked_pages, -index, index, record))

    selected_indices = set()
    # Page-opening sentences preserve broad evidence coverage.
    for index, record in enumerate(sentence_records):
        title_key = normalize_text(record.get("title", ""))
        if title_key and not any(
            normalize_text(sentence_records[i].get("title", "")) == title_key
            for i in selected_indices
        ):
            selected_indices.add(index)
            if len(selected_indices) >= limit:
                break

    for _, _, _, index, _ in sorted(scored, reverse=True):
        if len(selected_indices) >= limit:
            break
        selected_indices.add(index)

    # Preserve source order so adjacent statements and pronouns remain legible.
    return [sentence_records[index] for index in sorted(selected_indices)]


def build_llm_kg_prompt(
    question: str,
    sentence_records: List[Dict[str, Any]],
    max_triples: int,
) -> str:
    evidence_lines = []
    for i, rec in enumerate(sentence_records, start=1):
        evidence_lines.append(
            f"[{i}] title={clean_text(rec.get('title', ''))}; "
            f"sentence={clean_text(rec.get('sentence', ''))}"
        )
    evidence = "\n".join(evidence_lines)
    return f"""Extract a compact knowledge graph from the evidence for this multi-hop QA example.

Question: {clean_text(question)}

Evidence:
{evidence}

Rules:
- Use only entities, facts, and relations stated in the evidence sentences.
- Do not use or infer the gold answer.
- Process every evidence line. Extract all explicit binary facts that could be
  useful for answering the question, rather than returning only the most
  obvious first-hop facts.
- Treat each page title as the main entity of that page. Resolve pronouns and
  descriptive openings back to the page title when the sentence supports it.
- Preserve entity names and values exactly as written. Use the page title
  instead of a shortened alias when both refer to the same subject.
- Prefer triples that connect page titles, question entities, intermediate entities, and plausible answer candidates.
- Prioritize triples that may form multi-hop paths from question entities to answer candidates.
- Use short readable predicates such as "born_in", "located_in", "member_of", "created_by", "directed_by", "written_by", "founded_by", "country_of", "occupation", or concise natural-language relations.
- Avoid vague predicates like "related_to" unless no better predicate is possible.
- Do not include sentence IDs as entities.
- Return at most {max_triples} triples.

Return valid JSON only:
{{
  "triples": [
    {{"subject": "...", "predicate": "...", "object": "..."}}
  ]
}}
"""


def parse_llm_triples(text: str) -> List[List[str]]:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    obj: Any = None
    try:
        obj = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
            except Exception:
                obj = None

    if isinstance(obj, dict):
        raw = obj.get("triples", [])
    elif isinstance(obj, list):
        raw = obj
    else:
        raw = []

    triples: List[List[str]] = []
    for item in raw:
        if isinstance(item, dict):
            subject = item.get("subject", item.get("head", item.get("s")))
            predicate = item.get("predicate", item.get("relation", item.get("p")))
            obj = item.get("object", item.get("tail", item.get("o")))
            item_list = [subject, predicate, obj]
        else:
            item_list = to_python_list(item)

        if len(item_list) >= 3 and all(x is not None for x in item_list[:3]):
            subject, predicate, obj = item_list[:3]
            triples.append(
                [clean_text(subject), normalize_relation(predicate), clean_text(obj)]
            )
    return dedupe_triples(triples)


def construct_text_kg(
    *,
    sentence_records: List[Dict[str, Any]],
    question: str,
    config: KGConstructionConfig,
    llm_constructor: LLMKGConstructor | None = None,
) -> tuple[List[List[str]], str]:
    """
    Construct graph-context triples for text QA examples.

    KG construction for text benchmarks is strictly question/context-only.
    Dataset-provided evidence triples are annotations of the gold reasoning path,
    not inference-time inputs.
    """
    if config.backend in {"llm", "llm_with_title_bridges"} and llm_constructor is None:
        raise ValueError(f"KG backend {config.backend!r} requires an LLMKGConstructor.")

    if config.backend == "llm_with_title_bridges":
        llm_triples = llm_constructor.construct(question, sentence_records)
        bridge_triples = construct_wiki_bridge_triples(
            sentence_records=sentence_records,
            max_triples=config.max_triples,
        )
        triples = dedupe_triples([*llm_triples, *bridge_triples])
        if triples:
            return triples[: config.max_triples], "llm_plus_title_bridges"

    if config.backend == "llm":
        triples = llm_constructor.construct(question, sentence_records)
        if triples:
            return triples[: config.max_triples], "llm"

    if config.backend == "deterministic":
        triples = construct_wiki_bridge_triples(
            sentence_records=sentence_records,
            max_triples=config.max_triples,
        )
        if triples:
            return triples, "title_bridge_fallback"

    return [], "none"
