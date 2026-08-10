import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from utils.llm_client import load_local_env, openai_compatible_client, parse_model_ref


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


# Demonyms observed on 2Wiki's opening-sentence "is a/was a NATIONALITY ..."
# convention, restricted to the countries that actually appear in the closed
# ~33-relation Wikidata-derived vocabulary used by this dataset's supporting
# facts (country / country of citizenship / country of origin).
_DEMONYM_TO_COUNTRY: Dict[str, str] = {
    "American": "United States", "British": "United Kingdom", "English": "United Kingdom",
    "Scottish": "United Kingdom", "Welsh": "United Kingdom", "Irish": "Ireland",
    "Indian": "India", "French": "France", "German": "Germany", "Italian": "Italy",
    "Spanish": "Spain", "Portuguese": "Portugal", "Russian": "Russia", "Japanese": "Japan",
    "Chinese": "China", "Canadian": "Canada", "Australian": "Australia", "Dutch": "Netherlands",
    "Swedish": "Sweden", "Norwegian": "Norway", "Danish": "Denmark", "Polish": "Poland",
    "Brazilian": "Brazil", "Mexican": "Mexico", "Egyptian": "Egypt", "Turkish": "Turkey",
    "Greek": "Greece", "Swiss": "Switzerland", "Austrian": "Austria", "Belgian": "Belgium",
    "Finnish": "Finland", "Israeli": "Israel", "Malaysian": "Malaysia",
    "Filipino": "Philippines", "Vietnamese": "Vietnam", "Thai": "Thailand",
    "Indonesian": "Indonesia", "Pakistani": "Pakistan", "Bangladeshi": "Bangladesh",
    "Nigerian": "Nigeria", "Kenyan": "Kenya", "Argentine": "Argentina",
    "Argentinian": "Argentina", "Chilean": "Chile", "Colombian": "Colombia",
    "Peruvian": "Peru", "Cuban": "Cuba", "Venezuelan": "Venezuela", "Ukrainian": "Ukraine",
    "Czech": "Czech Republic", "Hungarian": "Hungary", "Romanian": "Romania",
    "Bulgarian": "Bulgaria", "Croatian": "Croatia", "Serbian": "Serbia",
    "Icelandic": "Iceland", "Maltese": "Malta", "Lithuanian": "Lithuania",
}

# Occupation cues that mark a nearby demonym as the subject's own
# nationality (country of citizenship) rather than a work's production
# country (country of origin) -- e.g. "is an American film director" vs
# "is a 1987 American ... film".
_PERSON_OCCUPATION_TERMS = (
    "director", "composer", "producer", "actor", "actress", "singer", "painter",
    "writer", "novelist", "politician", "footballer", "architect", "photographer",
    "poet", "musician", "scientist", "scholar", "humanist", "diplomat", "editor",
    "publisher", "presenter", "performer", "playwright", "sculptor",
)
_WORK_TYPE_TERMS = ("film", "song", "novel", "series", "album", "drama", "documentary")

_NAME_RE = r"[A-Z][\w.\-]*(?:\s[A-Z][\w.\-]*)*"

# The dataset mixes American ("October 14, 2016") and British ("14 October
# 2016") date order -- both conventions appear in real 2Wiki source text,
# confirmed on Metamathics/Candyland (publication dates) and Tongzhi
# Emperor (date of death: "12 January 1875"). The bare-year fallback (with
# optional "c." for uncertain years, e.g. "born Leon Blank, c. 1927") lets
# an uncertain BIRTH date not block extracting a well-formed DEATH date in
# the same parenthetical -- without it the whole regex fails to match at
# all when the birth portion doesn't parse, losing a recoverable death
# date too (confirmed on Lee Madden: "(born Leon Blank, c. 1927 - April 9,
# 2009)").
_DATE_RE = r"(?:c\.\s*)?(?:[A-Za-z]+ \d{1,2},?\s?\d{4}|\d{1,2} [A-Za-z]+ \d{4}|\d{4})"

# One canonical pattern per Wikidata-style relation, restricted to the
# closed relation vocabulary found in 2Wiki's supporting facts (see
# KG_CONSTRUCTION_FINDINGS.md relation-frequency analysis). Deliberately
# excludes family relations (father/mother/child/sibling): "son/daughter of
# NAME" can't be safely assigned to father vs. mother without gender
# information the regex doesn't have, and a wrong assignment is worse than
# no triple.
_RELATION_PATTERNS: Dict[str, List["re.Pattern[str]"]] = {
    "director": [re.compile(rf"(?i:directed by) ({_NAME_RE})")],
    "composer": [
        re.compile(rf"(?i:musical score|music) by ({_NAME_RE})"),
        re.compile(rf"(?i:composed by) ({_NAME_RE})"),
    ],
    "performer": [re.compile(rf"(?i:performed|sung|recorded) by ({_NAME_RE})")],
    "producer": [re.compile(rf"(?i:produced by) ({_NAME_RE})")],
    "publisher": [re.compile(rf"(?i:published by) ({_NAME_RE})")],
    "editor": [re.compile(rf"(?i:edited by) ({_NAME_RE})")],
    "creator": [re.compile(rf"(?i:created by) ({_NAME_RE})")],
    "founded_by": [re.compile(rf"(?i:founded by) ({_NAME_RE})")],
    "presenter": [re.compile(rf"(?i:presented|hosted) by ({_NAME_RE})")],
    "spouse": [re.compile(rf"(?i:wife|husband) of ({_NAME_RE})")],
    "born_in": [
        re.compile(rf"(?i:born)(?:\s+as\s+{_NAME_RE})?\s+(?i:in) ({_NAME_RE}(?:,\s?{_NAME_RE})?)")
    ],
    "died_in": [re.compile(rf"(?i:died in) ({_NAME_RE}(?:,\s?{_NAME_RE})?)")],
    "cause_of_death": [re.compile(r"(?i:died of|died from) ([a-z][a-z \-]*?)(?=[.,;]|$)")],
    "inception": [re.compile(r"(?i:founded|established) in (?:[A-Za-z]+ )?(\d{4})")],
    "publication_date": [
        # A short lazy gap absorbs an inserted location/publisher clause:
        # "released in New Zealand on April 14, 2008", "released through
        # Scarlet Records on 14 October 2016" -- bounded to 40 chars so it
        # can't jump to an unrelated later "on" in the same sentence.
        re.compile(rf"(?i:released|published)\b.{{0,40}}?\bon\b ({_DATE_RE})")
    ],
}

_BIRTH_DEATH_SPAN_RE = re.compile(
    # Handles all three real Wikipedia opening-parenthetical conventions:
    # "(born Month Day, Year)", "(Month Day, Year - Month Day, Year)", and
    # "(born Full Birth Name, Month Day, Year - Month Day, Year)" -- the
    # optional name-token run absorbs a birth name inserted before the date.
    # Also tolerates no space before "(" (clean_text can't fix that; it's
    # the source text itself, e.g. "Oswald( June 9, 1919 - May 22, 1989)").
    rf"\(\s*(?:born\s+(?:[A-Z][\w.\-]*,?\s+)*)?"
    rf"({_DATE_RE})"
    rf"(?:\s*[-–—]\s*({_DATE_RE}))?"
    r"\s*\)"
)

_INCEPTION_YEAR_RE = re.compile(r"(?i:is an?) (\d{4})\b")


def _clean_entity_value(raw: str) -> str:
    value = clean_text(raw).rstrip(",;:")
    if value.endswith("."):
        # Strip a sentence-final period, but keep it if the last token is a
        # single-letter abbreviation (e.g. the "K." in "T. K. Ramchand").
        tokens = value.split()
        if tokens and len(tokens[-1]) > 2:
            value = value[:-1]
    return value.strip()


def _extract_birth_death_dates(sentence: str) -> tuple[str | None, str | None]:
    match = _BIRTH_DEATH_SPAN_RE.search(sentence)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _extract_demonym_relation(sentence: str) -> tuple[str | None, str | None]:
    for word in re.findall(r"\b([A-Z][a-z]+)\b", sentence):
        country = _DEMONYM_TO_COUNTRY.get(word)
        if not country:
            continue
        idx = sentence.find(word)
        window = sentence[idx : idx + 80].lower()
        if any(term in window for term in _PERSON_OCCUPATION_TERMS):
            return "country_of_citizenship", country
        if any(term in window for term in _WORK_TYPE_TERMS):
            return "country_of_origin", country
    return None, None


def extract_pattern_relation_triples(
    sentence_records: List[Dict[str, Any]],
    max_triples: int,
) -> List[List[str]]:
    """Deterministic-first typed-relation extraction restricted to the closed
    Wikidata-style relation vocabulary observed in this dataset's supporting
    facts. Cheap, stable, and zero-hallucination, but only covers sentences
    that follow Wikipedia's templated opening-sentence conventions -- callers
    should backfill remaining connectivity with construct_wiki_bridge_triples.
    """
    if max_triples <= 0:
        return []

    triples: List[List[str]] = []
    for rec in sentence_records:
        if len(triples) >= max_triples:
            break

        subject = clean_text(rec.get("title", ""))
        sentence = clean_text(rec.get("sentence", ""))
        if not subject or not sentence:
            continue

        for relation, patterns in _RELATION_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(sentence)
                if not match:
                    continue
                value = _clean_entity_value(match.group(1))
                if value and value.lower() != subject.lower():
                    triples.append([subject, relation, value])
                break

        birth, death = _extract_birth_death_dates(sentence)
        if birth:
            triples.append([subject, "date_of_birth", _clean_entity_value(birth)])
        if death:
            triples.append([subject, "date_of_death", _clean_entity_value(death)])

        # The demonym heuristic only holds for a page's own opening
        # self-description ("X is a NATIONALITY OCCUPATION/work..."); later
        # sentences often describe plot details or other mentioned entities
        # (e.g. a fictional character's nationality), which the heuristic
        # would otherwise misattribute to the page's own subject.
        if rec.get("sent_idx") in (0, "0"):
            demonym_relation, country = _extract_demonym_relation(sentence)
            if demonym_relation:
                triples.append([subject, demonym_relation, country])

            # The single most common way a film/song/album page states its
            # own release year: "X is a 1976 Italian-Brazilian ... film" --
            # far more common in this dataset than an explicit "released
            # on"/"published on" date. Restricted to the opening sentence
            # for the same reason as the demonym check above.
            year_match = _INCEPTION_YEAR_RE.search(sentence)
            if year_match:
                triples.append([subject, "publication_date", year_match.group(1)])

    normalized = [[s, normalize_relation(r), o] for s, r, o in triples]
    return dedupe_triples(normalized)[:max_triples]


def construct_deterministic_kg_triples(
    sentence_records: List[Dict[str, Any]],
    max_triples: int,
) -> List[List[str]]:
    """Deterministic-first pass: typed relation patterns fire first (cheap,
    stable, restricted to the closed relation vocabulary), then generic
    title co-mention bridges backfill whatever connectivity the patterns
    didn't cover, up to the same triple budget.
    """
    pattern_triples = extract_pattern_relation_triples(sentence_records, max_triples)
    remaining_budget = max_triples - len(pattern_triples)
    bridge_triples = (
        construct_wiki_bridge_triples(
            sentence_records=sentence_records,
            max_triples=remaining_budget,
        )
        if remaining_budget > 0
        else []
    )
    return dedupe_triples([*pattern_triples, *bridge_triples])[:max_triples]


@dataclass
class KGConstructionConfig:
    backend: str = "auto"
    model: str = "gpt-4.1-mini"
    max_triples: int = 64
    max_context_sentences: int = 20
    sleep_seconds: float = 0.0
    request_timeout: float = 90.0
    max_retries: int = 4
    retry_initial_sleep: float = 5.0


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
        prompt = build_llm_kg_prompt(
            question=question,
            sentence_records=sentence_records[: self.config.max_context_sentences],
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
    provided_evidences: Any,
    sentence_records: List[Dict[str, Any]],
    question: str,
    config: KGConstructionConfig,
    llm_constructor: LLMKGConstructor | None = None,
) -> tuple[List[List[str]], str]:
    """
    Construct graph-context triples for text QA examples.

    The gold answer is deliberately not accepted here. KG construction for text
    benchmarks must be context/question-only to avoid answer leakage.
    """
    context_only_backends = {"llm", "llm_with_title_bridges", "deterministic"}
    provided = (
        []
        if config.backend in context_only_backends
        else normalize_evidence_triples(provided_evidences)
    )
    if provided and config.backend in {"auto", "provided", "llm_with_provided"}:
        if config.backend == "llm_with_provided" and llm_constructor is not None:
            llm_triples = llm_constructor.construct(question, sentence_records)
            return dedupe_triples([*provided, *llm_triples])[: config.max_triples], (
                "provided_plus_llm"
            )
        return provided[: config.max_triples], "provided"

    if config.backend == "llm_with_title_bridges" and llm_constructor:
        llm_triples = llm_constructor.construct(question, sentence_records)
        bridge_triples = construct_wiki_bridge_triples(
            sentence_records=sentence_records,
            max_triples=config.max_triples,
        )
        triples = dedupe_triples([*llm_triples, *bridge_triples])
        if triples:
            return triples[: config.max_triples], "llm_plus_title_bridges"

    if config.backend in {"llm", "auto", "llm_with_provided"} and llm_constructor:
        triples = llm_constructor.construct(question, sentence_records)
        if triples:
            return triples[: config.max_triples], "llm"

    if config.backend in {"auto", "deterministic"}:
        pattern_triples = extract_pattern_relation_triples(
            sentence_records=sentence_records,
            max_triples=config.max_triples,
        )
        remaining_budget = config.max_triples - len(pattern_triples)
        bridge_triples = (
            construct_wiki_bridge_triples(
                sentence_records=sentence_records,
                max_triples=remaining_budget,
            )
            if remaining_budget > 0
            else []
        )
        triples = dedupe_triples([*pattern_triples, *bridge_triples])[: config.max_triples]
        if triples:
            method = (
                "deterministic_pattern_plus_title_bridge"
                if pattern_triples
                else "title_bridge_fallback"
            )
            return triples, method

    return [], "none"
