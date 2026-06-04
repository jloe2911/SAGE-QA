"""
Context-only KG constructor — no answer leakage.

Builds a knowledge graph purely from context sentences.
The gold answer and gold evidences are never accessed.

Two backends:
  - deterministic : title co-occurrence in sentence text (zero cost, zero leakage)
  - llm           : LLM extracts typed (subject, predicate, object) triples
                    from sentences; answer is never passed to the prompt

Entry point: build_context_kg(sentence_records, question, config)
"""
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.llm_client import load_local_env, openai_compatible_client, parse_model_ref


# ── Text utilities ─────────────────────────────────────────────────────────────

def clean_text(text: Any) -> str:
    text = str(text or "").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def phrase_in_text(phrase: str, text: str) -> bool:
    p = normalize(phrase)
    t = normalize(text)
    if len(p) < 3 or not t:
        return False
    return f" {p} " in f" {t} "


def dedupe(triples: List[List[str]]) -> List[List[str]]:
    out, seen = [], set()
    for s, p, o in triples:
        s, p, o = clean_text(s), clean_text(p), clean_text(o)
        if not s or not p or not o or s == o:
            continue
        key = (s.lower(), p.lower(), o.lower())
        if key not in seen:
            seen.add(key)
            out.append([s, p, o])
    return out


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class ContextKGConfig:
    backend: str = "deterministic"   # "deterministic" | "llm" | "llm_with_deterministic"
    model: str = "openrouter:google/gemma-4-31b-it:free"
    max_triples: int = 64
    max_context_sentences: int = 20
    sleep_seconds: float = 0.0
    request_timeout: float = 90.0
    max_retries: int = 4
    retry_initial_sleep: float = 5.0


# ── Deterministic backend ──────────────────────────────────────────────────────

def build_deterministic_kg(
    sentence_records: List[Dict[str, Any]],
    max_triples: int = 64,
) -> List[List[str]]:
    """
    Extract triples purely from title co-occurrence in sentence text.

    For each sentence in article A: if it mentions the title of article B,
    emit (A, mentions_entity, B).

    No answer, no gold evidences — zero leakage.
    """
    titles = []
    seen_t: set = set()
    for rec in sentence_records:
        t = clean_text(rec.get("title", ""))
        if t and t not in seen_t:
            seen_t.add(t)
            titles.append(t)

    triples: List[List[str]] = []
    seen: set = set()

    def add(s: str, p: str, o: str) -> None:
        if len(triples) >= max_triples:
            return
        key = (s.lower(), p.lower(), o.lower())
        if key not in seen:
            seen.add(key)
            triples.append([s, p, o])

    for rec in sentence_records:
        src = clean_text(rec.get("title", ""))
        sentence = rec.get("sentence", "")
        for tgt in titles:
            if tgt == src:
                continue
            if phrase_in_text(tgt, sentence):
                add(src, "mentions_entity", tgt)
        if len(triples) >= max_triples:
            break

    return triples


# ── LLM backend ────────────────────────────────────────────────────────────────

def _build_llm_prompt(
    question: str,
    sentence_records: List[Dict[str, Any]],
    max_triples: int,
) -> str:
    lines = []
    for i, rec in enumerate(sentence_records, 1):
        title    = clean_text(rec.get("title", ""))
        sentence = clean_text(rec.get("sentence", ""))
        lines.append(f"[{i}] title={title}; sentence={sentence}")
    evidence = "\n".join(lines)

    return f"""Extract a compact knowledge graph from the following evidence sentences to help answer a multi-hop question.

Question: {clean_text(question)}

Evidence:
{evidence}

Rules:
- Use ONLY entities and relations stated in the evidence sentences.
- Focus on triples that connect article titles, named entities, and relations relevant to the question.
- Use short, readable predicates: "directed_by", "born_in", "spouse_of", "member_of", "located_in", etc.
- Do NOT use sentence indices as entities.
- Do NOT include the answer — extract only what is stated in the evidence.
- Return at most {max_triples} triples.

Return valid JSON only:
{{
  "triples": [
    {{"subject": "...", "predicate": "...", "object": "..."}}
  ]
}}
"""


def _parse_llm_response(text: str) -> List[List[str]]:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    obj: Any = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                pass

    if isinstance(obj, dict):
        raw = obj.get("triples", [])
    elif isinstance(obj, list):
        raw = obj
    else:
        return []

    out = []
    for item in raw:
        if isinstance(item, dict):
            s = item.get("subject", item.get("s", ""))
            p = item.get("predicate", item.get("p", item.get("relation", "")))
            o = item.get("object", item.get("o", ""))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            s, p, o = item[0], item[1], item[2]
        else:
            continue
        s, p, o = clean_text(s), clean_text(p), clean_text(o)
        if s and p and o and s != o:
            out.append([s, p, o])
    return dedupe(out)


class LLMContextKGBuilder:
    def __init__(self, config: ContextKGConfig):
        self.config = config
        self._client = None
        load_local_env()

    def _get_client(self):
        if self._client is None:
            self._client, _, _ = openai_compatible_client(self.config.model)
        return self._client

    def build(
        self,
        sentence_records: List[Dict[str, Any]],
        question: str,
    ) -> List[List[str]]:
        records = sentence_records[: self.config.max_context_sentences]
        prompt  = _build_llm_prompt(question, records, self.config.max_triples)
        client  = self._get_client()
        model_name = parse_model_ref(self.config.model).model

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract concise knowledge-graph triples from "
                        "multi-hop QA evidence. The answer is unknown — extract "
                        "only facts stated in the evidence. Return JSON only."
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
                    **kwargs, timeout=self.config.request_timeout
                )
                break
            except Exception as exc:
                if attempt >= self.config.max_retries:
                    raise
                msg = str(exc).lower()
                sleep = self.config.retry_initial_sleep * (2 ** attempt)
                if "429" not in msg and "rate" not in msg:
                    sleep = min(sleep, self.config.retry_initial_sleep)
                print(
                    f"  LLM KG retry {attempt+1}/{self.config.max_retries}: {exc!r}; "
                    f"sleeping {sleep:.1f}s",
                    flush=True,
                )
                time.sleep(sleep)

        if self.config.sleep_seconds > 0:
            time.sleep(self.config.sleep_seconds)
        if response is None:
            return []
        return _parse_llm_response(response.choices[0].message.content)


# ── Main entry point ───────────────────────────────────────────────────────────

def build_context_kg(
    sentence_records: List[Dict[str, Any]],
    question: str,
    config: Optional[ContextKGConfig] = None,
    llm_builder: Optional[LLMContextKGBuilder] = None,
) -> Tuple[List[List[str]], str]:
    """
    Build a KG from context sentences only. No answer, no gold evidences.

    Returns (triples, method_used).

    Backends:
      deterministic           — title co-occurrence only, zero cost
      llm                     — LLM extracts typed triples from sentences
      llm_with_deterministic  — LLM triples merged with deterministic triples
    """
    if config is None:
        config = ContextKGConfig()

    if config.backend in {"llm", "llm_with_deterministic"} and llm_builder is not None:
        llm_triples = llm_builder.build(sentence_records, question)
        if config.backend == "llm_with_deterministic":
            det_triples = build_deterministic_kg(sentence_records, config.max_triples)
            merged = dedupe([*llm_triples, *det_triples])[: config.max_triples]
            return merged, "llm_with_deterministic"
        return llm_triples[: config.max_triples], "llm"

    triples = build_deterministic_kg(sentence_records, config.max_triples)
    return triples, "deterministic"
