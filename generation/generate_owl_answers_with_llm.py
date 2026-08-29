import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.adaptive_support_aggregation import adaptive_support_aggregate
from evaluation.adaptive_support_aggregation_v2 import (
    AdaptiveV2Policy,
    adaptive_v2_support_aggregate,
    load_domain_policy,
)
from utils.llm_client import openai_compatible_client, parse_model_ref


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_unit(unit: str) -> str:
    """
    Light cleanup for OWL/fact/axiom strings.
    Keeps the content faithful but makes it easier for the LLM to read.
    """
    unit = str(unit)
    unit = unit.replace("\n", " ").replace("\t", " ")
    unit = re.sub(r"\s+", " ", unit).strip()

    # Optional URI fragment simplification:
    # <http://x#hasMother> -> hasMother
    unit = re.sub(r"<[^#>\s]+#([^>]+)>", r"\1", unit)

    return unit


def classify_owl_unit(unit: str) -> str:
    u = str(unit)
    schema_markers = [
        "SubObjectPropertyOf",
        "SubClassOf",
        "EquivalentClasses",
        "EquivalentObjectProperties",
        "InverseObjectProperties",
        "TransitiveObjectProperty",
        "SymmetricObjectProperty",
        "FunctionalObjectProperty",
        "ObjectPropertyDomain",
        "ObjectPropertyRange",
        "PropertyChain",
        "subPropertyOf",
        "subClassOf",
        "inverseOf",
        "domain",
        "range",
    ]

    if any(m in u for m in schema_markers):
        return "Axiom"

    return "Fact"


def get_top_support_units(
    item: Dict[str, Any],
    top_k: int,
    *,
    aggregation_mode: str = "fixed",
    adaptive_tau: float | None = None,
    adaptive_k_max: int = 5,
    adaptive_v2_policy: AdaptiveV2Policy | None = None,
) -> List[str]:
    """
    Uses top5 list if available, otherwise top1_subgraph_units.
    Returns union of support units from top-k candidates, preserving order.
    """
    units: List[str] = []
    seen = set()

    if item.get("evaluation_scope") == "answer_only":
        return [clean_unit(u) for u in item.get("answer_context_units", [])]

    top_list = item.get("top5") or item.get("topk") or []

    if isinstance(top_list, list) and top_list:
        top_list = sorted(top_list, key=lambda x: int(x.get("rank", 999)))

        if aggregation_mode == "adaptive":
            if adaptive_tau is None:
                raise ValueError("adaptive_tau is required in adaptive aggregation mode")
            return adaptive_support_aggregate(
                top_list,
                tau=adaptive_tau,
                k_max=adaptive_k_max,
            )["support_units"]
        if aggregation_mode == "adaptive_v2":
            if adaptive_v2_policy is None:
                raise ValueError(
                    "adaptive_v2_policy is required in adaptive_v2 aggregation mode"
                )
            return adaptive_v2_support_aggregate(
                top_list,
                policy=adaptive_v2_policy,
                k_max=adaptive_k_max,
            )["support_units"]
        if aggregation_mode != "fixed":
            raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")

        for cand in top_list[:top_k]:
            for u in cand.get("subgraph_units", []):
                if u not in seen:
                    seen.add(u)
                    units.append(u)

    if not units:
        for u in item.get("top1_subgraph_units", []):
            if u not in seen:
                seen.add(u)
                units.append(u)

    return units


def is_boolean_question(question: str) -> bool:
    q = str(question or "").strip().lower()
    return bool(
        re.match(
            r"^(is|are|was|were|do|does|did|has|have|had|can|could|should|would)\b",
            q,
        )
    )


def normalize_generated_answer(question: str, answer: str) -> str:
    answer = str(answer or "").strip()
    if not answer:
        return answer

    if is_boolean_question(question):
        normalized = answer.lower().strip()
        if normalized in {"true", "yes", "1"} or normalized.startswith("yes,"):
            return "TRUE"
        if normalized in {"false", "no", "0"} or normalized.startswith("no,"):
            return "FALSE"
        if "unknown" in normalized or "insufficient" in normalized:
            return "Unknown"

    return answer


def local_name(value: str) -> str:
    value = clean_unit(value)
    if "#" in value:
        value = value.split("#")[-1]
    if "/" in value and value.startswith("http"):
        value = value.rstrip("/").split("/")[-1]
    return value.strip("<>")


def canonical_relation(value: str) -> str:
    value = local_name(value)
    if ":" in value:
        value = value.split(":", 1)[-1]
    return value.lower()


def parse_ask_triple(item: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    """
    Extract the triple pattern from a simple ASK WHERE query.
    """
    query = str(item.get("sparql_query", "") or "")
    if not query:
        example_id = str(item.get("example_id", ""))
        if "ASK WHERE" in example_id:
            query = "ASK WHERE" + example_id.split("ASK WHERE", 1)[-1]

    uris = re.findall(r"<([^>]+)>", query)
    if len(uris) < 3:
        return None

    return local_name(uris[0]), local_name(uris[1]), local_name(uris[2])


def parse_ask_type_target(item: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Extract the subject/class pair from an ASK rdf:type query when available.

    FamilyOWL example ids include the SPARQL query, while some intermediate
    detail files may not carry a separate sparql_query field.
    """
    triple = parse_ask_triple(item)
    if triple is None:
        return None

    subject, predicate, obj = triple
    if canonical_relation(predicate) != "type":
        return None

    return subject, obj


def parse_owl_support(units: List[str]) -> Dict[str, Any]:
    triples = []
    types: Dict[str, set[str]] = {}
    domains: Dict[str, set[str]] = {}
    ranges: Dict[str, set[str]] = {}
    subclasses: Dict[str, set[str]] = {}
    subproperties: Dict[str, set[str]] = {}
    inverses: Dict[str, set[str]] = {}
    symmetric: set[str] = set()
    transitive: set[str] = set()

    def add(mapping: Dict[str, set[str]], key: str, value: str) -> None:
        mapping.setdefault(local_name(key), set()).add(local_name(value))

    for unit in units:
        cleaned = clean_unit(unit)

        func_match = re.match(r"([A-Za-z0-9_:]+)\(([^)]*)\)", cleaned)
        if func_match:
            fun = canonical_relation(func_match.group(1))
            args = [
                local_name(arg.strip())
                for arg in func_match.group(2).split(",")
                if arg.strip()
            ]
            if fun == "inverseobjectproperties" and len(args) >= 2:
                add(inverses, args[0], args[1])
                add(inverses, args[1], args[0])
            elif fun == "subobjectpropertyof" and len(args) >= 2:
                add(subproperties, args[0], args[1])
            elif fun == "symmetricobjectproperty" and args:
                symmetric.add(local_name(args[0]))
            elif fun == "transitiveobjectproperty" and args:
                transitive.add(local_name(args[0]))
            continue

        parts = cleaned.split()
        if len(parts) < 3:
            continue

        head = local_name(parts[0])
        rel = local_name(parts[1])
        tail = local_name(" ".join(parts[2:]))
        rel_lower = canonical_relation(rel)

        if rel_lower == "type" and tail in {
            "SymmetricProperty",
            "SymmetricObjectProperty",
        }:
            symmetric.add(head)
        elif rel_lower == "type" and tail in {
            "TransitiveProperty",
            "TransitiveObjectProperty",
        }:
            transitive.add(head)
        elif rel_lower == "type":
            add(types, head, tail)
        elif rel_lower == "domain":
            add(domains, head, tail)
        elif rel_lower == "range":
            add(ranges, head, tail)
        elif rel_lower == "subclassof":
            add(subclasses, head, tail)
        elif rel_lower in {"subpropertyof", "subobjectpropertyof"}:
            add(subproperties, head, tail)
        elif rel_lower == "inverseof":
            add(inverses, head, tail)
            add(inverses, tail, head)
        else:
            triples.append((head, rel, tail))

    return {
        "triples": triples,
        "types": types,
        "domains": domains,
        "ranges": ranges,
        "subclasses": subclasses,
        "subproperties": subproperties,
        "inverses": inverses,
        "symmetric": symmetric,
        "transitive": transitive,
    }


def subclass_closure(cls: str, subclasses: Dict[str, set[str]]) -> set[str]:
    seen = {local_name(cls)}
    agenda = [local_name(cls)]

    while agenda:
        cur = agenda.pop()
        for parent in subclasses.get(cur, set()):
            if parent not in seen:
                seen.add(parent)
                agenda.append(parent)

    return seen


def property_closure(prop: str, subproperties: Dict[str, set[str]]) -> set[str]:
    seen = {local_name(prop)}
    agenda = [local_name(prop)]

    while agenda:
        cur = agenda.pop()
        for parent in subproperties.get(cur, set()):
            if parent not in seen:
                seen.add(parent)
                agenda.append(parent)

    return seen


def infer_types(parsed: Dict[str, Any]) -> Tuple[Dict[str, set[str]], List[str]]:
    types = {k: set(v) for k, v in parsed["types"].items()}
    proof_steps = []

    def add_type(entity: str, cls: str, step: str) -> None:
        entity = local_name(entity)
        cls = local_name(cls)
        if cls not in types.setdefault(entity, set()):
            types[entity].add(cls)
            proof_steps.append(step)

    for entity, cls_set in parsed["types"].items():
        for cls in cls_set:
            proof_steps.append(f"{entity} rdf:type {cls}")

    for head, rel, tail in parsed["triples"]:
        rels = property_closure(rel, parsed["subproperties"])
        for rel_name in rels:
            for cls in parsed["domains"].get(rel_name, set()):
                add_type(head, cls, f"{head} {rel} {tail}; {rel_name} domain {cls}")
            for cls in parsed["ranges"].get(rel_name, set()):
                add_type(tail, cls, f"{head} {rel} {tail}; {rel_name} range {cls}")

            for inv_rel in parsed["inverses"].get(rel_name, set()):
                inv_rels = property_closure(inv_rel, parsed["subproperties"])
                for inv_rel_name in inv_rels:
                    for cls in parsed["domains"].get(inv_rel_name, set()):
                        add_type(
                            tail,
                            cls,
                            f"{head} {rel} {tail}; {inv_rel_name} inverseOf "
                            f"{rel_name}; {inv_rel_name} domain {cls}",
                        )
                    for cls in parsed["ranges"].get(inv_rel_name, set()):
                        add_type(
                            head,
                            cls,
                            f"{head} {rel} {tail}; {inv_rel_name} inverseOf "
                            f"{rel_name}; {inv_rel_name} range {cls}",
                        )

    return types, proof_steps


def infer_property_assertions(
    parsed: Dict[str, Any],
) -> Dict[Tuple[str, str, str], str]:
    entailed: Dict[Tuple[str, str, str], str] = {}

    def add(head: str, rel: str, tail: str, proof: str) -> None:
        entailed.setdefault(
            (local_name(head), local_name(rel), local_name(tail)), proof
        )

    for head, rel, tail in parsed["triples"]:
        for rel_name in property_closure(rel, parsed["subproperties"]):
            add(head, rel_name, tail, f"{head} {rel} {tail}")

            if rel_name in parsed["symmetric"]:
                add(
                    tail,
                    rel_name,
                    head,
                    f"{head} {rel} {tail}; {rel_name} is symmetric",
                )

            for inv_rel in parsed["inverses"].get(rel_name, set()):
                for inv_rel_name in property_closure(inv_rel, parsed["subproperties"]):
                    add(
                        tail,
                        inv_rel_name,
                        head,
                        f"{head} {rel} {tail}; {inv_rel_name} inverseOf {rel_name}",
                    )

    return entailed


def infer_owl_boolean_answer(
    item: Dict[str, Any], support_units: List[str]
) -> Optional[Dict[str, str]]:
    """
    Deterministic proof layer for simple OWL ASK rdf:type questions.

    It recognizes direct type assertions, subclass propagation, domain/range
    typing, and one-step inverse-property domain/range typing.
    """
    if not is_boolean_question(str(item.get("question", ""))):
        return None

    ask_triple = parse_ask_triple(item)
    if ask_triple is None:
        return None

    subject, predicate, obj = ask_triple
    parsed = parse_owl_support(support_units)

    if canonical_relation(predicate) == "type":
        target_class = obj
        types, proof_steps = infer_types(parsed)
        entailed = set()
        for cls in types.get(subject, set()):
            entailed.update(subclass_closure(cls, parsed["subclasses"]))

        if target_class in entailed:
            return {
                "answer": "TRUE",
                "explanation": (
                    f"Deterministic OWL proof: {subject} is entailed to have type "
                    f"{target_class} from retrieved support. "
                    f"Steps considered: {'; '.join(proof_steps[:5])}."
                ),
            }
    else:
        entailed_properties = infer_property_assertions(parsed)
        target_key = (subject, predicate, obj)
        if target_key in entailed_properties:
            return {
                "answer": "TRUE",
                "explanation": (
                    f"Deterministic OWL proof: retrieved support entails "
                    f"{subject} {predicate} {obj}. "
                    f"Step: {entailed_properties[target_key]}."
                ),
            }

    return None


def query_terms(item: Dict[str, Any], question: str) -> set[str]:
    terms = set()
    triple = parse_ask_triple(item)
    if triple:
        for value in triple:
            terms.add(local_name(value).lower())

    for token in re.findall(r"[A-Za-z0-9_]+", question):
        if len(token) > 2:
            terms.add(token.lower())

    return terms


def order_support_for_reader(
    item: Dict[str, Any], support_units: List[str], question: str
) -> List[str]:
    """
    Put likely proof-relevant facts/axioms first for the LLM fallback.
    """
    terms = query_terms(item, question)

    def score(unit: str) -> Tuple[int, int]:
        cleaned = clean_unit(unit)
        lowered = cleaned.lower()
        overlap = sum(1 for term in terms if term and term in lowered)
        is_fact = 1 if classify_owl_unit(cleaned) == "Fact" else 0
        return overlap, is_fact

    return sorted(support_units, key=score, reverse=True)


def owl_reasoning_line(unit: str) -> str:
    cleaned = clean_unit(unit)
    unit_type = classify_owl_unit(cleaned)

    # Many FamilyOWL units are readable triples already: subject relation object.
    parts = cleaned.split()
    if len(parts) >= 3 and "(" not in parts[0]:
        head = parts[0]
        relation = parts[1]
        tail = " ".join(parts[2:])
        return f"{unit_type}: {head} -> {relation} -> {tail}"

    return f"{unit_type}: {cleaned}"


def build_prompt(question: str, support_units: List[str]) -> str:
    reasoning_lines = []

    for i, unit in enumerate(support_units, start=1):
        reasoning_lines.append(f"[{i}] {owl_reasoning_line(unit)}")

    reasoning_context = (
        "\n".join(reasoning_lines)
        if reasoning_lines
        else "No retrieved reasoning paths."
    )

    if is_boolean_question(question):
        answer_format = (
            'This is a yes/no ontology question. The "answer" value must be exactly '
            '"TRUE" or "FALSE". Do not return "Unknown".'
        )
        uncertainty_instruction = (
            "- If the reasoning paths do not fully settle the question, choose the "
            "better supported binary label from the retrieved facts and axioms."
        )
    else:
        answer_format = (
            'The "answer" value must be the shortest entity/name/value that answers '
            "the question. For multiple answers, separate items with semicolons."
        )
        uncertainty_instruction = (
            '- If the reasoning paths are insufficient, answer "Unknown"; do not guess.'
        )

    return f"""You are answering an ontology-grounded question using retrieved reasoning paths.

Reasoning Paths:
{reasoning_context}

Question:
{question}

Instructions:
- Answer using only the retrieved reasoning paths.
- Treat Facts as instance-level evidence and Axioms as schema/rule evidence.
- If a fact and the necessary axiom together entail the question, answer with the entailed result.
{uncertainty_instruction}
- {answer_format}
- Return valid JSON only, with this exact schema:
{{"answer": "...", "explanation": "..."}}
"""


def parse_json_response(text: str) -> Dict[str, str]:
    text = text.strip()

    # Remove markdown fences if present.
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        obj = json.loads(text)
        return {
            "answer": str(obj.get("answer", "")).strip(),
            "explanation": str(obj.get("explanation", "")).strip(),
        }
    except Exception:
        pass

    # Fallback: try to find JSON substring.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return {
                "answer": str(obj.get("answer", "")).strip(),
                "explanation": str(obj.get("explanation", "")).strip(),
            }
        except Exception:
            pass

    return {
        "answer": text.strip(),
        "explanation": "",
    }


def call_openai(prompt: str, model: str, max_tokens: int = 300) -> str:
    client, model_name, _ = openai_compatible_client(model)

    kwargs = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You answer questions using only provided evidence and return valid JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "max_tokens": max_tokens,
    }

    # GPT-5 family may reject temperature=0.0, so omit temperature for gpt-5*.
    if not model_name.lower().startswith("gpt-5"):
        kwargs["temperature"] = 0.0

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def generate_answers(
    details_path: str,
    output_path: str,
    top_k: int,
    backend: str,
    model: str,
    answer_only_paths: Optional[List[str]] = None,
    max_examples: int = 0,
    resume: bool = False,
    aggregation_mode: str = "fixed",
    adaptive_tau: float | None = None,
    adaptive_k_max: int = 5,
    adaptive_v2_policy: AdaptiveV2Policy | None = None,
):
    details = load_json(details_path)
    answer_only_paths = answer_only_paths or []

    for path in answer_only_paths:
        if Path(path).exists():
            details.extend(load_jsonl(path))

    if max_examples and max_examples > 0:
        details = details[:max_examples]

    done_ids = set()
    done_index: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []

    if resume and Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    example_id = row.get("example_id")
                    rows.append(row)
                    done_ids.add(example_id)
                    done_index[example_id] = len(rows) - 1

    for idx, item in enumerate(details, start=1):
        example_id = item.get("example_id", "")

        question = str(item.get("question", ""))
        gold_answer = str(item.get("answer", item.get("gold_answer", "")))
        support_units = get_top_support_units(
            item,
            top_k=top_k,
            aggregation_mode=aggregation_mode,
            adaptive_tau=adaptive_tau,
            adaptive_k_max=adaptive_k_max,
            adaptive_v2_policy=adaptive_v2_policy,
        )

        if resume and example_id in done_ids:
            proof = infer_owl_boolean_answer(item=item, support_units=support_units)
            if proof is not None:
                old_row = rows[done_index[example_id]]
                if old_row.get("predicted_answer") != proof["answer"]:
                    old_row["predicted_answer"] = proof["answer"]
                    old_row["explanation"] = proof["explanation"]
                    old_row["raw_response"] = json.dumps(
                        {
                            "answer": proof["answer"],
                            "explanation": proof["explanation"],
                        },
                        ensure_ascii=False,
                    )
                    old_row["answer_source"] = "deterministic_owl_proof"
                    old_row["support_units"] = support_units
                    write_jsonl(output_path, rows)
            continue

        row = {
            "example_id": example_id,
            "dataset": item.get("dataset", ""),
            "hop": item.get("hop", ""),
            "answer_type": item.get("answer_type", ""),
            "evaluation_scope": item.get("evaluation_scope", "support"),
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": "",
            "explanation": "",
            "support_units": support_units,
            "top_k": top_k,
            "aggregation_mode": aggregation_mode,
            "adaptive_tau": adaptive_tau,
            "adaptive_k_max": (
                adaptive_k_max
                if aggregation_mode in {"adaptive", "adaptive_v2"}
                else None
            ),
            "adaptive_v2_domain": (
                adaptive_v2_policy.domain if adaptive_v2_policy is not None else None
            ),
        }

        try:
            proof = infer_owl_boolean_answer(item=item, support_units=support_units)

            if proof is not None:
                row["predicted_answer"] = proof["answer"]
                row["explanation"] = proof["explanation"]
                row["raw_response"] = json.dumps(
                    {"answer": proof["answer"], "explanation": proof["explanation"]},
                    ensure_ascii=False,
                )
                row["answer_source"] = "deterministic_owl_proof"
                rows.append(row)
                write_jsonl(output_path, rows)
                continue

            reader_support_units = order_support_for_reader(
                item=item,
                support_units=support_units,
                question=question,
            )
            row["support_units"] = reader_support_units
            prompt = build_prompt(question=question, support_units=reader_support_units)

            if backend == "openai":
                raw = call_openai(prompt=prompt, model=model)
            else:
                raise ValueError(f"Unknown backend: {backend}")

            parsed = parse_json_response(raw)

            row["predicted_answer"] = normalize_generated_answer(
                question, parsed["answer"]
            )
            row["explanation"] = parsed["explanation"]
            row["raw_response"] = raw

        except Exception as e:
            row["error"] = repr(e)

        rows.append(row)

        # Write incrementally for safety.
        write_jsonl(output_path, rows)

        if idx % 25 == 0:
            print(f"Generated {idx}/{len(details)}")

    print(f"Generated {len(rows)}/{len(details)}")
    print(f"Saved generated answers to: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--details", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--aggregation-mode", choices=("fixed", "adaptive", "adaptive_v2"), default="fixed"
    )
    parser.add_argument("--adaptive-tau", type=float, default=None)
    parser.add_argument("--adaptive-k-max", type=int, default=5)
    parser.add_argument("--adaptive-v2-policy-dir", type=Path, default=None)
    parser.add_argument("--adaptive-v2-domain", choices=("text", "ontology"), default=None)
    parser.add_argument("--backend", type=str, default="openai", choices=["openai"])
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--answer-only", type=str, nargs="*", default=[])
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if args.aggregation_mode == "adaptive" and args.adaptive_tau is None:
        parser.error("--adaptive-tau is required with --aggregation-mode adaptive")
    if args.aggregation_mode == "adaptive_v2" and (
        args.adaptive_v2_policy_dir is None or args.adaptive_v2_domain is None
    ):
        parser.error(
            "--adaptive-v2-policy-dir and --adaptive-v2-domain are required "
            "with --aggregation-mode adaptive_v2"
        )
    adaptive_v2_policy = (
        load_domain_policy(args.adaptive_v2_policy_dir, domain=args.adaptive_v2_domain)
        if args.aggregation_mode == "adaptive_v2"
        else None
    )

    if args.backend == "openai":
        ref = parse_model_ref(args.model)
        if ref.provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
            raise EnvironmentError(
                "Model provider is openrouter, but OPENROUTER_API_KEY is not set."
            )
        if ref.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "Model provider is openai, but OPENAI_API_KEY is not set."
            )
        if ref.provider is None and not (
            os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        ):
            raise EnvironmentError(
                "Neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set. In PowerShell, run: "
                '$env:OPENROUTER_API_KEY="YOUR_KEY"'
            )

    generate_answers(
        details_path=args.details,
        output_path=args.output,
        top_k=args.top_k,
        backend=args.backend,
        model=args.model,
        answer_only_paths=args.answer_only,
        max_examples=args.max_examples,
        resume=args.resume,
        aggregation_mode=args.aggregation_mode,
        adaptive_tau=args.adaptive_tau,
        adaptive_k_max=args.adaptive_k_max,
        adaptive_v2_policy=adaptive_v2_policy,
    )


if __name__ == "__main__":
    main()
