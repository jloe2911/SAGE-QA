import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def get_top_support_units(item: Dict[str, Any], top_k: int) -> List[str]:
    """
    Uses top5 list if available, otherwise top1_subgraph_units.
    Returns union of support units from top-k candidates, preserving order.
    """
    units: List[str] = []
    seen = set()

    top_list = item.get("top5") or item.get("topk") or []

    if isinstance(top_list, list) and top_list:
        top_list = sorted(top_list, key=lambda x: int(x.get("rank", 999)))

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


def build_prompt(question: str, support_units: List[str]) -> str:
    evidence_lines = []

    for i, unit in enumerate(support_units, start=1):
        unit_type = classify_owl_unit(unit)
        evidence_lines.append(f"[{i}] {unit_type}: {clean_unit(unit)}")

    evidence = "\n".join(evidence_lines) if evidence_lines else "No retrieved support."

    return f"""You are answering an ontology-grounded question using only the retrieved OWL support.

Question:
{question}

Retrieved support:
{evidence}

Instructions:
- Answer using only the retrieved support.
- If the support is insufficient, answer "Unknown".
- Keep the answer short.
- Return valid JSON only with this schema:
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
    try:
        from openai import OpenAI
    except Exception as e:
        raise ImportError(
            "Could not import openai. Install with: pip install openai"
        ) from e

    client = OpenAI()

    kwargs = {
        "model": model,
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
    if not model.lower().startswith("gpt-5"):
        kwargs["temperature"] = 0.0

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def generate_answers(
    details_path: str,
    output_path: str,
    top_k: int,
    backend: str,
    model: str,
    max_examples: int = 0,
    resume: bool = False,
):
    details = load_json(details_path)

    if max_examples and max_examples > 0:
        details = details[:max_examples]

    done_ids = set()
    rows: List[Dict[str, Any]] = []

    if resume and Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows.append(row)
                    done_ids.add(row.get("example_id"))

    for idx, item in enumerate(details, start=1):
        example_id = item.get("example_id", "")
        if resume and example_id in done_ids:
            continue

        question = str(item.get("question", ""))
        gold_answer = str(item.get("answer", item.get("gold_answer", "")))
        support_units = get_top_support_units(item, top_k=top_k)

        row = {
            "example_id": example_id,
            "dataset": item.get("dataset", ""),
            "hop": item.get("hop", ""),
            "answer_type": item.get("answer_type", ""),
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": "",
            "explanation": "",
            "support_units": support_units,
            "top_k": top_k,
        }

        try:
            prompt = build_prompt(question=question, support_units=support_units)

            if backend == "openai":
                raw = call_openai(prompt=prompt, model=model)
            else:
                raise ValueError(f"Unknown backend: {backend}")

            parsed = parse_json_response(raw)

            row["predicted_answer"] = parsed["answer"]
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
    parser.add_argument("--backend", type=str, default="openai", choices=["openai"])
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if args.backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. In PowerShell, run: "
            '$env:OPENAI_API_KEY="YOUR_KEY"'
        )

    generate_answers(
        details_path=args.details,
        output_path=args.output,
        top_k=args.top_k,
        backend=args.backend,
        model=args.model,
        max_examples=args.max_examples,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
