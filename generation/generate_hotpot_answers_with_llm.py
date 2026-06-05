import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.llm_client import openai_compatible_client


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_local_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_sentence_unit(unit: str) -> Tuple[str, int, str]:
    """
    SENT::Page Title::3::sentence text
    -> title, index, sentence
    """
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


def get_hotpot_id(example_id: str) -> str:
    prefix = "HotpotQA__test__"
    if example_id.startswith(prefix):
        return example_id[len(prefix) :]

    parts = example_id.split("__")
    if len(parts) >= 3 and parts[0] == "HotpotQA":
        return parts[-1]

    return example_id


def get_support_units(item: Dict[str, Any], top_k: int) -> List[str]:
    """
    Returns union of sentence units from top-k candidates.
    For top_k=1, this is just the top-ranked support candidate.
    """
    units = []
    seen = set()

    top_list = item.get("top5") or item.get("topk") or []
    if isinstance(top_list, list) and top_list:
        top_list = sorted(top_list, key=lambda x: int(x.get("rank", 999)))
        for cand in top_list[:top_k]:
            for u in cand.get("subgraph_units", []):
                if u not in seen:
                    seen.add(u)
                    units.append(u)

    if not units and item.get("top1_subgraph_units"):
        for u in item["top1_subgraph_units"]:
            if u not in seen:
                seen.add(u)
                units.append(u)

    return units


def detect_answer_guidance(question: str) -> str:
    q = str(question or "").strip()
    q_lower = q.lower()
    guidance = []

    if q_lower.startswith("which "):
        guidance.append(
            "This is a 'which' question. Return the requested item itself, "
            "not the intermediate date, director, author, parent, country, or explanation."
        )
        choices = extract_or_choices(q)
        if choices:
            guidance.append(
                "If the answer is one of the compared choices, copy exactly one "
                f"of these choices: {'; '.join(choices)}."
            )

    if q_lower.startswith(("where ", "what is the place", "what was the place")):
        guidance.append(
            "This asks for a place. Return only the minimal place/entity name "
            "needed to answer. Do not append broader locations unless the question "
            "explicitly asks for them."
        )

    if q_lower.startswith("who "):
        guidance.append(
            "This asks for a person or organization. Return only the name, without titles, roles, or appositive descriptions unless they are part of the name."
        )

    if "grandfather" in q_lower:
        guidance.append(
            "For grandfather questions, identify the parent first, then return that parent's father; do not return the parent."
        )

    if "born first" in q_lower or "older" in q_lower:
        guidance.append(
            "For older/born-first comparisons, return the person or item asked for, not the birth date."
        )

    if "came out first" in q_lower or "released first" in q_lower:
        guidance.append(
            "For release-order comparisons, return the film/item title that was released earlier, not the release date."
        )

    if "died first" in q_lower:
        guidance.append(
            "For died-first comparisons, return the person or item asked for, not the death date."
        )

    if q_lower.startswith(
        (
            "is ",
            "are ",
            "was ",
            "were ",
            "do ",
            "does ",
            "did ",
            "has ",
            "have ",
            "had ",
        )
    ):
        guidance.append(
            "This is a yes/no question. Compare the requested properties directly and answer exactly yes or no."
        )

    if not guidance:
        return "No additional answer-type guidance."
    return "\n".join(f"- {item}" for item in guidance)


def extract_or_choices(question: str) -> List[str]:
    q = re.sub(r"\s+", " ", str(question or "")).strip(" ?")
    patterns = [
        r",\s*([^,?]+?)\s+or\s+([^,?]+)$",
        r"\bbetween\s+(.+?)\s+and\s+(.+)$",
        r"\bboth\s+movies,\s*(.+?)\s+and\s+(.+?)(?:,|$)",
        r"\bboth\s+films,\s*(.+?)\s+and\s+(.+?)(?:,|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            choices = [clean_choice(match.group(1)), clean_choice(match.group(2))]
            return [choice for choice in choices if choice]
    return []


def clean_choice(choice: str) -> str:
    choice = re.sub(
        r"^(?:the|film|movie|person|song)\s+",
        "",
        str(choice).strip(),
        flags=re.IGNORECASE,
    )
    return choice.strip(" ,;:")


def build_prompt(question: str, support_units: List[str]) -> str:
    reasoning_lines = []
    for i, unit in enumerate(support_units, start=1):
        title, idx, sent = parse_sentence_unit(unit)
        if title and idx >= 0:
            reasoning_lines.append(f"[{i}] {title} -> sentence {idx} -> {sent}")
        else:
            reasoning_lines.append(f"[{i}] Evidence -> {sent}")

    reasoning_context = (
        "\n".join(reasoning_lines)
        if reasoning_lines
        else "No retrieved reasoning paths."
    )

    return f"""You are answering a multi-hop question using retrieved reasoning paths.

Reasoning Paths:
{reasoning_context}

Question:
{question}

Answer Type Guidance:
{detect_answer_guidance(question)}

Instructions:
- Use only the reasoning paths above. Do not use outside knowledge.
- First identify the minimal path evidence needed to answer, but do not show hidden reasoning outside JSON.
Return the shortest possible answer:
- For yes/no questions, answer exactly "yes" or "no".
- For entity, title, date, number, or place questions, copy the shortest exact answer span from the evidence when possible.
- If the question asks which of two named choices satisfies a comparison, the answer must be the choice name, not the evidence value used to compare them.
- Do not include parenthetical explanations, appositives, occupations, or broader locations unless they are required to identify the answer.
- For multiple answers, separate items with semicolons.
- If the reasoning paths are insufficient, answer "unknown".
- Provide a one-sentence explanation grounded in the reasoning paths.

Return valid JSON only, with exactly these keys:
{{
  "answer": "...",
  "explanation": "..."
}}
"""


def is_yes_no_question(question: str) -> bool:
    return bool(
        re.match(
            r"^\s*(is|are|was|were|do|does|did|has|have|had|can|could|should|would)\b",
            str(question or "").lower(),
        )
    )


def normalize_generated_answer(question: str, answer: str) -> str:
    answer = str(answer or "").strip()
    if not answer:
        return answer

    if is_yes_no_question(question):
        normalized = answer.lower()
        if normalized in {"yes", "true"} or normalized.startswith("yes,"):
            return "yes"
        if normalized in {"no", "false"} or normalized.startswith("no,"):
            return "no"

    if answer.lower() in {"unknown", "insufficient information"}:
        return "unknown"
    return answer


def is_unknown_answer(answer: str) -> bool:
    answer = str(answer or "").strip().lower()
    return answer in {
        "",
        "unknown",
        "insufficient information",
        "not enough information",
        "cannot be determined",
        "cannot determine",
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robustly extract JSON from an LLM response.
    """
    text = text.strip()

    # Remove markdown fences if present.
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try to find first JSON object.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Fallback: use full response as answer.
    return {
        "answer": text.strip(),
        "explanation": "",
    }


def call_openai(prompt: str, model: str, temperature: float = 0.0) -> str:
    """
    Calls OpenAI chat completion.

    Some newer models, e.g. gpt-5-mini, do not support custom temperature
    and require the default value. For those models, we omit temperature.
    """
    client, model_name, _ = openai_compatible_client(model)

    kwargs = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful question answering system. Use only the given evidence.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    # GPT-5 mini currently rejects explicit temperature=0.0.
    # Use default temperature for GPT-5-family models.
    if not model_name.startswith("gpt-5"):
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)

    return response.choices[0].message.content


def call_local_transformers(
    prompt: str, model_name: str, max_new_tokens: int = 128
) -> str:
    """
    Optional local generation fallback using transformers.
    This is slower and may be lower quality unless you use an instruction model.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Remove prompt prefix if model echoes it.
    if text.startswith(prompt):
        text = text[len(prompt) :].strip()

    return text.strip()


def generate_answers(
    details_path: str,
    output_path: str,
    top_k: int,
    backend: str,
    model: str,
    max_examples: int = 0,
    sleep_seconds: float = 0.0,
    fallback_top_k: int = 0,
):
    details = load_json(details_path)

    if max_examples and max_examples > 0:
        details = details[:max_examples]

    rows = []

    for i, item in enumerate(details, start=1):
        example_id = item.get("example_id", "")
        hotpot_id = get_hotpot_id(example_id)
        question = item.get("question", "")
        gold_answer = item.get("answer", "")

        support_units = get_support_units(item, top_k=top_k)

        try:

            def ask(units: List[str]) -> Tuple[str, Dict[str, Any], str]:
                prompt = build_prompt(question, units)
                if backend == "openai":
                    raw_text = call_openai(prompt=prompt, model=model)
                elif backend == "local":
                    raw_text = call_local_transformers(prompt=prompt, model_name=model)
                else:
                    raise ValueError(f"Unknown backend: {backend}")
                parsed_obj = extract_json_object(raw_text)
                answer_text = normalize_generated_answer(
                    question, str(parsed_obj.get("answer", "")).strip()
                )
                return answer_text, parsed_obj, raw_text

            predicted_answer, parsed, raw = ask(support_units)
            answer_source_top_k = top_k

            if (
                fallback_top_k
                and fallback_top_k > top_k
                and is_unknown_answer(predicted_answer)
            ):
                fallback_units = get_support_units(item, top_k=fallback_top_k)
                if fallback_units != support_units:
                    fallback_answer, fallback_parsed, fallback_raw = ask(fallback_units)
                    if not is_unknown_answer(fallback_answer):
                        support_units = fallback_units
                        predicted_answer = fallback_answer
                        parsed = fallback_parsed
                        raw = fallback_raw
                        answer_source_top_k = fallback_top_k

            explanation = str(parsed.get("explanation", "")).strip()

            row = {
                "example_id": example_id,
                "hotpot_id": hotpot_id,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "explanation": explanation,
                "support_units": support_units,
                "top_k": top_k,
                "answer_source_top_k": answer_source_top_k,
                "fallback_top_k": fallback_top_k,
                "raw_response": raw,
            }

        except Exception as e:
            row = {
                "example_id": example_id,
                "hotpot_id": hotpot_id,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": "",
                "explanation": "",
                "support_units": support_units,
                "top_k": top_k,
                "answer_source_top_k": top_k,
                "fallback_top_k": fallback_top_k,
                "error": repr(e),
            }

        rows.append(row)

        if i % 10 == 0:
            print(f"Generated {i}/{len(details)}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    write_jsonl(output_path, rows)
    print(f"Saved generated answers to: {output_path}")


def main():
    load_local_env()
    parser = argparse.ArgumentParser()

    parser.add_argument("--details", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--backend", type=str, choices=["openai", "local"], default="openai"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name or local HF model name.",
    )

    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--fallback-top-k",
        type=int,
        default=0,
        help=(
            "If the first reader call returns unknown/empty, retry with this "
            "larger top-k evidence union."
        ),
    )

    args = parser.parse_args()

    generate_answers(
        details_path=args.details,
        output_path=args.output,
        top_k=args.top_k,
        backend=args.backend,
        model=args.model,
        max_examples=args.max_examples,
        sleep_seconds=args.sleep_seconds,
        fallback_top_k=args.fallback_top_k,
    )


if __name__ == "__main__":
    main()
