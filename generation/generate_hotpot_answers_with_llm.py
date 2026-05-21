import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def build_prompt(question: str, support_units: List[str]) -> str:
    evidence_lines = []
    for i, unit in enumerate(support_units, start=1):
        title, idx, sent = parse_sentence_unit(unit)
        if title and idx >= 0:
            evidence_lines.append(f"[{i}] ({title}, sentence {idx}) {sent}")
        else:
            evidence_lines.append(f"[{i}] {sent}")

    evidence = "\n".join(evidence_lines)

    return f"""You are answering a HotpotQA multi-hop question.

Use only the provided evidence. Do not use outside knowledge.
Return a short answer, not a full sentence, unless the answer is yes or no.
Then provide a brief explanation grounded in the evidence.

Question:
{question}

Evidence:
{evidence}

Return valid JSON with exactly these keys:
{{
  "answer": "...",
  "explanation": "..."
}}
"""


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
    from openai import OpenAI

    client = OpenAI()

    kwargs = {
        "model": model,
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
    if not model.startswith("gpt-5"):
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
        prompt = build_prompt(question, support_units)

        try:
            if backend == "openai":
                raw = call_openai(prompt=prompt, model=model)
            elif backend == "local":
                raw = call_local_transformers(prompt=prompt, model_name=model)
            else:
                raise ValueError(f"Unknown backend: {backend}")

            parsed = extract_json_object(raw)
            predicted_answer = str(parsed.get("answer", "")).strip()
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

    args = parser.parse_args()

    generate_answers(
        details_path=args.details,
        output_path=args.output,
        top_k=args.top_k,
        backend=args.backend,
        model=args.model,
        max_examples=args.max_examples,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()
