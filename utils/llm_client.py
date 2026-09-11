import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMModelRef:
    provider: str | None
    model: str


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


def parse_model_ref(model: str) -> LLMModelRef:
    model = str(model or "").strip()
    if ":" not in model:
        return LLMModelRef(provider=None, model=model)

    provider, model_name = model.split(":", 1)
    provider = provider.strip().lower()
    model_name = model_name.strip()
    if provider not in {"openrouter", "openai"} or not model_name:
        return LLMModelRef(provider=None, model=model)
    return LLMModelRef(provider=provider, model=model_name)


def openai_compatible_client(model: str):
    from openai import OpenAI

    load_local_env()
    ref = parse_model_ref(model)
    provider = ref.provider

    if provider is None:
        provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"

    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise EnvironmentError(
                "Model provider is openrouter, but OPENROUTER_API_KEY is not set."
            )
        return (
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
                default_headers={
                    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://github.com"),
                    "X-Title": os.getenv("OPENROUTER_APP_TITLE", "SAGE-QA"),
                },
            ),
            ref.model,
            provider,
        )

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("Model provider is openai, but OPENAI_API_KEY is not set.")
    return OpenAI(api_key=key), ref.model, provider
