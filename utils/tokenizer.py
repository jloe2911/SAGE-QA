from transformers import (
    AutoTokenizer,
    BertTokenizer,
    DistilBertTokenizer,
)


def _load_with_local_fallback(loader, model_name: str, **kwargs):
    try:
        return loader.from_pretrained(model_name, local_files_only=True, **kwargs)
    except Exception:
        return loader.from_pretrained(model_name, **kwargs)


def load_tokenizer(model_name: str):
    normalized_name = model_name.lower()

    if "distilbert" in normalized_name:
        return _load_with_local_fallback(DistilBertTokenizer, model_name)

    if "bert" in normalized_name:
        return _load_with_local_fallback(BertTokenizer, model_name)

    return _load_with_local_fallback(AutoTokenizer, model_name, use_fast=False)
