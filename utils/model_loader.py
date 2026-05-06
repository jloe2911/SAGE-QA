from transformers import AutoModel, BertModel, DistilBertModel


def _load_with_local_fallback(loader, model_name: str):
    try:
        return loader.from_pretrained(
            model_name,
            local_files_only=True,
            use_safetensors=False,
        )
    except Exception:
        return loader.from_pretrained(model_name, use_safetensors=False)


def load_encoder(model_name: str):
    normalized_name = model_name.lower()

    if "distilbert" in normalized_name:
        return _load_with_local_fallback(DistilBertModel, model_name)

    if "bert" in normalized_name:
        return _load_with_local_fallback(BertModel, model_name)

    return _load_with_local_fallback(AutoModel, model_name)
