import os
from huggingface_hub import InferenceClient

_clients = {}
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def get_llm_client(model: str | None = None):
    selected_model = model or DEFAULT_MODEL
    if selected_model not in _clients:
        _clients[selected_model] = InferenceClient(
            model=selected_model,
            token=os.environ.get("HF_TOKEN")
        )
    return _clients[selected_model]
