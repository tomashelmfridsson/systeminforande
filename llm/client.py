import os
from huggingface_hub import InferenceClient

_client = None

def get_llm_client():
    global _client
    if _client is None:
        _client = InferenceClient(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            token=os.environ.get("HF_TOKEN")
        )
    return _client
