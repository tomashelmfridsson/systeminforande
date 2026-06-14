import os
from huggingface_hub import InferenceClient

_client = None

def get_llm_client():
    global _client
    if _client is None:
        _client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            token=os.environ.get("HF_TOKEN")
        )
    return _client