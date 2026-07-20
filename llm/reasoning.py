from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm.client import get_llm_client
from llm.prompts import base_llm_instructions, reasoning_prompt


@dataclass(frozen=True)
class LLMCallResult:
    text: str
    usage: dict[str, int | None]


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_usage_value(usage: Any, key: str) -> Any:
    value = getattr(usage, key, None)
    if value is not None:
        return value
    try:
        return usage.get(key)
    except AttributeError:
        return None


def extract_hf_usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if not usage:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

    return {
        "prompt_tokens": _safe_int(_get_usage_value(usage, "prompt_tokens")),
        "completion_tokens": _safe_int(_get_usage_value(usage, "completion_tokens")),
        "total_tokens": _safe_int(_get_usage_value(usage, "total_tokens")),
    }


def _call_llm_with_usage(prompt: str, model: str | None = None) -> LLMCallResult:
    client = get_llm_client(model=model)
    response = client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": base_llm_instructions()
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1200,
        temperature=0.2
    )

    return LLMCallResult(
        text=response.choices[0].message["content"],
        usage=extract_hf_usage(response),
    )


def _call_llm(prompt: str, model: str | None = None) -> str:
    return _call_llm_with_usage(prompt, model=model).text

# =========================
# FÖRDEFINIERADE FRÅGOR
# =========================

def generate_reasoning(
    *,
    title: str,
    main_question: str,
    question: str,
    answer: dict,
    model: str | None = None,
) -> str:
    """
    Genererar resonemang för fördefinierade frågor.
    """
    prompt = reasoning_prompt(
        title=title,
        main_question=main_question,
        question=question,
        answer=_format_answer(answer)
    )

    return _call_llm(prompt, model=model)


# =========================
# RAG (FRITEXT)
# =========================

def generate_reasoning_from_prompt(prompt: str, model: str | None = None) -> str:
    """
    Genererar resonemang från färdig prompt (RAG).
    """
    return _call_llm(prompt, model=model)


def generate_reasoning_from_prompt_with_usage(prompt: str, model: str | None = None) -> LLMCallResult:
    """
    Genererar resonemang från färdig prompt (RAG) och behåller tokenanvändning.
    """
    return _call_llm_with_usage(prompt, model=model)


# =========================
# FORMATTERING
# =========================

def _format_answer(answer: dict) -> str:
    """
    Gör om svar-objekt (Beskrivning, Exempel, Lista, etc.)
    till stabil text för LLM.
    """
    lines = []
    for key, value in answer.items():
        lines.append(f"{key}:")
        if isinstance(value, list):
            for item in value:
                lines.append(f"- {item}")
        else:
            lines.append(str(value))
        lines.append("")
    return "\n".join(lines)
