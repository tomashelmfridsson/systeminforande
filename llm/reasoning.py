from llm.client import get_llm_client
from llm.prompts import base_llm_instructions, reasoning_prompt

def _call_llm(prompt: str, model: str | None = None) -> str:
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

    return response.choices[0].message["content"]

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
