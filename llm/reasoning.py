from llm.client import get_llm_client
from llm.prompts import reasoning_prompt

# =========================
# LLM-KLIENT (EN ENDA)
# =========================

_client = get_llm_client()

def _call_llm(prompt: str) -> str:
    response = _client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "Du är en expertassistent.\n"
                    "Du ska ALLTID svara på svenska.\n"
                    "Du får endast använda information från det givna underlaget.\n"
                    "Du får inte tillföra ny fakta."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=700,
        temperature=0.3
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
    answer: dict
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

    return _call_llm(prompt)


# =========================
# RAG (FRITEXT)
# =========================

def generate_reasoning_from_prompt(prompt: str) -> str:
    """
    Genererar resonemang från färdig prompt (RAG).
    """
    return _call_llm(prompt)


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