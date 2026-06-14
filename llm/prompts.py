def reasoning_prompt(
    *,
    title: str,
    main_question: str,
    question: str,
    answer: str
) -> str:
    return f"""
Du ska ge ett tydligt, strukturerat resonemang.

Titel:
{title}

Huvudfråga:
{main_question}

Underfråga:
{question}

Faktasvar:
{answer}

Förklara varför detta svar är korrekt.
- Använd endast informationen ovan
- Lägg inte till ny fakta
- Skriv på svenska
"""