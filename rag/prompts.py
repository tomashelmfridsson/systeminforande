from llm.prompts import base_llm_instructions


def rag_prompt(query: str, chunks: list) -> str:
    context = "\n\n".join(
        f"UTDRAG:\n{c['text']}" for c in chunks
    )

    return f"""
{base_llm_instructions()}

Typ av fråga:
Fri fråga baserad på källutdrag

Fråga:
{query}

Utdrag ur dokumentation:
{context}

Uppgift:
Besvara frågan så långt underlaget räcker och formulera ett försiktigt resonemang.
Om underlaget beskriver flera etapper, faser, steg eller aktiviteter ska du försöka täcka samtliga relevanta delar innan du går in på detaljer.
"""
