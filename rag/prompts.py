def rag_prompt(query: str, chunks: list) -> str:
    context = "\n\n".join(
        f"UTDRAG:\n{c['text']}" for c in chunks
    )

    return f"""

Instruktion:
Svara på svenska.
Du är en sakkunnig assistent.
Använd ENDAST informationen i utdragen nedan.
Tillför ingen ny fakta.
Gör inga antaganden.

Uppgift:
Besvara frågan med en sammanhängande och förklarande text.
Undvik listor, metadata och rubriker.
Förklara vad begreppet innebär utifrån materialet.

Om materialet inte tydligt besvarar frågan:
Säg att det inte finns en entydig definition i materialet.

Fråga:
{query}

Utdrag ur dokumentation:
{context}
"""