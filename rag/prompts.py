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
Använd bara fakta som uttryckligen stöds av utdragen från PDF:erna eller hemsidan nedan.
Identifiera först vad användaren faktiskt frågar efter och håll svaret inom den ramen.
Behandla dokumentmetadata som icke-faktuell, och dra inte slutsatser från figurer, bilder eller tabeller om deras faktiska innehåll saknas i textutdraget.
Om underlaget inte räcker ska du tydligt säga att uppgiften inte kan verifieras från de tillgängliga PDF-/hemsidekällorna.
Om underlaget beskriver flera etapper, faser, steg eller aktiviteter ska du försöka täcka samtliga relevanta delar innan du går in på detaljer.
"""
