from __future__ import annotations

import os
import re
from typing import Any, Callable

from rag.extractive import build_extractive_reasoning
from rag.grounding import INSUFFICIENT_EVIDENCE_ANSWER, grounded_answer_or_fallback
from rag.prompts import rag_prompt

LLMRewriteFn = Callable[[str, str | None], str]
SYNTHESIS_FEATURE_FLAG_ENV = "SYSTEMINFORANDE_ENABLE_LLM_SYNTHESIS"
SYNTHESIS_MODEL_ENV = "SYSTEMINFORANDE_LLM_SYNTHESIS_MODEL"
DEFAULT_SYNTHESIS_MODEL = "openai/gpt-oss-120b"
SUPPORTED_EXPERIMENT_MODELS = (
    "openai/gpt-oss-120b",
    "zai-org/GLM-5.2",
)

_STOPWORDS = {
    "och", "att", "det", "som", "för", "med", "den", "detta", "eller", "inte", "kan",
    "ska", "bör", "också", "bara", "från", "till", "om", "hur", "vad", "vilka", "vilken",
    "finns", "utifrån", "materialet", "visar", "beskriver", "frågan", "underlaget", "samt",
    "genom", "det", "de", "ett", "en", "har", "sin", "sitt", "sina", "denna", "dessa",
}


def _parse_feature_flag(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_synthesis_settings(
    *,
    enable_synthesis: bool | None = None,
    llm_model: str | None = None,
    default_model: str | None = None,
) -> dict[str, Any]:
    env_enabled = _parse_feature_flag(os.getenv(SYNTHESIS_FEATURE_FLAG_ENV, "true"))
    resolved_enabled = env_enabled if enable_synthesis is None else bool(enable_synthesis)

    requested_model = (llm_model or "").strip()
    env_model = os.getenv(SYNTHESIS_MODEL_ENV, "").strip()
    fallback_model = (default_model or DEFAULT_SYNTHESIS_MODEL).strip() or DEFAULT_SYNTHESIS_MODEL
    resolved_model = requested_model or env_model or fallback_model

    return {
        "enabled": resolved_enabled,
        "model": resolved_model,
        "requested_model": requested_model or None,
        "env_default_model": env_model or None,
        "enabled_source": "override" if enable_synthesis is not None else "environment",
    }


def build_synthesis_prompt(query: str, chunks: list[dict], fallback_answer: str = "") -> str:
    source_context = "\n\n".join(
        (
            f"KÄLLA: {chunk.get('source', '')}\n"
            f"RUBRIK: {chunk.get('title', '')}\n"
            f"UTDRAG: {chunk.get('text', '')}"
        )
        for chunk in chunks
    )

    return (
        f"{rag_prompt(query, chunks)}\n\n"
        "Svara självständigt utifrån källunderlaget nedan. Använd inte ett förbyggt bassvar som struktur.\n"
        "Identifiera vad användaren faktiskt frågar efter innan du skriver, och välj bara de delar av underlaget som svarar på just den frågan.\n"
        "Resonemanget ska vara fritt formulerat, naturligt och utvecklat, men varje sakpåstående måste kunna stödjas av källunderlaget.\n"
        "Behandla PDF-metadata som icke-faktuell: författarnamn, dokumenttitlar, dokumentmallar, versionsrader, copyright, sidhuvuden, sidfötter och sid-/etikettrader får inte bli svarsinnehåll.\n"
        "Håll dig till frågans fokus, till exempel skillnaden mellan planering, organisering, tekniska krav, ansvar och arbetsmodell.\n"
        "Text som 'Nedanstående figur', 'Nedanstående bild' eller 'Tabellen nedan' räcker inte som sakstöd om den faktiska informationen bara finns i figuren, bilden eller tabellen och inte finns i textutdraget.\n"
        "Om underlaget är smalt får du resonera om vad som faktiskt går att belägga och vad som inte går att belägga.\n"
        "Börja med ett direkt svar. Utveckla därefter vad svaret betyder, varför de belagda punkterna spelar roll och hur de hänger ihop.\n"
        "För frågor om det finns en arbetsmodell: stanna inte vid 'ja' om underlaget stödjer mer, utan beskriv modellens delar och hur de hänger ihop.\n"
        "Använd normalt 4 till 8 meningar på svenska när frågan är avgränsad. För breda fria frågor, till exempel om hinder, faser eller flera sammanhängande områden, använd normalt 6 till 10 meningar när källunderlaget räcker.\n"
        "Förklara vad punkterna innebär, varför de spelar roll för införandet och hur de hänger ihop med varandra, utan att lägga till generiska råd som inte finns i källorna.\n"
        "Skriv i flytande svensk prosa och undvik punktlista om frågan inte ber om det.\n"
        "Skriv inga dokument- eller sidreferenser inne i resonemanget; källor redovisas separat utanför LLM-svaret.\n"
        "Undvik mallfraser som 'Frågan verkar beröra', 'Materialet visar att', 'Materialet anger att' och 'de hämtade utdragen'. Undvik också motfrågor och mekaniska svarsmallar.\n\n"
        f"Fråga:\n{query}\n\n"
        f"Fallback-svar om LLM-svaret inte blir källbundet:\n{fallback_answer}\n\n"
        f"Källunderlag:\n{source_context}"
    )


def build_final_grounded_answer(
    query: str,
    chunks: list[dict],
    *,
    enable_synthesis: bool,
    llm_model: str | None = None,
    llm_rewrite: LLMRewriteFn | None = None,
) -> dict[str, object]:
    raw_extractive_answer = build_extractive_reasoning(query, chunks)
    extractive_answer = _normalize_extractive_answer(raw_extractive_answer)
    result: dict[str, object] = {
        "extractive_answer": extractive_answer,
        "final_answer": extractive_answer,
        "synthesis_enabled": enable_synthesis,
        "synthesis_used": False,
        "llm_model": llm_model,
        "llm_status": "disabled",
        "synthesis_prompt": "",
    }

    if not enable_synthesis:
        return result

    if extractive_answer == INSUFFICIENT_EVIDENCE_ANSWER:
        result["llm_status"] = "skipped_due_to_insufficient_evidence"
        return result

    if llm_rewrite is None:
        result["llm_status"] = "skipped_no_llm_callback"
        return result

    synthesis_prompt = build_synthesis_prompt(query, chunks, extractive_answer)
    result["synthesis_prompt"] = synthesis_prompt

    rewritten_answer = (llm_rewrite(synthesis_prompt, llm_model) or "").strip()
    if not rewritten_answer:
        result["llm_status"] = "fallback_to_extractive_due_to_empty_rewrite"
        return result

    if not _passes_grounding_check(rewritten_answer, chunks, extractive_answer, query):
        result["llm_status"] = "fallback_to_extractive_due_to_grounding_check"
        return result

    result["final_answer"] = rewritten_answer
    result["synthesis_used"] = True
    result["llm_status"] = "rewrite_applied"
    return result


def _normalize_extractive_answer(answer: str | None) -> str:
    text = (answer or "").strip()
    if not text:
        return grounded_answer_or_fallback("")
    if text.startswith("Det finns inte tillräckligt tydligt underlag"):
        return grounded_answer_or_fallback("")
    return text


def _passes_grounding_check(candidate: str, chunks: list[dict], extractive_answer: str, query: str = "") -> bool:
    text = _strip_metadata(candidate)
    if len(text) < 40:
        return False
    if _has_disallowed_template_phrase(text):
        return False
    if _has_disallowed_metadata_phrase(text):
        return False

    support_tokens = _content_tokens(extractive_answer)
    support_tokens.update(_content_tokens(query))
    for chunk in chunks:
        support_tokens.update(_content_tokens(chunk.get("source", "")))
        support_tokens.update(_content_tokens(chunk.get("title", "")))
        support_tokens.update(_content_tokens(chunk.get("text", "")))

    candidate_tokens = _content_tokens(text)
    if len(candidate_tokens) < 6:
        return False

    overlap = candidate_tokens & support_tokens
    if len(overlap) / max(len(candidate_tokens), 1) < 0.28:
        return False

    for sentence in _split_sentences(text):
        sentence_tokens = _content_tokens(sentence)
        if not sentence_tokens:
            continue
        supported = sentence_tokens & support_tokens
        if len(supported) < min(2, len(sentence_tokens)) and len(supported) / max(len(sentence_tokens), 1) < 0.30:
            return False

    return True


def _has_disallowed_template_phrase(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        phrase in lowered
        for phrase in (
            "materialet visar att",
            "materialet anger att",
            "materialet beskriver att",
            "de hämtade utdragen",
            "frågan verkar beröra",
        )
    )


def _has_disallowed_metadata_phrase(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        phrase in lowered
        for phrase in (
            "hans johansson",
            "© citrus",
            "citrus i stockholm",
            "citrus projektstyrning",
            "version 1",
        )
    )


def _strip_metadata(answer: str) -> str:
    text = (answer or "").strip()
    for marker in ("\n\n---\n\n### Källor", "\n\n---\n\n### Debug"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _content_tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-zÅÄÖåäö0-9]+", (text or "").lower()):
        if len(token) < 4 or token in _STOPWORDS:
            continue
        tokens.add(token)
        ascii_token = (
            token.replace("å", "a")
            .replace("ä", "a")
            .replace("ö", "o")
        )
        tokens.add(ascii_token)
    return tokens
