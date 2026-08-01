from __future__ import annotations

import json
import re
from typing import Any, Callable

DEFAULT_ANSWER_MODEL = "openai/gpt-oss-20b"
DEFAULT_REVIEW_MODEL = "openai/gpt-oss-20b"
DEFAULT_CORRECTION_MODEL = "openai/gpt-oss-20b"
MAX_ANSWER_EVIDENCE_CHUNKS = 8
MAX_CHUNK_EXCERPT_CHARS = 650
MAX_REVIEW_EVIDENCE_CHUNKS = 6
MAX_REVIEW_CHUNK_EXCERPT_CHARS = 450
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Det finns inte tillräckligt tydligt underlag i de hämtade källutdragen "
    "för att besvara frågan på ett säkert sätt."
)

LLMAnswerFn = Callable[[str, str | None], str]
LLMReviewFn = Callable[[str, str | None], str]

_ALLOWED_TOP_LEVEL_KEYS = {
    "original_question",
    "answer",
    "answer_scope",
    "evidence_used",
    "evidence_ids_used",
    "unsupported_or_uncertain",
    "source_coverage",
    "grounding_notes",
}
_ALLOWED_SCOPES = {"direct", "partial_due_to_thin_evidence", "insufficient_evidence"}
_SCOPE_ALIASES = {"sufficient": "direct"}
_ALLOWED_EVIDENCE_KEYS = {"chunk_id", "source", "pages", "claim_supported"}
_ALLOWED_REVIEW_TOP_LEVEL_KEYS = {"status", "reason", "revision", "evidence_ids_used"}
_ALLOWED_REVIEW_STATUSES = {"approved", "rejected", "revision"}
_STOPWORDS = {
    "och", "att", "det", "den", "de", "som", "för", "med", "till", "från", "eller",
    "inte", "kan", "ska", "bör", "också", "dessutom", "genom", "utifrån", "ett", "en", "samt",
    "har", "hur", "vad", "vilka", "vilken", "när", "efter", "innan", "frågan", "systemet",
    "underlaget", "källan", "källorna", "materialet", "beskriver", "visar", "anger",
}
_DISALLOWED_INTERNAL_PHRASES = (
    "agent 1",
    "agent1",
    "retrievalfråga",
    "retrievalfrågan",
    "retrieval query",
    "debugfält",
)


def build_evidence_answer_prompt(
    original_question: str,
    chunks: list[dict[str, Any]],
    rewrite_metadata: dict[str, Any] | None = None,
    *,
    model_target: str = DEFAULT_ANSWER_MODEL,
) -> str:
    compact_chunks = _compact_chunks(chunks)
    evidence_block = "\n".join(
        (
            f"chunk_id={chunk['chunk_id']} | källa={chunk['source']} | sidor={_format_pages(chunk.get('pages'))}\n"
            f"rubrik={chunk['title']}\n"
            f"utdrag={chunk['text']}"
        )
        for chunk in compact_chunks
    )
    metadata = _compact_rewrite_metadata(rewrite_metadata or {})

    return (
        f"Evidence comparator och answer builder för svensk RAG. Modellmål: {model_target}.\n"
        "Reasoning: high. Analysera utdragen noggrant internt, men visa aldrig ditt resonemang eller någon analys utanför JSON-svaret.\n"
        "Svara på originalfrågan, inte på retrievalfrågan eller någon omskriven sökvariant.\n"
        "Använd bara de hämtade evidensutdragen nedan. Lägg inte till generiska råd, best practice, roller, möten eller styrning om de inte står i evidensen.\n"
        "Använd accepterad rewrite-metadata bara för att förstå ordformer och samma begreppsfamilj mellan fråga och evidens, inte som egna fakta.\n"
        "Skriv naturlig och idiomatisk svensk prosa. Acceptera grammatiska böjningar och svenska sammansättningar när de stöds av evidensen. Börja med ett direkt svar på frågan och utveckla sedan svaret med de viktigaste förklaringarna, konsekvenserna och konkreta delarna från utdragen. Sikta på 2–5 sammanhängande stycken eller en tydlig punktlista när frågan ber om flera delar; undvik både telegramsvar och onödig upprepning. Undvik mekaniska starter som 'Kort sagt handlar det om följande'.\n"
        "Om frågan ber om en bedömning eller prioritering, sammanfatta försiktigt vad som framstår som viktigast i flera utdrag och markera det som en tolkning när källorna inte uttrycker rangordningen direkt.\n"
        "Returnera i första hand det bästa utkastet utifrån utdragen. Gör ingen egen kvalitetsbedömning; Agent 3 granskar svaret senare.\n"
        "Om evidence_used saknas eller blir ofullständigt kompletterar systemet metadata från de bästa hämtade utdragen.\n"
        "Returnera enbart strikt JSON, utan markdown eller prosa utanför objektet.\n"
        "JSON-fält: original_question, answer, answer_scope, evidence_used, unsupported_or_uncertain, grounding_notes.\n"
        "answer_scope måste vara exakt direct, partial_due_to_thin_evidence eller insufficient_evidence.\n"
        "Varje evidence_used-objekt ska bara ange chunk_id och claim_supported. Använd exakt ett listat chunk_id; systemet kompletterar source och pages.\n"
        "Det äldre top-level-fältet evidence_ids_used behöver inte returneras; om det ändå finns ignoreras det till förmån för evidence_used.\n"
        "Returnera inte source_coverage; systemet beräknar detta från validerad evidens och svaret.\n"
        "Tokenbudget: cirka 2 200–3 200 input tokens och högst 700 output tokens.\n\n"
        f"Fråga:\n{original_question}\n\n"
        f"Accepterad rewrite-metadata för retrieval, endast som stöd för termrelationer:\n{metadata}\n\n"
        f"Kompakt evidens:\n{evidence_block}"
    )


def generate_evidence_answer(
    original_question: str,
    chunks: list[dict[str, Any]],
    rewrite_metadata: dict[str, Any] | None,
    llm_answer: LLMAnswerFn | None,
    *,
    model: str = DEFAULT_ANSWER_MODEL,
) -> dict[str, Any]:
    if not chunks or not _has_evidence_text(chunks):
        return _fallback(original_question, "thin_evidence", model=model)
    if llm_answer is None:
        return _fallback(original_question, "agent2_no_llm_callback", model=model)

    prompt = build_evidence_answer_prompt(
        original_question,
        chunks,
        rewrite_metadata,
        model_target=model,
    )
    try:
        raw_response = llm_answer(prompt, model)
    except Exception:
        return _fallback(original_question, "agent2_exception", model=model)

    return parse_evidence_answer_response(
        original_question,
        chunks,
        raw_response,
        rewrite_metadata=rewrite_metadata,
        model=model,
    )


def build_evidence_correction_prompt(
    original_question: str,
    draft_answer: str,
    review_reason: str,
    chunks: list[dict[str, Any]],
    rewrite_metadata: dict[str, Any] | None = None,
    *,
    model_target: str = DEFAULT_CORRECTION_MODEL,
) -> str:
    base_prompt = build_evidence_answer_prompt(
        original_question,
        chunks,
        rewrite_metadata,
        model_target=model_target,
    )
    return (
        f"Korrigeringssteg för svensk RAG. Modellmål: {model_target}.\n"
        "Reasoning: high. Analysera Agent 3:s kritik och evidensen noggrant internt, men returnera endast JSON-kontraktet.\n"
        "Agent 3 underkände 20B-utkastet. Skriv ett förbättrat, mer utförligt och naturligt svenskt svar som rättar kritiken, "
        "men använd fortfarande endast den listade evidensen. Börja med ett direkt svar, förklara de viktigaste följderna och markera försiktiga slutsatser som tolkningar när källorna inte uttrycker dem direkt.\n"
        f"Underkänt utkast:\n{draft_answer}\n\n"
        f"Agent 3:s kritik:\n{review_reason or 'Svaret kunde inte godkännas enligt grounding-kontraktet.'}\n\n"
        f"{base_prompt}"
    )


def generate_corrected_evidence_answer(
    original_question: str,
    draft_answer: str,
    review_reason: str,
    chunks: list[dict[str, Any]],
    rewrite_metadata: dict[str, Any] | None,
    llm_answer: LLMAnswerFn | None,
    *,
    model: str = DEFAULT_CORRECTION_MODEL,
) -> dict[str, Any]:
    if not chunks or not _has_evidence_text(chunks):
        return _fallback(original_question, "correction_thin_evidence", model=model)
    if llm_answer is None:
        return _fallback(original_question, "correction_no_llm_callback", model=model)

    prompt = build_evidence_correction_prompt(
        original_question,
        draft_answer,
        review_reason,
        chunks,
        rewrite_metadata,
        model_target=model,
    )
    try:
        raw_response = llm_answer(prompt, model)
    except Exception:
        return _fallback(original_question, "correction_exception", model=model)

    return parse_evidence_answer_response(
        original_question,
        chunks,
        raw_response,
        rewrite_metadata=rewrite_metadata,
        model=model,
    )


def build_answer_review_prompt(
    original_question: str,
    draft_answer: str,
    evidence_snippets: list[dict[str, Any]],
    evidence_ids: list[str],
) -> str:
    allowed_ids = [str(item) for item in evidence_ids if str(item).strip()]
    allowed_id_set = set(allowed_ids)
    compact_chunks = [
        chunk
        for chunk in _compact_review_evidence(evidence_snippets)
        if chunk["chunk_id"] in allowed_id_set
    ]
    evidence_block = "\n".join(
        (
            f"evidence_id={chunk['chunk_id']} | källa={chunk['source']} | sidor={_format_pages(chunk.get('pages'))}\n"
            f"utdrag={chunk['text']}"
        )
        for chunk in compact_chunks
    )
    return (
        "Agent 3: review och grounding judge för svensk RAG. Modellmål: openai/gpt-oss-20b.\n"
        "Granska om draftsvar i huvudsak svarar på originalfrågan och har rimligt stöd i evidensen.\n"
        "Var tolerant mot formuleringar, ofullständighet och mindre avvikelser. Använd revision för en kort källstödd förbättring när det går.\n"
        "Använd rejected endast vid uppenbart fel ämne, påhittade centrala fakta, tydligt motsägelsefullt svar eller helt avsaknad av stöd.\n"
        "Returnera enbart strikt JSON utan markdown: {status, reason, revision, evidence_ids_used}.\n"
        "status måste vara approved, rejected eller revision. revision används bara om en kort källstödd korrigering kan ges från evidensen.\n"
        "evidence_ids_used får bara innehålla listade id. Tokenbudget: <=1 800 input tokens och <=200 output tokens.\n\n"
        f"Originalfråga:\n{original_question}\n\n"
        f"Draftsvar:\n{draft_answer}\n\n"
        f"Tillåtna evidence_ids:\n{json.dumps(allowed_ids, ensure_ascii=False)}\n\n"
        f"Kompakt evidens:\n{evidence_block}"
    )


def generate_answer_review(
    original_question: str,
    draft_answer: str,
    evidence_snippets: list[dict[str, Any]],
    evidence_ids: list[str],
    llm_review: LLMReviewFn | None,
    *,
    model: str = DEFAULT_REVIEW_MODEL,
) -> dict[str, Any]:
    if not draft_answer.strip() or not evidence_snippets:
        return _review_fallback(original_question, draft_answer, "thin_review_input", model=model)
    if llm_review is None:
        return _review_fallback(original_question, draft_answer, "agent3_no_llm_callback", model=model)

    prompt = build_answer_review_prompt(original_question, draft_answer, evidence_snippets, evidence_ids)
    try:
        raw_response = llm_review(prompt, model)
    except Exception:
        return _review_fallback(original_question, draft_answer, "agent3_exception", model=model)

    return parse_answer_review_response(
        original_question,
        draft_answer,
        evidence_snippets,
        evidence_ids,
        raw_response,
        model=model,
    )


def parse_answer_review_response(
    original_question: str,
    draft_answer: str,
    evidence_snippets: list[dict[str, Any]],
    evidence_ids: list[str],
    raw_response: str | None,
    *,
    model: str = DEFAULT_REVIEW_MODEL,
) -> dict[str, Any]:
    raw = (raw_response or "").strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        return _review_fallback(original_question, draft_answer, "agent3_invalid_json", model=model)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _review_fallback(original_question, draft_answer, "agent3_invalid_json", model=model)
    if not isinstance(payload, dict):
        return _review_fallback(original_question, draft_answer, "agent3_invalid_json", model=model)
    unexpected = sorted(set(payload) - _ALLOWED_REVIEW_TOP_LEVEL_KEYS)
    if unexpected:
        return _review_fallback(original_question, draft_answer, "agent3_unexpected_fields", model=model, extra={"unexpected_fields": unexpected})

    status = str(payload.get("status") or "").strip()
    if status not in _ALLOWED_REVIEW_STATUSES:
        return _review_fallback(original_question, draft_answer, "agent3_schema_error", model=model)

    allowed_ids = {str(item) for item in evidence_ids if str(item).strip()}
    used_ids = _valid_review_evidence_ids(payload.get("evidence_ids_used", []), allowed_ids)
    if status in {"approved", "revision"} and not used_ids:
        return _review_fallback(original_question, draft_answer, "agent3_missing_evidence", model=model)

    reason = str(payload.get("reason") or "").strip()[:500]
    revision = str(payload.get("revision") or "").strip()
    review_text = revision if status == "revision" and revision else draft_answer
    used_id_set = set(used_ids)
    cited_evidence = [
        chunk
        for index, chunk in enumerate(evidence_snippets[:MAX_REVIEW_EVIDENCE_CHUNKS], start=1)
        if evidence_chunk_id(chunk, index) in used_id_set
    ]
    if status in {"approved", "revision"} and not _review_answer_supported(original_question, review_text, cited_evidence):
        return _review_fallback(original_question, draft_answer, "agent3_grounding_failed", model=model)

    return {
        "status": status,
        "model": model,
        "original_question": original_question,
        "draft_answer": draft_answer,
        "reason": reason,
        "revision": revision if status == "revision" else "",
        "evidence_ids_used": used_ids,
        "debug": {
            "agent": "answer_review",
            "model": model,
            "fallback_reason": None,
            "evidence_chunk_count": len(evidence_snippets[:MAX_REVIEW_EVIDENCE_CHUNKS]),
            "token_budget": {"input_target": 1800, "output_target": 200},
        },
    }


def parse_evidence_answer_response(
    original_question: str,
    chunks: list[dict[str, Any]],
    raw_response: str | None,
    *,
    rewrite_metadata: dict[str, Any] | None = None,
    model: str = DEFAULT_ANSWER_MODEL,
) -> dict[str, Any]:
    raw = (raw_response or "").strip()
    recovered_from_raw = False
    if not raw.startswith("{") or not raw.endswith("}"):
        fenced = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if fenced:
            raw = fenced.group(0).strip()
            recovered_from_raw = True
        else:
            answer = re.sub(r"```(?:json)?|```", "", raw, flags=re.IGNORECASE).strip()
            if len(answer) >= 40:
                inferred = _infer_evidence_for_draft(chunks)
                if inferred:
                    return _raw_draft_response(original_question, answer, inferred, model=model)
            return _fallback(original_question, "agent2_invalid_json", model=model)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        inferred = _infer_evidence_for_draft(chunks)
        if len(raw) >= 40 and inferred:
            return _raw_draft_response(original_question, raw, inferred, model=model)
        return _fallback(original_question, "agent2_invalid_json", model=model)

    if not isinstance(payload, dict):
        return _fallback(original_question, "agent2_invalid_json", model=model)

    unexpected = sorted(set(payload) - _ALLOWED_TOP_LEVEL_KEYS)
    if unexpected:
        return _fallback(original_question, "agent2_unexpected_fields", model=model, extra={"unexpected_fields": unexpected})

    model_original_question = str(payload.get("original_question") or "").strip()

    answer_scope = str(payload.get("answer_scope") or "").strip()
    answer_scope = _SCOPE_ALIASES.get(answer_scope, answer_scope)
    if answer_scope not in _ALLOWED_SCOPES:
        return _fallback(original_question, "agent2_schema_error", model=model)
    answer = str(payload.get("answer") or "").strip()
    if len(answer) < 40:
        return _fallback(original_question, "agent2_empty_answer", model=model)
    if _has_internal_or_metadata_leakage(answer):
        return _fallback(original_question, "agent2_grounding_failed", model=model)

    chunk_lookup = {evidence_chunk_id(chunk, index): chunk for index, chunk in enumerate(chunks[:MAX_ANSWER_EVIDENCE_CHUNKS], start=1)}
    evidence_used, evidence_error, evidence_debug = _validate_evidence_used(
        payload.get("evidence_used"),
        chunk_lookup,
    )
    if not evidence_used:
        inferred = []
        for index, chunk in enumerate(chunks[:3], start=1):
            chunk_id = evidence_chunk_id(chunk, index)
            source = str(chunk.get("source") or "").strip()
            if chunk_id in chunk_lookup and source:
                inferred.append(
                    {
                        "chunk_id": chunk_id,
                        "source": source[:160],
                        "pages": chunk.get("pages") or [],
                        "claim_supported": "",
                    }
                )
        if not inferred:
            return _fallback(
                original_question,
                evidence_error or "agent2_evidence_missing",
                model=model,
                extra=evidence_debug,
            )
        evidence_used = inferred
        evidence_debug = {**evidence_debug, "evidence_inferred": True}

    unsupported = _valid_string_list(payload.get("unsupported_or_uncertain", []), max_items=6, max_length=180)

    coverage = {
        "uses_retrieved_chunks": True,
        "answers_original_question": True,
        "ignores_metadata_as_facts": True,
    }

    return {
        "status": "ok",
        "model": model,
        "original_question": original_question,
        "answer": answer,
        "answer_scope": answer_scope,
        "evidence_used": evidence_used,
        "evidence_ids_used": [item["chunk_id"] for item in evidence_used],
        "unsupported_or_uncertain": unsupported,
        "source_coverage": coverage,
        "grounding_notes": str(payload.get("grounding_notes") or "").strip()[:600],
        "model_original_question": model_original_question,
        "debug": {
            "agent": "evidence_answer",
            "model": model,
            "fallback_reason": None,
            "recovery_reason": "json_fence_extracted" if recovered_from_raw else None,
            "evidence_chunk_count": len(chunks[:MAX_ANSWER_EVIDENCE_CHUNKS]),
            "token_budget": {"input_target": "2200-3200", "output_target": 500},
        },
    }


def _infer_evidence_for_draft(chunks: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    inferred = []
    for index, chunk in enumerate(chunks[:limit], start=1):
        chunk_id = evidence_chunk_id(chunk, index)
        source = str(chunk.get("source") or "").strip()
        if source:
            inferred.append(
                {
                    "chunk_id": chunk_id,
                    "source": source[:160],
                    "pages": chunk.get("pages") or [],
                    "claim_supported": "",
                }
            )
    return inferred


def _raw_draft_response(
    original_question: str,
    answer: str,
    evidence_used: list[dict[str, Any]],
    *,
    model: str,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "model": model,
        "original_question": original_question,
        "answer": answer,
        "answer_scope": "partial_due_to_thin_evidence",
        "evidence_used": evidence_used,
        "evidence_ids_used": [item["chunk_id"] for item in evidence_used],
        "unsupported_or_uncertain": [],
        "source_coverage": {"uses_retrieved_chunks": True, "answers_original_question": True, "ignores_metadata_as_facts": True},
        "grounding_notes": "Agent 2-utkastet återhämtades från modellens råtext; Agent 3 ansvarar för granskning.",
        "debug": {"agent": "evidence_answer", "model": model, "fallback_reason": None, "recovery_reason": "raw_text_recovered"},
    }


def _fallback(
    original_question: str,
    reason: str,
    *,
    model: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    debug = {
        "agent": "evidence_answer",
        "model": model,
        "fallback_reason": reason,
        "evidence_chunk_count": 0,
        "token_budget": {"input_target": "2200-3200", "output_target": 500},
    }
    if extra:
        debug.update(extra)
    return {
        "status": "fallback",
        "model": model,
        "original_question": original_question,
        "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
        "answer_scope": "insufficient_evidence",
        "evidence_used": [],
        "evidence_ids_used": [],
        "unsupported_or_uncertain": [],
        "source_coverage": {
            "uses_retrieved_chunks": False,
            "answers_original_question": False,
            "ignores_metadata_as_facts": True,
        },
        "grounding_notes": "Fallback används eftersom Agent 2-resultatet inte kunde valideras.",
        "debug": debug,
    }


def _review_fallback(
    original_question: str,
    draft_answer: str,
    reason: str,
    *,
    model: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    debug = {
        "agent": "answer_review",
        "model": model,
        "fallback_reason": reason,
        "evidence_chunk_count": 0,
        "token_budget": {"input_target": 1800, "output_target": 200},
    }
    if extra:
        debug.update(extra)
    return {
        "status": "rejected",
        "model": model,
        "original_question": original_question,
        "draft_answer": draft_answer,
        "reason": "Review kunde inte godkänna svaret eftersom det inte kunde valideras mot originalfrågan och evidensen.",
        "revision": INSUFFICIENT_EVIDENCE_MESSAGE,
        "evidence_ids_used": [],
        "debug": debug,
    }


def _compact_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for index, chunk in enumerate(chunks[:MAX_ANSWER_EVIDENCE_CHUNKS], start=1):
        text = " ".join(str(chunk.get("text") or "").split())[:MAX_CHUNK_EXCERPT_CHARS]
        compact.append(
            {
                "chunk_id": evidence_chunk_id(chunk, index),
                "source": str(chunk.get("source") or ""),
                "title": str(chunk.get("title") or ""),
                "pages": chunk.get("pages") or [],
                "text": text,
            }
        )
    return compact


def _compact_review_evidence(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for index, chunk in enumerate(chunks[:MAX_REVIEW_EVIDENCE_CHUNKS], start=1):
        text = " ".join(str(chunk.get("text") or "").split())[:MAX_REVIEW_CHUNK_EXCERPT_CHARS]
        compact.append(
            {
                "chunk_id": evidence_chunk_id(chunk, index),
                "source": str(chunk.get("source") or ""),
                "pages": chunk.get("pages") or [],
                "text": text,
            }
        )
    return compact


def _valid_review_evidence_ids(value: Any, allowed_ids: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    seen = set()
    for item in value:
        evidence_id = str(item or "").strip()
        if not evidence_id or evidence_id not in allowed_ids or evidence_id in seen:
            continue
        out.append(evidence_id)
        seen.add(evidence_id)
        if len(out) >= MAX_REVIEW_EVIDENCE_CHUNKS:
            break
    return out


def _review_answer_supported(original_question: str, answer: str, evidence_snippets: list[dict[str, Any]]) -> bool:
    if _has_internal_or_metadata_leakage(answer):
        return False
    answer_tokens = _content_tokens(answer)
    if len(answer_tokens) < 4:
        return False
    question_tokens = _content_tokens(original_question)
    evidence_tokens: set[str] = set()
    for chunk in evidence_snippets[:MAX_REVIEW_EVIDENCE_CHUNKS]:
        evidence_tokens.update(_content_tokens(chunk.get("text", "")))
    if not evidence_tokens:
        return False
    if not (answer_tokens & question_tokens):
        return False
    supported_tokens = answer_tokens & (evidence_tokens | question_tokens)
    return len(supported_tokens) / max(len(answer_tokens), 1) >= 0.30


def _compact_rewrite_metadata(metadata: dict[str, Any]) -> str:
    allowed = {
        "status": metadata.get("status"),
        "original_question": metadata.get("original_question"),
        "semantic_terms": metadata.get("semantic_terms", [])[:12] if isinstance(metadata.get("semantic_terms", []), list) else [],
        "negative_constraints": metadata.get("negative_constraints", [])[:6] if isinstance(metadata.get("negative_constraints", []), list) else [],
    }
    return json.dumps(allowed, ensure_ascii=False)


def evidence_chunk_id(chunk: dict[str, Any], index: int) -> str:
    value = chunk.get("id") or chunk.get("chunk_id")
    if value:
        return str(value)
    source = str(chunk.get("source") or "chunk")
    pages = chunk.get("pages") or []
    page_part = "-".join(str(page) for page in pages[:2]) if isinstance(pages, list) else str(pages)
    safe_source = re.sub(r"[^A-Za-z0-9ÅÄÖåäö_-]+", "_", source).strip("_")[:60]
    return f"{safe_source or 'chunk'}:{page_part or index}"


def _format_pages(pages: Any) -> str:
    if isinstance(pages, list):
        return ",".join(str(page) for page in pages)
    return str(pages or "")


def _validate_evidence_used(
    value: Any,
    chunk_lookup: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None, dict[str, Any]]:
    allowed_ids = sorted(chunk_lookup)
    if not isinstance(value, list) or not value:
        return [], "agent2_evidence_missing", {"allowed_evidence_ids": allowed_ids}
    out = []
    seen = set()
    for item in value:
        if not isinstance(item, dict):
            return [], "agent2_evidence_invalid_shape", {"allowed_evidence_ids": allowed_ids}
        unexpected = sorted(set(item) - _ALLOWED_EVIDENCE_KEYS)
        if unexpected:
            return [], "agent2_evidence_invalid_shape", {
                "allowed_evidence_ids": allowed_ids,
                "unexpected_evidence_fields": unexpected,
            }
        chunk_id = str(item.get("chunk_id") or "").strip()
        if not chunk_id:
            return [], "agent2_evidence_missing_id", {"allowed_evidence_ids": allowed_ids}
        if chunk_id not in chunk_lookup:
            return [], "agent2_evidence_unknown_id", {
                "allowed_evidence_ids": allowed_ids,
                "unknown_evidence_ids": [chunk_id[:160]],
            }
        chunk = chunk_lookup[chunk_id]
        claim = str(item.get("claim_supported") or "").strip()
        source = str(chunk.get("source") or "").strip()
        pages = chunk.get("pages") or []
        if not source:
            return [], "agent2_evidence_source_missing", {"evidence_id": chunk_id}
        if chunk_id in seen:
            continue
        out.append(
            {
                "chunk_id": chunk_id,
                "source": source[:160],
                "pages": pages,
                "claim_supported": claim[:220],
            }
        )
        seen.add(chunk_id)
        if len(out) >= 8:
            break
    return out, None, {"allowed_evidence_ids": allowed_ids}


def _has_evidence_text(chunks: list[dict[str, Any]]) -> bool:
    return any(_content_tokens(chunk.get("text", "")) for chunk in chunks[:MAX_ANSWER_EVIDENCE_CHUNKS])


def _valid_string_list(value: Any, *, max_items: int, max_length: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip()[:max_length])
            if len(out) >= max_items:
                break
    return out


def _has_internal_or_metadata_leakage(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _DISALLOWED_INTERNAL_PHRASES)


def _answer_grounding_failure(
    original_question: str,
    answer: str,
    chunks: list[dict[str, Any]],
    evidence_used: list[dict[str, Any]],
    rewrite_metadata: dict[str, Any],
) -> str | None:
    support_tokens: set[str] = set()
    support_tokens.update(_content_tokens(original_question))
    cited_ids = {item["chunk_id"] for item in evidence_used}
    for index, chunk in enumerate(chunks[:MAX_ANSWER_EVIDENCE_CHUNKS], start=1):
        if evidence_chunk_id(chunk, index) in cited_ids:
            support_tokens.update(_content_tokens(chunk.get("text", "")))
    for item in evidence_used:
        support_tokens.update(_content_tokens(item.get("claim_supported", "")))
    for term in rewrite_metadata.get("semantic_terms", []) if isinstance(rewrite_metadata, dict) else []:
        if not isinstance(term, dict):
            continue
        support_tokens.update(_content_tokens(term.get("surface", "")))
        support_tokens.update(_content_tokens(term.get("normalized_family", "")))

    if not support_tokens:
        return "no_cited_support_tokens"

    answer_tokens = _content_tokens(answer)
    if len(answer_tokens) < 4:
        return "answer_too_short"
    total_ratio = len(answer_tokens & support_tokens) / max(len(answer_tokens), 1)
    if total_ratio < 0.30:
        return f"low_total_overlap:{total_ratio:.2f}"

    for index, claim_unit in enumerate(_split_claim_units(answer), start=1):
        claim_tokens = _content_tokens(claim_unit)
        if not claim_tokens:
            continue
        overlap = claim_tokens & support_tokens
        claim_ratio = len(overlap) / max(len(claim_tokens), 1)
        if len(overlap) >= 2:
            if claim_ratio >= 0.30:
                continue
        if claim_ratio < 0.35:
            return f"low_claim_overlap:{index}:{claim_ratio:.2f}"
    return None


def _split_claim_units(text: str) -> list[str]:
    units = []
    for sentence in _split_sentences(text):
        units.extend(
            part.strip()
            for part in re.split(r"\s*(?:;|\bdessutom\b|\bsamt\b|\bmen\b)\s*", sentence, flags=re.IGNORECASE)
            if part.strip()
        )
    return units


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text or "") if part.strip()]


def _content_tokens(text: Any) -> set[str]:
    folded = _fold(str(text or ""))
    tokens = set()
    for raw in re.findall(r"[a-z0-9åäö]+", folded):
        token = _stem_token(raw)
        if len(token) < 4 or token in _STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _stem_token(token: str) -> str:
    for suffix in (
        "ningarna", "ningens", "ningen", "ningar", "andes", "ande", "ades", "ade", "ats",
        "ning", "ing", "arna", "erna", "ens", "het", "are", "arna", "erna", "orna",
        "en", "et", "ar", "er", "at", "as", "a", "s",
    ):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]
    return token


def _fold(text: str) -> str:
    return (
        (text or "")
        .lower()
        .replace("å", "a")
        .replace("ä", "a")
        .replace("ö", "o")
    )
