from __future__ import annotations

import json
import re
from typing import Any, Callable

DEFAULT_ANSWER_MODEL = "openai/gpt-oss-120b"
MAX_ANSWER_EVIDENCE_CHUNKS = 8
MAX_CHUNK_EXCERPT_CHARS = 650
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Det finns inte tillräckligt tydligt underlag i de hämtade källutdragen "
    "för att besvara frågan på ett säkert sätt."
)

LLMAnswerFn = Callable[[str, str | None], str]

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
_ALLOWED_EVIDENCE_KEYS = {"chunk_id", "source", "pages", "claim_supported"}
_ALLOWED_COVERAGE_KEYS = {
    "uses_retrieved_chunks",
    "answers_original_question",
    "ignores_metadata_as_facts",
}
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
        "Evidence comparator och answer builder för svensk RAG. Modellmål: openai/gpt-oss-120b.\n"
        "Svara på originalfrågan, inte på retrievalfrågan eller någon omskriven sökvariant.\n"
        "Använd bara de hämtade evidensutdragen nedan. Lägg inte till generiska råd, best practice, roller, möten eller styrning om de inte står i evidensen.\n"
        "Använd accepterad rewrite-metadata bara för att förstå ordformer och samma begreppsfamilj mellan fråga och evidens, inte som egna fakta.\n"
        "Skriv naturlig svensk prosa. Acceptera grammatiska böjningar och svenska sammansättningar när de stöds av evidensen.\n"
        "Varje central svarspunkt måste stödjas av minst ett evidence_used-objekt med chunk_id och rapporteras i evidence_ids_used.\n"
        "Om evidensen inte räcker: answer_scope=insufficient_evidence och ge ett kort ärligt icke-svar.\n"
        "Returnera enbart strikt JSON, utan markdown eller prosa utanför objektet.\n"
        "JSON-fält: original_question, answer, answer_scope, evidence_used, evidence_ids_used, unsupported_or_uncertain, source_coverage, grounding_notes.\n"
        "source_coverage måste ange uses_retrieved_chunks, answers_original_question och ignores_metadata_as_facts.\n"
        "Tokenbudget: cirka 2 200–3 200 input tokens och högst 500 output tokens.\n\n"
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

    prompt = build_evidence_answer_prompt(original_question, chunks, rewrite_metadata)
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


def parse_evidence_answer_response(
    original_question: str,
    chunks: list[dict[str, Any]],
    raw_response: str | None,
    *,
    rewrite_metadata: dict[str, Any] | None = None,
    model: str = DEFAULT_ANSWER_MODEL,
) -> dict[str, Any]:
    raw = (raw_response or "").strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        return _fallback(original_question, "agent2_invalid_json", model=model)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback(original_question, "agent2_invalid_json", model=model)

    if not isinstance(payload, dict):
        return _fallback(original_question, "agent2_invalid_json", model=model)

    unexpected = sorted(set(payload) - _ALLOWED_TOP_LEVEL_KEYS)
    if unexpected:
        return _fallback(original_question, "agent2_unexpected_fields", model=model, extra={"unexpected_fields": unexpected})

    if payload.get("original_question") != original_question:
        return _fallback(original_question, "agent2_original_question_mismatch", model=model)

    answer_scope = str(payload.get("answer_scope") or "").strip()
    if answer_scope not in _ALLOWED_SCOPES:
        return _fallback(original_question, "agent2_schema_error", model=model)
    if answer_scope == "insufficient_evidence":
        return _fallback(original_question, "thin_evidence", model=model)

    answer = str(payload.get("answer") or "").strip()
    if len(answer) < 40:
        return _fallback(original_question, "agent2_empty_answer", model=model)
    if _has_internal_or_metadata_leakage(answer):
        return _fallback(original_question, "agent2_grounding_failed", model=model)

    chunk_lookup = {_chunk_id(chunk, index): chunk for index, chunk in enumerate(chunks[:MAX_ANSWER_EVIDENCE_CHUNKS], start=1)}
    evidence_used = _valid_evidence_used(payload.get("evidence_used"), chunk_lookup)
    if not evidence_used:
        return _fallback(original_question, "agent2_missing_evidence", model=model)
    cited_chunk_ids = {item["chunk_id"] for item in evidence_used}
    supported_chunk_ids = _answer_supported_chunk_ids(answer, chunks)
    if not supported_chunk_ids.issubset(cited_chunk_ids):
        return _fallback(original_question, "agent2_missing_evidence", model=model)

    coverage = payload.get("source_coverage")
    if not _valid_source_coverage(coverage):
        return _fallback(original_question, "agent2_grounding_failed", model=model)

    unsupported = _valid_string_list(payload.get("unsupported_or_uncertain", []), max_items=6, max_length=180)
    if unsupported and answer_scope == "direct":
        return _fallback(original_question, "agent2_grounding_failed", model=model)

    if not _answer_is_grounded(original_question, answer, chunks, evidence_used, rewrite_metadata or {}):
        return _fallback(original_question, "agent2_grounding_failed", model=model)

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
        "debug": {
            "agent": "evidence_answer",
            "model": model,
            "fallback_reason": None,
            "evidence_chunk_count": len(chunks[:MAX_ANSWER_EVIDENCE_CHUNKS]),
            "token_budget": {"input_target": "2200-3200", "output_target": 500},
        },
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


def _compact_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for index, chunk in enumerate(chunks[:MAX_ANSWER_EVIDENCE_CHUNKS], start=1):
        text = " ".join(str(chunk.get("text") or "").split())[:MAX_CHUNK_EXCERPT_CHARS]
        compact.append(
            {
                "chunk_id": _chunk_id(chunk, index),
                "source": str(chunk.get("source") or ""),
                "title": str(chunk.get("title") or ""),
                "pages": chunk.get("pages") or [],
                "text": text,
            }
        )
    return compact


def _compact_rewrite_metadata(metadata: dict[str, Any]) -> str:
    allowed = {
        "status": metadata.get("status"),
        "original_question": metadata.get("original_question"),
        "semantic_terms": metadata.get("semantic_terms", [])[:12] if isinstance(metadata.get("semantic_terms", []), list) else [],
        "negative_constraints": metadata.get("negative_constraints", [])[:6] if isinstance(metadata.get("negative_constraints", []), list) else [],
    }
    return json.dumps(allowed, ensure_ascii=False)


def _chunk_id(chunk: dict[str, Any], index: int) -> str:
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


def _valid_evidence_used(value: Any, chunk_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out = []
    seen = set()
    for item in value:
        if not isinstance(item, dict) or set(item) - _ALLOWED_EVIDENCE_KEYS:
            return []
        chunk_id = str(item.get("chunk_id") or "").strip()
        claim = str(item.get("claim_supported") or "").strip()
        source = str(item.get("source") or "").strip()
        if not chunk_id or chunk_id not in chunk_lookup or not claim or not source:
            return []
        if chunk_id in seen:
            continue
        out.append(
            {
                "chunk_id": chunk_id,
                "source": source[:160],
                "pages": item.get("pages", []),
                "claim_supported": claim[:220],
            }
        )
        seen.add(chunk_id)
        if len(out) >= 8:
            break
    return out


def _valid_source_coverage(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) - _ALLOWED_COVERAGE_KEYS:
        return False
    return (
        value.get("uses_retrieved_chunks") is True
        and value.get("answers_original_question") is True
        and value.get("ignores_metadata_as_facts") is True
    )


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


def _answer_is_grounded(
    original_question: str,
    answer: str,
    chunks: list[dict[str, Any]],
    evidence_used: list[dict[str, Any]],
    rewrite_metadata: dict[str, Any],
) -> bool:
    support_tokens: set[str] = set()
    support_tokens.update(_content_tokens(original_question))
    for chunk in chunks[:MAX_ANSWER_EVIDENCE_CHUNKS]:
        support_tokens.update(_content_tokens(chunk.get("text", "")))
    for item in evidence_used:
        support_tokens.update(_content_tokens(item.get("claim_supported", "")))
    for term in rewrite_metadata.get("semantic_terms", []) if isinstance(rewrite_metadata, dict) else []:
        if not isinstance(term, dict):
            continue
        support_tokens.update(_content_tokens(term.get("surface", "")))
        support_tokens.update(_content_tokens(term.get("normalized_family", "")))

    if not support_tokens:
        return False

    answer_tokens = _content_tokens(answer)
    if len(answer_tokens) < 4:
        return False
    if len(answer_tokens & support_tokens) / max(len(answer_tokens), 1) < 0.30:
        return False

    for sentence in _split_sentences(answer):
        sentence_tokens = _content_tokens(sentence)
        if not sentence_tokens:
            continue
        overlap = sentence_tokens & support_tokens
        if len(sentence_tokens - support_tokens) >= 2:
            return False
        if len(overlap) >= 2:
            continue
        if len(overlap) / max(len(sentence_tokens), 1) >= 0.35:
            continue
        return False
    return True


def _answer_supported_chunk_ids(answer: str, chunks: list[dict[str, Any]]) -> set[str]:
    answer_tokens = _content_tokens(answer)
    supported = set()
    for index, chunk in enumerate(chunks[:MAX_ANSWER_EVIDENCE_CHUNKS], start=1):
        chunk_tokens = _content_tokens(chunk.get("text", ""))
        if len(answer_tokens & chunk_tokens) >= 3:
            supported.add(_chunk_id(chunk, index))
    return supported


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
