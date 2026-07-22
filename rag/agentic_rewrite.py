from __future__ import annotations

import json
import re
from typing import Any, Callable

DEFAULT_REWRITE_MODEL = "openai/gpt-oss-20b"
MAX_RETRIEVAL_QUERIES = 5
MIN_REWRITE_CONFIDENCE = 0.55

_ALLOWED_TOP_LEVEL_KEYS = {
    "original_question",
    "retrieval_queries",
    "must_keep_focus",
    "semantic_terms",
    "negative_constraints",
    "confidence",
}
_ALLOWED_QUERY_KEYS = {"query", "purpose"}
_ALLOWED_TERM_KEYS = {"surface", "normalized_family", "kind"}
_ALLOWED_PURPOSES = {
    "literal",
    "swedish_inflection",
    "synonym",
    "compound",
    "broader_context",
}
_ALLOWED_TERM_KINDS = {
    "lemma",
    "inflection",
    "synonym",
    "compound",
    "spelling_variant",
}
_STOPWORDS = {
    "vad",
    "är",
    "hur",
    "ska",
    "kan",
    "det",
    "den",
    "och",
    "att",
    "som",
    "för",
    "från",
    "med",
    "till",
    "om",
    "i",
    "på",
    "av",
    "en",
    "ett",
    "vid",
    "finns",
    "vilka",
    "vilken",
    "efter",
    "system",
    "systemet",
    "fråga",
    "frågan",
}
_ANSWER_LIKE_PATTERNS = (
    re.compile(r"^(ja|nej)\b", re.IGNORECASE),
    re.compile(r"\b(det finns|det saknas|svaret är|innebär att)\b", re.IGNORECASE),
)

LLMRewriteFn = Callable[[str, str | None], str]


def build_retrieval_rewrite_prompt(question: str) -> str:
    return (
        "Agent 1 för svensk RAG-retrieval. Modellmål: openai/gpt-oss-20b.\n"
        "Svara inte på frågan. Skapa bara meaning-preserving sökvarianter för retrieval.\n"
        "Returnera enbart strikt JSON, utan markdown eller prosa utanför objektet.\n"
        "Budget: högst 600 input tokens och 200 output tokens i normalfall.\n"
        "Regler: max 5 retrieval_queries; behåll frågans scope; tillåt synonymer, böjningar, sammansättningar, singular/plural och sannolikt PDF-ordval; "
        "ändra inte fråga om t.ex. överlämning till driftsättning eller annat närliggande men annat ämne.\n"
        "JSON-schema i kortform: {original_question: exakt frågan, retrieval_queries: [{query, purpose}], semantic_terms: [{surface, normalized_family, kind}], negative_constraints: [str], confidence: 0..1}.\n"
        "purpose måste vara literal, swedish_inflection, synonym, compound eller broader_context.\n"
        f"Fråga:\n{question}"
    )


def generate_retrieval_rewrite(
    question: str,
    llm_rewrite: LLMRewriteFn | None,
    *,
    model: str = DEFAULT_REWRITE_MODEL,
) -> dict[str, Any]:
    if llm_rewrite is None:
        return _fallback(question, "agent1_no_llm_callback", model=model)

    prompt = build_retrieval_rewrite_prompt(question)
    try:
        raw_response = llm_rewrite(prompt, model)
    except Exception:
        return _fallback(question, "agent1_exception", model=model)

    return parse_retrieval_rewrite_response(question, raw_response, model=model)


def parse_retrieval_rewrite_response(
    question: str,
    raw_response: str | None,
    *,
    model: str = DEFAULT_REWRITE_MODEL,
) -> dict[str, Any]:
    raw = (raw_response or "").strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        return _fallback(question, "agent1_invalid_json", model=model)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback(question, "agent1_invalid_json", model=model)

    if not isinstance(payload, dict):
        return _fallback(question, "agent1_invalid_json", model=model)

    unexpected_keys = sorted(set(payload) - _ALLOWED_TOP_LEVEL_KEYS)
    if unexpected_keys:
        return _fallback(question, "agent1_unexpected_fields", model=model, extra={"unexpected_fields": unexpected_keys})

    if payload.get("original_question") != question:
        return _fallback(question, "original_question_mismatch", model=model)

    confidence = _safe_float(payload.get("confidence"))
    if confidence is None or confidence < MIN_REWRITE_CONFIDENCE:
        return _fallback(question, "low_confidence", model=model, extra={"confidence": confidence})

    raw_queries = payload.get("retrieval_queries")
    if not isinstance(raw_queries, list):
        return _fallback(question, "agent1_schema_error", model=model)

    semantic_terms = _valid_semantic_terms(payload.get("semantic_terms", []))
    negative_constraints = _valid_string_list(payload.get("negative_constraints", []), max_items=6, max_length=120)

    dropped: list[dict[str, str]] = []
    accepted = [
        {
            "query": question,
            "purpose": "literal",
            "weight": 1.0,
        }
    ]
    seen = {_dedupe_key(question)}

    for item in raw_queries:
        if len(accepted) >= MAX_RETRIEVAL_QUERIES:
            break
        normalized = _normalize_query_item(item)
        if normalized is None:
            dropped.append({"query": _query_for_debug(item), "reason": "schema_error"})
            continue

        query_text = normalized["query"]
        key = _dedupe_key(query_text)
        if key in seen:
            continue
        if _looks_like_answer(query_text):
            dropped.append({"query": query_text, "reason": "answers_question"})
            continue
        if _is_semantic_drift(question, query_text, semantic_terms, negative_constraints):
            dropped.append({"query": query_text, "reason": "semantic_drift"})
            continue

        accepted.append(
            {
                "query": query_text,
                "purpose": normalized["purpose"],
                "weight": round(max(0.35, 1.0 - len(accepted) * 0.12), 2),
            }
        )
        seen.add(key)

    return {
        "status": "ok",
        "model": model,
        "original_question": question,
        "retrieval_queries": accepted,
        "semantic_terms": semantic_terms[:12],
        "negative_constraints": negative_constraints,
        "debug": {
            "agent": "retrieval_rewrite",
            "model": model,
            "fallback_reason": None,
            "confidence": confidence,
            "accepted_query_count": len(accepted),
            "dropped_queries": dropped,
            "token_budget": {"input_target": 600, "output_target": 200},
        },
    }


def _fallback(
    question: str,
    reason: str,
    *,
    model: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    debug = {
        "agent": "retrieval_rewrite",
        "model": model,
        "fallback_reason": reason,
        "accepted_query_count": 1,
        "dropped_queries": [],
        "token_budget": {"input_target": 600, "output_target": 200},
    }
    if extra:
        debug.update(extra)
    return {
        "status": "fallback",
        "model": model,
        "original_question": question,
        "retrieval_queries": [
            {
                "query": question,
                "purpose": "literal",
                "weight": 1.0,
            }
        ],
        "semantic_terms": [],
        "negative_constraints": [],
        "debug": debug,
    }


def _normalize_query_item(item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None
    if set(item) - _ALLOWED_QUERY_KEYS:
        return None
    query = str(item.get("query") or "").strip()
    purpose = str(item.get("purpose") or "").strip()
    if not query or len(query) > 180 or purpose not in _ALLOWED_PURPOSES:
        return None
    return {"query": re.sub(r"\s+", " ", query), "purpose": purpose}


def _valid_semantic_terms(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    terms = []
    for item in value:
        if not isinstance(item, dict) or set(item) - _ALLOWED_TERM_KEYS:
            continue
        surface = str(item.get("surface") or "").strip()
        normalized_family = str(item.get("normalized_family") or "").strip()
        kind = str(item.get("kind") or "").strip()
        if not surface or not normalized_family or kind not in _ALLOWED_TERM_KINDS:
            continue
        terms.append(
            {
                "surface": surface[:60],
                "normalized_family": normalized_family[:60],
                "kind": kind,
            }
        )
        if len(terms) >= 12:
            break
    return terms


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


def _is_semantic_drift(
    original_question: str,
    candidate_query: str,
    semantic_terms: list[dict[str, str]],
    negative_constraints: list[str],
) -> bool:
    original_focus = _content_tokens(original_question)
    candidate_focus = _content_tokens(candidate_query)
    if not candidate_focus:
        return True

    original_folded = _fold(original_question)
    candidate_folded = _fold(candidate_query)

    if _has_direct_focus_bridge(original_focus, candidate_focus, original_folded, candidate_folded):
        return False

    if _has_semantic_term_bridge(original_folded, candidate_folded, semantic_terms):
        return False

    if negative_constraints and _matches_negative_constraint(candidate_folded, negative_constraints):
        return True

    return _char_ngram_similarity(original_folded, candidate_folded) < 0.18


def _has_direct_focus_bridge(
    original_focus: set[str],
    candidate_focus: set[str],
    original_folded: str,
    candidate_folded: str,
) -> bool:
    shared = original_focus & candidate_focus
    if len(shared) >= 1 and not (shared <= {"system", "systemet"}):
        return True
    for token in original_focus:
        if len(token) >= 5 and token in candidate_folded:
            return True
    for token in candidate_focus:
        if len(token) >= 5 and token in original_folded:
            return True
    return False


def _has_semantic_term_bridge(
    original_folded: str,
    candidate_folded: str,
    semantic_terms: list[dict[str, str]],
) -> bool:
    for term in semantic_terms:
        surface = _fold(term["surface"])
        family = _fold(term["normalized_family"])
        if not surface or surface not in candidate_folded:
            continue
        family_tokens = _content_tokens(family)
        if any(token in original_folded for token in family_tokens if len(token) >= 4):
            return True
        if _char_ngram_similarity(original_folded, family) >= 0.16:
            return True
    return False


def _matches_negative_constraint(candidate_folded: str, negative_constraints: list[str]) -> bool:
    candidate_tokens = _content_tokens(candidate_folded)
    for constraint in negative_constraints:
        constraint_tokens = _content_tokens(constraint)
        if candidate_tokens & constraint_tokens:
            return True
    return False


def _looks_like_answer(query_text: str) -> bool:
    stripped = query_text.strip()
    if any(pattern.search(stripped) for pattern in _ANSWER_LIKE_PATTERNS):
        return True
    return stripped.endswith(".") and len(_content_tokens(stripped)) >= 6


def _content_tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-zåäöA-ZÅÄÖ0-9]+", _fold(text)):
        token = _stem_token(token)
        if len(token) < 3 or token in _STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _stem_token(token: str) -> str:
    for suffix in ("arnas", "ernas", "ande", "ades", "ats", "ade", "ing", "ning", "arna", "erna", "het", "en", "et", "ar", "er", "at"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]
    return token


def _char_ngram_similarity(left: str, right: str) -> float:
    left_grams = _char_ngrams(left)
    right_grams = _char_ngrams(right)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _char_ngrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    return {compact[index : index + 4] for index in range(max(0, len(compact) - 3))}


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _query_for_debug(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("query") or "")
    return str(item or "")


def _dedupe_key(text: str) -> str:
    return re.sub(r"\s+", " ", _fold(text)).strip()


def _fold(text: str) -> str:
    return (
        (text or "")
        .lower()
        .replace("å", "a")
        .replace("ä", "a")
        .replace("ö", "o")
    )
