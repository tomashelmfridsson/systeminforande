# rag/search.py

import json
import math
import os
import re
from collections import Counter


DATA_DIR = "rag/data"
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
STOPWORDS = {
    "vad", "är", "hur", "ska", "kan", "det", "de", "den", "och", "att", "som",
    "för", "från", "med", "till", "om", "i", "på", "av", "en", "ett", "vid",
    "utifrån", "finns", "används", "ingår", "vilka", "vilken", "då", "har",
    "eller", "utan", "också", "samt", "bara", "inte", "under", "över", "efter"
}
TITLE_BOOST = 1.8
DEFINITION_TITLE_BOOST = 1.5
BM25_K1 = 1.5
BM25_B = 0.75
DOMAIN_RULES = [
    {
        "when_any": {"arbetsområde", "arbetsområd"},
        "source_any": {"checklista_arbetsomraden.pdf"},
        "title_any": {"arbetsområd", "modell", "kompetens", "systembyte"},
        "boost": 6.0,
        "require_source_for_generic_titles": True,
    },
    {
        "when_any": {"införandekrav", "införandekravet", "kravområde", "kravområd"},
        "source_any": {"checklista_inforandekrav.pdf"},
        "title_any": {"införandekrav", "kravtyp", "kravområd", "modell", "kompetens"},
        "boost": 6.0,
        "require_source_for_generic_titles": True,
    },
    {
        "when_any": {"fas", "faser", "etapp", "etapper", "aktivitet", "aktivitete"},
        "source_any": {"checklista_arbetsomraden.pdf", "checklista_inforandekrav.pdf"},
        "title_any": {"fas", "etapp", "aktivitet", "inledning"},
        "boost": 4.5,
        "require_source_for_generic_titles": True,
    },
    {
        "when_any": {"planering", "plan", "tidplan"},
        "source_any": {"checklista_arbetsomraden.pdf", "102_projektbeskrivning.pdf", "112_projektbeskrivning.pdf"},
        "title_any": {"plan", "planering", "genomförande"},
        "boost": 3.5,
    },
]
OVERVIEW_SECTION_BOOST = 4.0
DETAIL_SECTION_PENALTY = 1.5

_CHUNKS_CACHE = None
_INDEX_CACHE = None


def load_chunks():
    global _CHUNKS_CACHE
    if _CHUNKS_CACHE is not None:
        return _CHUNKS_CACHE

    if not os.path.exists(CHUNKS_FILE):
        _CHUNKS_CACHE = []
        return _CHUNKS_CACHE

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        _CHUNKS_CACHE = json.load(f)
    return _CHUNKS_CACHE


def _tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"\w+", text.lower())
    tokens = []
    for token in raw_tokens:
        normalized = _normalize_token(token)
        if normalized and normalized not in STOPWORDS and len(normalized) > 2:
            tokens.append(normalized)
    return tokens


def _normalize_token(token: str) -> str:
    token = token.lower()
    if token.endswith("erna") and len(token) > 6:
        token = token[:-1]
    elif token.endswith("arna") and len(token) > 6:
        token = token[:-1]
    elif token.endswith("or") and len(token) > 5:
        token = token[:-2]
    elif token.endswith("ar") and len(token) > 5:
        token = token[:-2]
    elif token.endswith("er") and len(token) > 5:
        token = token[:-2]
    elif token.endswith("en") and len(token) > 5:
        token = token[:-2]
    elif token.endswith("n") and len(token) > 5:
        token = token[:-1]
    return token


def _build_index():
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    chunks = load_chunks()
    documents = []
    doc_freq = Counter()
    total_length = 0

    for chunk in chunks:
        title_tokens = _tokenize(chunk.get("title", ""))
        text_tokens = _tokenize(chunk.get("text", ""))
        tokens = title_tokens + text_tokens
        token_counts = Counter(tokens)

        for term in token_counts:
            doc_freq[term] += 1

        total_length += len(tokens)
        documents.append(
            {
                "chunk": chunk,
                "tokens": tokens,
                "token_counts": token_counts,
                "length": len(tokens),
                "title_tokens": set(title_tokens),
                "source_lower": (chunk.get("source", "") or "").lower(),
                "title_lower": (chunk.get("title", "") or "").lower(),
                "section": chunk.get("section", "") or "",
            }
        )

    avg_doc_len = total_length / len(documents) if documents else 0.0
    _INDEX_CACHE = {
        "documents": documents,
        "doc_freq": doc_freq,
        "doc_count": len(documents),
        "avg_doc_len": avg_doc_len,
    }
    return _INDEX_CACHE


def _idf(term: str, doc_freq: Counter, doc_count: int) -> float:
    freq = doc_freq.get(term, 0)
    if doc_count == 0:
        return 0.0
    return math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))


def _bm25_score(query_terms: list[str], document: dict, index: dict) -> float:
    if not query_terms or document["length"] == 0 or index["avg_doc_len"] == 0:
        return 0.0

    score = 0.0
    for term in query_terms:
        term_freq = document["token_counts"].get(term, 0)
        if term_freq == 0:
            continue

        idf = _idf(term, index["doc_freq"], index["doc_count"])
        numerator = term_freq * (BM25_K1 + 1)
        denominator = term_freq + BM25_K1 * (
            1 - BM25_B + BM25_B * (document["length"] / index["avg_doc_len"])
        )
        score += idf * (numerator / denominator)

    return score


def _heuristic_boost(query: str, query_terms: list[str], document: dict) -> float:
    boost = 0.0
    title_overlap = len(set(query_terms) & document["title_tokens"])
    boost += title_overlap * TITLE_BOOST

    if query.lower().startswith("vad är") and any(
        token in document["title_tokens"]
        for token in ["inledning", "syfte", "omfattning", "arbetsområde", "arbetsområd"]
    ):
        boost += DEFINITION_TITLE_BOOST

    boost += _domain_boost(set(query_terms), document)
    boost += _query_intent_boost(query, set(query_terms), document)

    return boost


def _domain_boost(query_terms: set[str], document: dict) -> float:
    boost = 0.0
    source_lower = document["source_lower"]
    title_lower = document["title_lower"]

    for rule in DOMAIN_RULES:
        if not (query_terms & rule["when_any"]):
            continue

        source_match = any(source_term in source_lower for source_term in rule["source_any"])
        title_match = any(title_term in title_lower for title_term in rule["title_any"])

        matched = source_match or title_match

        if rule.get("require_source_for_generic_titles") and title_match and not source_match:
            generic_titles = {"inledning", "syfte", "krav", "modell", "fas", "etapp", "aktivitet"}
            title_tokens = set(_tokenize(title_lower))
            if title_tokens & generic_titles:
                matched = False

        if matched:
            boost += rule["boost"]

    return boost


def _query_intent_boost(query: str, query_terms: set[str], document: dict) -> float:
    boost = 0.0
    title_lower = document["title_lower"]
    source_lower = document["source_lower"]
    section = document["section"]
    is_definition = query.lower().startswith("vad är") or "syfte" in query_terms
    is_overview = bool(query_terms & {"kravområd", "arbetsområd", "fas", "etapp"})

    if "checklista_arbetsomraden.pdf" in source_lower:
        if is_definition and any(term in title_lower for term in ["systembyte", "kompetens", "modell"]):
            boost += OVERVIEW_SECTION_BOOST
        if is_overview and section.startswith("4."):
            boost -= DETAIL_SECTION_PENALTY

    if "checklista_inforandekrav.pdf" in source_lower:
        if is_definition and any(term in title_lower for term in ["kravtyp", "modell", "kompetens"]):
            boost += OVERVIEW_SECTION_BOOST
        if is_overview and any(term in title_lower for term in ["kravtyp", "modell"]):
            boost += OVERVIEW_SECTION_BOOST
        if is_overview and section.startswith("4."):
            boost -= DETAIL_SECTION_PENALTY

    if "planering" in query_terms and "checklista_arbetsomraden.pdf" in source_lower:
        if any(term in title_lower for term in ["modell", "systembyte", "kompetens"]):
            boost += 2.5

    return boost


def search(query: str, top_k: int = 5):
    index = _build_index()
    print(f"🔍 Search: laddade {index['doc_count']} chunkar")
    if not index["documents"]:
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return []

    scored = []
    for document in index["documents"]:
        score = _bm25_score(query_terms, document, index)
        score += _heuristic_boost(query, query_terms, document)
        if score > 0:
            scored.append((score, document["chunk"]))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]
