# rag/search.py

import json
import math
import os
import re
from collections import Counter
from pathlib import Path


DATA_DIR = "rag/data"
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
STOPWORDS = {
    "vad", "är", "hur", "ska", "kan", "det", "de", "den", "och", "att", "som",
    "för", "från", "med", "till", "om", "i", "på", "av", "en", "ett", "vid",
    "utifrån", "finns", "används", "ingår", "vilka", "vilken", "då", "har",
    "eller", "utan", "också", "samt", "bara", "inte", "under", "över", "efter",
    "bör", "sätt", "vilket",
}
CANONICAL_TERM_MAP = {
    "arbetsomrade": "arbetsomrade",
    "arbetsomraden": "arbetsomrade",
    "arbetsomraden": "arbetsomrade",
    "acceptans": "acceptanstest",
    "acceptanstest": "acceptanstest",
    "acceptanstesten": "acceptanstest",
    "acceptanstester": "acceptanstest",
    "acceptenstest": "acceptanstest",
    "acceptanstes": "acceptanstest",
    "acceptanstst": "acceptanstest",
    "leveransgodkannande": "leveransgodkannande",
    "godkannande": "leveransgodkannande",
    "inforande": "inforande",
    "inforandet": "inforande",
    "implementering": "implementering",
    "implementationen": "implementering",
    "implmentering": "implementering",
    "planering": "planering",
    "planeras": "planering",
    "uppfoljning": "uppfoljning",
    "foljas": "uppfoljning",
    "verifiering": "verifiering",
    "testplan": "testplan",
    "testpla": "testplan",
}
QUERY_SYNONYMS = {
    "arbetsmodell": {
        "inforandemodell",
        "inforandeprocess",
        "projektstyrningsmodell",
        "implementering",
    },
    "inforandemodell": {
        "arbetsmodell",
        "inforandeprocess",
        "projektstyrningsmodell",
    },
    "acceptanstest": {
        "testplan",
        "testrapport",
        "leveransgodkannande",
        "verifiering",
        "godkannande",
    },
    "arbetsomrade": {
        "arbetsomraden",
        "checklista",
        "aktivitet",
        "planering",
    },
    "implementering": {
        "inforande",
        "planering",
        "uppfoljning",
        "verifiering",
        "konvertering",
        "acceptanstest",
    },
}
TITLE_BOOST = 1.8
SOURCE_BOOST = 2.2
FAMILY_BOOST = 3.5
DEFINITION_TITLE_BOOST = 1.5
BM25_K1 = 1.5
BM25_B = 0.75
MIN_RESULT_SCORE = 1.5
RELATIVE_SCORE_CUTOFF = 0.35
DOMAIN_RULES = [
    {
        "when_any": {"arbetsomrade"},
        "source_any": {"checklista_arbetsomraden.pdf"},
        "title_any": {"arbetsomrade", "modell", "kompetens", "systembyte"},
        "boost": 6.0,
        "require_source_for_generic_titles": True,
    },
    {
        "when_any": {"inforandekrav", "kravomrade"},
        "source_any": {"checklista_inforandekrav.pdf"},
        "title_any": {"inforandekrav", "kravtyp", "kravomrade", "modell", "kompetens"},
        "boost": 6.0,
        "require_source_for_generic_titles": True,
    },
    {
        "when_any": {"fas", "etapp", "aktivitet"},
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
    {
        "when_any": {"acceptanstest", "leveransgodkannande", "testplan", "testrapport"},
        "source_any": {
            "210_acceptanstest_testplan.pdf",
            "211_acceptanstest_testrapport.pdf",
            "220_acceptanstest_delprojektplan.pdf",
            "221_acceptanstest_testoversikt.pdf",
            "222_acceptanstest_testplan.pdf",
            "223_acceptanstest_korningsschema.pdf",
            "224_acceptanstest_presentation.pdf",
            "225_acceptanstest_praktiskt_om_prestandatester.pdf",
            "226_acceptanstest_checklista_icke_funktionella_krav_batch.pdf",
            "227_acceptanstest_checklista_icke_funktionella_krav_online.pdf",
            "230_acceptanstest_projektstatusrapport.pdf",
            "231_acceptanstest_krav_leveransgodkannande.pdf",
            "checklista_arbetsomraden.pdf",
            "checklista_inforandekrav.pdf",
        },
        "title_any": {"acceptanstest", "testplan", "testrapport", "leveransgodkannande"},
        "boost": 7.0,
    },
    {
        "when_any": {"implementering", "inforande", "planering", "uppfoljning", "verifiering", "konvertering"},
        "source_any": {
            "102_projektbeskrivning.pdf",
            "112_projektbeskrivning.pdf",
            "210_acceptanstest_testplan.pdf",
            "350_konvertering_strategi.pdf",
            "354_konvertering_verifiering_kontroller.pdf",
        },
        "title_any": {"planering", "uppfoljning", "strategi", "verifiering", "genomforande"},
        "boost": 4.5,
    },
]
OVERVIEW_SECTION_BOOST = 4.0
DETAIL_SECTION_PENALTY = 1.5
WEB_RESULT_PENALTY = 5.0

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
    raw_tokens = re.findall(r"\w+", _ascii_fold(text.lower()))
    tokens = []
    for token in raw_tokens:
        normalized = _normalize_token(token)
        if normalized and normalized not in NORMALIZED_STOPWORDS and len(normalized) > 2:
            tokens.append(normalized)
    return tokens


def _ascii_fold(text: str) -> str:
    return (
        text.replace("å", "a")
        .replace("ä", "a")
        .replace("ö", "o")
        .replace("Å", "A")
        .replace("Ä", "A")
        .replace("Ö", "O")
    )


def _normalize_token(token: str) -> str:
    token = _ascii_fold(token.lower())
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
    return CANONICAL_TERM_MAP.get(token, token)


NORMALIZED_STOPWORDS = {_normalize_token(word) for word in STOPWORDS}


def _extract_source_tokens(source: str) -> list[str]:
    stem = Path(source).stem
    return _tokenize(re.sub(r"[_-]+", " ", stem))


def _document_family(source_tokens: list[str]) -> str:
    families = {
        "acceptanstest",
        "konvertering",
        "utbildning",
        "dokumentation",
        "omlaggning",
        "provdrift",
        "driftsattning",
        "projekt",
        "arbetsomrade",
        "inforandekrav",
    }
    for token in source_tokens:
        if token in families:
            return token
    return ""


def classify_query_intent(query: str) -> str:
    query_terms = set(_tokenize(query))
    query_lower = query.lower().strip()

    if query_lower.startswith("vad är") or query_lower.startswith("vad innebär"):
        return "definition"
    if "syfte" in query_terms:
        return "purpose"
    if query_lower.startswith("vilka") or "vilka" in query_lower:
        if query_terms & {"kravområd", "arbetsområd", "fas", "etapp", "aktivitet"}:
            return "overview_list"
        return "list"
    if query_lower.startswith("hur"):
        return "process"
    if query_terms & {"nar", "beslut", "faststalld", "driftsattning"}:
        return "timing_or_decision"
    return "general"


def _expand_query_terms(query_terms: list[str]) -> list[str]:
    expanded = []
    seen = set()

    for term in query_terms:
        if term not in seen:
            expanded.append(term)
            seen.add(term)

        for synonym in QUERY_SYNONYMS.get(term, set()):
            normalized = _normalize_token(synonym)
            if normalized and normalized not in STOPWORDS and len(normalized) > 2 and normalized not in seen:
                expanded.append(normalized)
                seen.add(normalized)

    return expanded


def _edit_distance_at_most(a: str, b: str, max_distance: int) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > max_distance:
        return False

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        row_min = current[0]
        for j, char_b in enumerate(b, start=1):
            substitution_cost = 0 if char_a == char_b else 1
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + substitution_cost,
            )
            current.append(value)
            row_min = min(row_min, value)

        if row_min > max_distance:
            return False
        previous = current

    return previous[-1] <= max_distance


def _fuzzy_matches(term: str, vocabulary: set[str]) -> list[str]:
    if len(term) < 5:
        return []

    max_distance = 1 if len(term) < 10 else 2
    candidates = []
    for candidate in vocabulary:
        if candidate == term:
            continue
        if candidate[:1] != term[:1]:
            continue
        if _edit_distance_at_most(term, candidate, max_distance):
            candidates.append(candidate)

    return sorted(candidates, key=lambda candidate: (abs(len(candidate) - len(term)), candidate))[:3]


def _expand_query_with_vocabulary(query_terms: list[str], vocabulary: set[str]) -> list[str]:
    expanded = _expand_query_terms(query_terms)
    seen = set(expanded)

    for term in list(expanded):
        for candidate in _fuzzy_matches(term, vocabulary):
            if candidate not in seen:
                expanded.append(candidate)
                seen.add(candidate)

    return expanded


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
        source_tokens = _extract_source_tokens(chunk.get("source", ""))
        tokens = title_tokens + text_tokens + source_tokens
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
                "source_tokens": set(source_tokens),
                "document_family": _document_family(source_tokens),
                "source_lower": _ascii_fold((chunk.get("source", "") or "").lower()),
                "title_lower": _ascii_fold((chunk.get("title", "") or "").lower()),
                "section": chunk.get("section", "") or "",
            }
        )

    avg_doc_len = total_length / len(documents) if documents else 0.0
    _INDEX_CACHE = {
        "documents": documents,
        "doc_freq": doc_freq,
        "doc_count": len(documents),
        "avg_doc_len": avg_doc_len,
        "vocabulary": set(doc_freq),
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
    boost += len(set(query_terms) & document["source_tokens"]) * SOURCE_BOOST

    if query.lower().startswith("vad är") and any(
        token in document["title_tokens"]
        for token in ["inledning", "syfte", "omfattning", "arbetsomrade"]
    ):
        boost += DEFINITION_TITLE_BOOST

    boost += _domain_boost(set(query_terms), document)
    boost += _query_intent_boost(query, set(query_terms), document)
    if document["chunk"].get("source_type") == "web":
        boost -= WEB_RESULT_PENALTY

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
    is_overview = bool(query_terms & {"kravomrade", "arbetsomrade", "fas", "etapp"})

    if "checklista_arbetsomraden.pdf" in source_lower:
        if is_definition and any(term in title_lower for term in ["systembyte", "kompetens", "modell"]):
            boost += OVERVIEW_SECTION_BOOST
        if is_overview and section.startswith("4."):
            boost -= DETAIL_SECTION_PENALTY
        if is_definition and section in {"1", "2", "3"}:
            boost += 3.0

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

    if "acceptanstest" in query_terms:
        if document["document_family"] == "acceptanstest":
            boost += FAMILY_BOOST
        if is_definition and section.startswith("1"):
            boost += 2.0
        if "acceptanstest" in title_lower:
            boost += 2.0

    if "arbetsomrade" in query_terms and document["document_family"] == "arbetsomrade":
        boost += FAMILY_BOOST
        if section.startswith("4."):
            boost -= DETAIL_SECTION_PENALTY

    return boost


def _score_document(query: str, query_terms: list[str], document: dict, index: dict) -> dict:
    bm25 = _bm25_score(query_terms, document, index)
    title_overlap = len(set(query_terms) & document["title_tokens"]) * TITLE_BOOST

    definition_boost = 0.0
    if query.lower().startswith("vad är") and any(
        token in document["title_tokens"]
        for token in ["inledning", "syfte", "omfattning", "arbetsområde", "arbetsområd"]
    ):
        definition_boost = DEFINITION_TITLE_BOOST

    domain_boost = _domain_boost(set(query_terms), document)
    intent_boost = _query_intent_boost(query, set(query_terms), document)
    total = bm25 + title_overlap + definition_boost + domain_boost + intent_boost

    return {
        "bm25": round(bm25, 4),
        "title_overlap": round(title_overlap, 4),
        "definition_boost": round(definition_boost, 4),
        "domain_boost": round(domain_boost, 4),
        "intent_boost": round(intent_boost, 4),
        "total": round(total, 4),
    }


def _matched_query_terms(query_terms: list[str], document: dict) -> set[str]:
    document_terms = set(document["token_counts"]) | document["title_tokens"] | document["source_tokens"]
    return {term for term in query_terms if term in document_terms}


def _has_retrieval_support(
    original_query_terms: list[str],
    expanded_query_terms: list[str],
    document: dict,
) -> bool:
    matched_original = _matched_query_terms(original_query_terms, document)
    if matched_original:
        return True

    if len(original_query_terms) == 1:
        matched_expanded = _matched_query_terms(expanded_query_terms, document)
        return bool(matched_expanded)

    return False


def _prune_scored_results(scored: list[dict], top_k: int) -> list[dict]:
    if not scored:
        return []

    top_score = scored[0]["score"]
    cutoff = max(MIN_RESULT_SCORE, top_score * RELATIVE_SCORE_CUTOFF)
    pruned = [item for item in scored if item["score"] >= cutoff]
    return pruned[:top_k]


def search(query: str, top_k: int = 5):
    index = _build_index()
    print(f"🔍 Search: laddade {index['doc_count']} chunkar")
    if not index["documents"]:
        return []

    original_query_terms = _tokenize(query)
    if not original_query_terms:
        return []
    query_terms = _expand_query_with_vocabulary(original_query_terms, index["vocabulary"])

    scored = []
    for document in index["documents"]:
        score = _score_document(query, query_terms, document, index)["total"]
        if score <= 0:
            continue
        if not _has_retrieval_support(original_query_terms, query_terms, document):
            continue
        scored.append({"score": score, "chunk": document["chunk"]})

    scored.sort(key=lambda item: item["score"], reverse=True)
    pruned = _prune_scored_results(scored, top_k)
    return [(item["score"], item["chunk"]) for item in pruned]


def explain_search(query: str, top_k: int = 5) -> dict:
    index = _build_index()
    original_query_terms = _tokenize(query)
    query_terms = _expand_query_with_vocabulary(original_query_terms, index["vocabulary"])
    intent = classify_query_intent(query)

    scored = []
    for document in index["documents"]:
        parts = _score_document(query, query_terms, document, index)
        if parts["total"] <= 0:
            continue
        if not _has_retrieval_support(original_query_terms, query_terms, document):
            continue
        scored.append(
            {
                "score": parts["total"],
                "parts": parts,
                "chunk": document["chunk"],
                "matched_terms": sorted(_matched_query_terms(query_terms, document)),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    top = _prune_scored_results(scored, top_k)

    return {
        "query": query,
        "query_terms": original_query_terms,
        "expanded_query_terms": query_terms,
        "intent": intent,
        "top_results": top,
    }
