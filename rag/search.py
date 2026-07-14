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
    "projektbibliotek": {
        "projektstyrning",
        "dokument",
        "lagras",
    },
    "projektorganisation": {
        "projektstyrning",
        "roller",
        "bemanning",
        "ansvar",
    },
    "checklista": {
        "mall",
        "kravmall",
    },
    "inforandekrav": {
        "kravtyper",
        "kravomraden",
        "kravmall",
        "checklista",
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
OVERVIEW_SECTION_BOOST = 4.0
DETAIL_SECTION_PENALTY = 1.5
WEB_RESULT_PENALTY = 5.0
RERANK_POOL_SIZE = 12
RERANK_LEXICAL_WEIGHT = 0.7
RERANK_SIMILARITY_WEIGHT = 0.3
EXISTENCE_TOKENS = {"checklista", "mall", "modell", "process", "kravmall", "arbetsgang"}
OVERVIEW_TOKENS = {
    "allmant", "oversikt", "modell", "modeller", "systembyte", "kravtyp", "kravtyper",
    "kompetens", "kompetenser", "arbetsomrade", "arbetsomraden", "projektstyrningsmodell",
}
PROCESS_TOKENS = {
    "planering", "uppfoljning", "genomforande", "strategi", "verifiering", "aktiviteter",
    "aktivitetsforteckning", "grov", "plan", "tidplan",
}
ROLE_TOKENS = {"projektorganisation", "roller", "ansvar", "arbetsformer", "bemanning"}
LIBRARY_TOKENS = {"projektbibliotek", "dokument", "filer", "historik", "forvaltning"}
GENERIC_QUERY_TOKENS = {
    "planering", "uppfoljning", "genomforande", "strategi", "verifiering", "modell",
    "process", "checklista", "mall", "ansvar", "roller", "dokument", "lagras",
    "inforande", "implementering", "projektstyrning",
}

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

    boost += _generic_pattern_boost(query, set(query_terms), document)
    boost += _query_intent_boost(query, set(query_terms), document)
    if document["chunk"].get("source_type") == "web":
        boost -= WEB_RESULT_PENALTY

    return boost


def _generic_pattern_boost(query: str, query_terms: set[str], document: dict) -> float:
    boost = 0.0
    title_tokens = document["title_tokens"]
    source_tokens = document["source_tokens"]
    combined_tokens = title_tokens | source_tokens
    section = document["section"]
    intent = classify_query_intent(query)
    overlap = len(query_terms & combined_tokens)
    topical_terms = _topical_query_terms(query_terms)
    topical_title_overlap = len(topical_terms & title_tokens)
    topical_source_overlap = len(topical_terms & source_tokens)

    if intent == "definition":
        boost += overlap * 1.2
        if section in {"1", "2", "3"}:
            boost += 2.0
        if _section_depth(section) <= 1:
            boost += 1.5
        if title_tokens & OVERVIEW_TOKENS:
            boost += 3.5
        if query_terms & EXISTENCE_TOKENS and title_tokens & EXISTENCE_TOKENS:
            boost += 4.0

    if intent in {"overview_list", "list"}:
        if title_tokens & OVERVIEW_TOKENS:
            boost += 4.0
        if section.startswith("4."):
            boost -= DETAIL_SECTION_PENALTY

    if intent == "process":
        if title_tokens & PROCESS_TOKENS:
            boost += 4.0
        if source_tokens & PROCESS_TOKENS:
            boost += 2.5
        if topical_title_overlap:
            boost += 4.5
        elif topical_source_overlap:
            boost -= 1.5

    if query.lower().startswith("finns det"):
        if overlap >= 1:
            boost += 2.0
        if title_tokens & EXISTENCE_TOKENS:
            boost += 3.0

    if "projektorganisation" in query_terms:
        if title_tokens & ROLE_TOKENS:
            boost += 6.0
        if section.endswith(".1") and "projektorganisation" in title_tokens:
            boost -= 2.0

    if "projektbibliotek" in query_terms and title_tokens & LIBRARY_TOKENS:
        boost += 6.0

    if "arbetsomrade" in query_terms:
        if title_tokens & {"arbetsomrade", "arbetsomraden", "systembyte", "modeller", "kompetenser"}:
            boost += 5.0
        if "acceptanstest" in title_tokens and "arbetsomrade" not in combined_tokens:
            boost -= 4.0

    if "inforandekrav" in query_terms:
        if title_tokens & {"kravtyp", "kravtyper", "kravomrade", "kravomraden", "modeller", "kompetenser"}:
            boost += 4.0
        if "checklista" in query_terms and "checklista" in source_tokens:
            boost += 4.0

    return boost


def _query_intent_boost(query: str, query_terms: set[str], document: dict) -> float:
    boost = 0.0
    title_lower = document["title_lower"]
    source_lower = document["source_lower"]
    section = document["section"]
    is_definition = query.lower().startswith("vad är") or "syfte" in query_terms

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
    if "arbetsomrade" in query_terms:
        if is_definition and "arbetsomrad" not in title_lower and "arbetsomrad" not in source_lower:
            boost -= 6.0

    if "projektorganisation" in query_terms:
        if is_definition and "projektorganisation" not in title_lower and "projektorganisation" not in source_lower:
            boost -= 5.0

    if "projektbibliotek" in query_terms:
        if is_definition and "projektbibliotek" not in title_lower and "projektbibliotek" not in source_lower:
            boost -= 5.0

    if is_definition and _section_depth(section) >= 2 and not (document["title_tokens"] & query_terms):
        boost -= 1.5

    return boost


def _section_depth(section: str) -> int:
    if not section:
        return 99
    return section.count(".")


def _topical_query_terms(query_terms: set[str]) -> set[str]:
    return {
        term for term in query_terms
        if term not in GENERIC_QUERY_TOKENS
    }


def _score_document(query: str, query_terms: list[str], document: dict, index: dict) -> dict:
    bm25 = _bm25_score(query_terms, document, index)
    title_overlap = len(set(query_terms) & document["title_tokens"]) * TITLE_BOOST

    definition_boost = 0.0
    if query.lower().startswith("vad är") and any(
        token in document["title_tokens"]
        for token in ["inledning", "syfte", "omfattning", "arbetsområde", "arbetsområd"]
    ):
        definition_boost = DEFINITION_TITLE_BOOST

    domain_boost = _generic_pattern_boost(query, set(query_terms), document)
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


def _rerank_candidates(query: str, scored: list[dict]) -> list[dict]:
    if len(scored) < 2:
        return scored

    lexical_sorted = sorted(scored, key=lambda item: item["score"], reverse=True)
    rerank_pool = lexical_sorted[:RERANK_POOL_SIZE]
    tail = lexical_sorted[RERANK_POOL_SIZE:]

    lexical_scores = [item["score"] for item in rerank_pool]
    lexical_min = min(lexical_scores)
    lexical_max = max(lexical_scores)
    lexical_span = lexical_max - lexical_min

    similarities = _candidate_similarities(query, rerank_pool)
    sim_min = min(similarities) if similarities else 0.0
    sim_max = max(similarities) if similarities else 0.0
    sim_span = sim_max - sim_min

    reranked = []
    for item, similarity in zip(rerank_pool, similarities):
        lexical_norm = 1.0 if lexical_span == 0 else (item["score"] - lexical_min) / lexical_span
        similarity_norm = 1.0 if sim_span == 0 else (similarity - sim_min) / sim_span
        combined = (
            lexical_norm * RERANK_LEXICAL_WEIGHT
            + similarity_norm * RERANK_SIMILARITY_WEIGHT
        )
        updated = dict(item)
        updated["rerank_similarity"] = round(similarity, 4)
        updated["rerank_score"] = round(combined, 4)
        reranked.append(updated)

    reranked.sort(key=lambda item: (item["rerank_score"], item["score"]), reverse=True)
    return reranked + tail


def _candidate_similarities(query: str, scored: list[dict]) -> list[float]:
    query_vector = _char_tfidf_vector(_rerank_text(query))
    document_vectors = [
        _char_tfidf_vector(_rerank_text_for_chunk(item["chunk"]))
        for item in scored
    ]

    texts = [_rerank_text(query)] + [_rerank_text_for_chunk(item["chunk"]) for item in scored]
    doc_freq = Counter()
    for text in texts:
        for gram in set(_char_ngrams(text)):
            doc_freq[gram] += 1

    doc_count = len(texts)
    query_weighted = _apply_idf(query_vector, doc_freq, doc_count)
    query_norm = _vector_norm(query_weighted)
    if query_norm == 0:
        return [0.0 for _ in scored]

    similarities = []
    for vector in document_vectors:
        weighted = _apply_idf(vector, doc_freq, doc_count)
        norm = _vector_norm(weighted)
        if norm == 0:
            similarities.append(0.0)
            continue
        similarities.append(_dot_product(query_weighted, weighted) / (query_norm * norm))

    return similarities


def _rerank_text(text: str) -> str:
    folded = _ascii_fold(text.lower())
    return re.sub(r"\s+", " ", folded).strip()


def _rerank_text_for_chunk(chunk: dict) -> str:
    title = chunk.get("title", "") or ""
    source = Path(chunk.get("source", "") or "").stem.replace("_", " ")
    text = chunk.get("text", "") or ""
    # Duplicate title to make high-level document semantics matter more than long body text.
    return _rerank_text(f"{title} {title} {source} {text}")


def _char_tfidf_vector(text: str) -> Counter:
    return Counter(_char_ngrams(text))


def _char_ngrams(text: str) -> list[str]:
    compact = f" {text} "
    grams = []
    for size in (3, 4, 5):
        if len(compact) < size:
            continue
        for index in range(len(compact) - size + 1):
            gram = compact[index:index + size]
            if gram.strip():
                grams.append(gram)
    return grams


def _apply_idf(vector: Counter, doc_freq: Counter, doc_count: int) -> dict[str, float]:
    weighted = {}
    for gram, count in vector.items():
        idf = _idf(gram, doc_freq, doc_count)
        weighted[gram] = count * idf
    return weighted


def _vector_norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _dot_product(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


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

    scored = _rerank_candidates(query, scored)
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

    scored = _rerank_candidates(query, scored)
    top = _prune_scored_results(scored, top_k)

    return {
        "query": query,
        "query_terms": original_query_terms,
        "expanded_query_terms": query_terms,
        "intent": intent,
        "top_results": top,
    }
