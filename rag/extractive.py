import re
from collections import OrderedDict

STOPWORDS = {
    "vad", "är", "hur", "ska", "kan", "det", "de", "den", "och", "att", "som",
    "för", "från", "med", "till", "om", "i", "på", "av", "en", "ett", "vid",
    "utifrån", "finns", "används", "ingår", "vilka", "vilken", "då", "har"
}


def build_extractive_reasoning(query: str, chunks: list, max_sentences: int = 5) -> str:
    """
    Bygg ett deterministiskt resonemang från toppchunkar utan extern LLM.
    Returnerar en kort, sammanhängande svensk text baserad enbart på källutdrag.
    """
    sentences = _collect_relevant_sentences(query, chunks, limit=max_sentences)
    if not sentences:
        return (
            "Det finns inte tillräckligt tydliga formuleringar i de mest relevanta "
            "källutdragen för att bygga ett resonemang."
        )

    intro = _build_intro(query, chunks)
    body = " ".join(sentences)
    closing = _build_closing(chunks, sentences)
    return " ".join(part for part in [intro, body, closing] if part).strip()


def _collect_relevant_sentences(query: str, chunks: list, limit: int) -> list[str]:
    query_terms = _terms(query)
    ranked = []
    seen = set()

    for chunk in chunks:
        for sentence in _split_sentences(chunk.get("text", "")):
            normalized = _normalize_sentence(sentence)
            if not normalized or normalized in seen:
                continue

            score = _score_sentence(sentence, query_terms, chunk)
            if score <= 0:
                continue

            ranked.append((score, normalized))
            seen.add(normalized)

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [sentence for _, sentence in ranked[:limit]]
    return _dedupe_preserve_order(selected)


def _build_intro(query: str, chunks: list) -> str:
    sources = [chunk.get("title") for chunk in chunks if chunk.get("title")]
    if not sources:
        return "Utifrån de mest relevanta källutdragen framgår följande."

    first_title = sources[0].rstrip(".")
    return (
        f"Utifrån de mest relevanta källutdragen om \"{query}\" "
        f"framgår följande i materialet, särskilt i avsnittet \"{first_title}\"."
    )


def _build_closing(chunks: list, sentences: list[str]) -> str:
    pages = []
    for chunk in chunks:
        pages.extend(chunk.get("pages") or [])

    if not pages or not sentences:
        return ""

    unique_pages = sorted(set(pages))
    if len(unique_pages) == 1:
        page_text = f"sida {unique_pages[0]}"
    else:
        page_text = f"sidorna {unique_pages[0]}-{unique_pages[-1]}"

    return f"Sammanställningen bygger på de högst rankade utdragen från {page_text}."


def _score_sentence(sentence: str, query_terms: set[str], chunk: dict) -> int:
    sentence_terms = _terms(sentence)
    overlap = len(query_terms & sentence_terms)
    if overlap == 0:
        return 0

    score = overlap * 3

    title_terms = _terms(chunk.get("title", ""))
    score += len(query_terms & title_terms) * 2

    if len(sentence_terms) > 8:
        score += 1

    if any(token in sentence.lower() for token in ["ska", "innebär", "beskriv", "omfattar", "ansvar", "mål"]):
        score += 1

    return score


def _split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []

    raw_parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        raw_parts.extend(re.split(r"(?<=[.!?])\s+", line))

    return [part.strip(" -") for part in raw_parts if part.strip(" -")]


def _normalize_sentence(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    if len(sentence) < 40 or len(sentence) > 320:
        return ""
    if re.fullmatch(r"[\d\W_]+", sentence):
        return ""
    return sentence


def _terms(text: str) -> set[str]:
    return {
        term for term in re.findall(r"\w+", text.lower())
        if term not in STOPWORDS and len(term) > 2
    }


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    return list(OrderedDict.fromkeys(items))
