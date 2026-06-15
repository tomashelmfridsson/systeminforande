import re
from rag.search import classify_query_intent, load_chunks

STOPWORDS = {
    "vad", "är", "hur", "ska", "kan", "det", "de", "den", "och", "att", "som",
    "för", "från", "med", "till", "om", "i", "på", "av", "en", "ett", "vid",
    "utifrån", "finns", "används", "ingår", "vilka", "vilken", "då", "har"
}

BAD_TITLE_PATTERNS = (
    "_____",
    "sida",
    "version nr",
    "utfärdare",
    "datum",
    "kursdokumentation",
)

LEAD_PATTERNS = [
    (re.compile(r"\bvad är\b", re.I), "Materialet beskriver detta som"),
    (re.compile(r"\bvad innebär\b", re.I), "Materialet beskriver detta som"),
    (re.compile(r"\bvad är syftet\b", re.I), "Syftet beskrivs i materialet som att"),
    (re.compile(r"\bvilka\b", re.I), "Materialet lyfter särskilt fram att"),
    (re.compile(r"\bhur\b", re.I), "Materialet visar att"),
    (re.compile(r"\bnär\b", re.I), "Materialet anger att"),
]


def build_extractive_reasoning(query: str, chunks: list, max_points: int = 3) -> str:
    """
    Bygger ett kort resonemang i egna ord utifrån toppchunkar.
    Om underlaget är för brusigt returneras ett försiktigt fallback-svar.
    """
    intent = classify_query_intent(query)

    if intent in {"overview_list", "list"}:
        list_reasoning = _build_list_reasoning(query, chunks)
        if list_reasoning:
            return list_reasoning

    evidence = _collect_evidence(query, chunks, limit=max_points)
    if not evidence:
        return (
            "Det finns inte tillräckligt tydligt underlag i de högst rankade källutdragen "
            "för att formulera ett pålitligt resonemang."
        )

    intro = _build_intro(query, chunks)
    points = " ".join(_rewrite_point(text) for text in evidence)
    closing = _build_closing(chunks)
    return " ".join(part for part in [intro, points, closing] if part).strip()


def _build_list_reasoning(query: str, chunks: list) -> str:
    query_lower = query.lower()

    if "etapp" in query_lower:
        stages = _extract_stage_names(chunks)
        if stages:
            return (
                "Materialet visar att införandet delas in i flera etapper. "
                + "De etapper som tydligast framgår är "
                + _join_list(stages)
                + ". "
                + _build_closing(chunks)
            ).strip()

    if "kompetens" in query_lower:
        competencies = _extract_competency_groups(chunks)
        if competencies:
            return (
                "Materialet visar att ett lyckat systeminförande kräver flera olika kompetenser. "
                + "De kompetensområden som tydligast lyfts fram är "
                + _join_list(competencies)
                + ". "
                + _build_closing(chunks)
            ).strip()

    if "kravområd" in query_lower:
        areas = _extract_requirement_areas(_expand_same_source_sections(chunks, "checklista_inforandekrav.pdf"))
        if areas:
            return (
                "Materialet visar att införandekrav delas in i flera kravområden som tillsammans "
                "täcker hela införandet. Exempel på sådana områden är "
                + _join_list(areas[:8])
                + ". "
                + _build_closing(chunks)
            ).strip()

    titles = _extract_section_titles(chunks)
    if titles:
        return (
            "Materialet visar att frågan besvaras genom flera återkommande områden eller delar. "
            + "De som framträder tydligast är "
            + _join_list(titles[:6])
            + ". "
            + _build_closing(chunks)
        ).strip()

    return ""


def _expand_same_source_sections(chunks: list, source_name: str) -> list:
    if not any((chunk.get("source") or "").lower() == source_name for chunk in chunks):
        return chunks

    all_chunks = load_chunks()
    expanded = [chunk for chunk in all_chunks if (chunk.get("source") or "").lower() == source_name]
    return expanded or chunks


def _collect_evidence(query: str, chunks: list, limit: int) -> list[str]:
    query_terms = _terms(query)
    candidates = []
    seen = set()

    for chunk in chunks:
        for sentence in _split_sentences(chunk.get("text", "")):
            cleaned = _clean_sentence(sentence)
            if not cleaned or cleaned in seen:
                continue

            score = _score_candidate(cleaned, query_terms, chunk)
            if score <= 0:
                continue

            candidates.append((score, cleaned))
            seen.add(cleaned)

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [text for _, text in candidates[:limit]]

    quality_hits = sum(1 for text in selected if _looks_human_readable(text))
    if quality_hits < max(1, min(limit, len(selected))):
        return []

    return selected


def _build_intro(query: str, chunks: list) -> str:
    lead = "Materialet visar att"
    for pattern, replacement in LEAD_PATTERNS:
        if pattern.search(query):
            lead = replacement
            break

    title = _best_title(chunks)
    if title:
        return f"{lead} detta framgår framför allt av avsnittet \"{title}\"."
    return f"{lead}"


def _build_closing(chunks: list) -> str:
    pages = sorted({page for chunk in chunks for page in (chunk.get("pages") or [])})
    if not pages:
        return ""
    if len(pages) == 1:
        return f"Bedömningen bygger på underlag från sida {pages[0]}."
    return f"Bedömningen bygger på underlag från sidorna {pages[0]}-{pages[-1]}."


def _extract_stage_names(chunks: list) -> list[str]:
    stages = []
    seen = set()

    for chunk in chunks:
        text = chunk.get("text", "")
        for match in re.finditer(r"Etapp\s+\d+\s*[–-]\s*([^\n.]+)", text, flags=re.I):
            stage = match.group(1).strip(" .")
            stage = re.sub(r"\s+", " ", stage)
            if len(stage) < 3:
                continue
            key = stage.lower()
            if key not in seen:
                stages.append(stage.lower())
                seen.add(key)

    return stages[:6]


def _extract_competency_groups(chunks: list) -> list[str]:
    groups = []
    seen = set()

    for chunk in chunks:
        text = " ".join(chunk.get("text", "").split())
        lowered = text.lower()

        if "system-, verksamhets-" in lowered or "system-, verksamhets- samt teknik-" in lowered:
            for item in ["systemkompetens", "verksamhetskompetens", "teknik- och driftkompetens"]:
                if item not in seen:
                    groups.append(item)
                    seen.add(item)

        if "för system krävs" in lowered and "systemkompetens" not in seen:
            groups.append("systemkompetens")
            seen.add("systemkompetens")
        if "för verksamheten krävs" in lowered and "verksamhetskompetens" not in seen:
            groups.append("verksamhetskompetens")
            seen.add("verksamhetskompetens")
        if "för teknik och drift krävs" in lowered and "teknik- och driftkompetens" not in seen:
            groups.append("teknik- och driftkompetens")
            seen.add("teknik- och driftkompetens")

    return groups[:6]


def _extract_requirement_areas(chunks: list) -> list[str]:
    areas = []
    seen = set()

    for chunk in chunks:
        title = (chunk.get("title") or "").strip()
        section = chunk.get("section", "")
        if section.startswith("4."):
            title = re.sub(r"^\d+(\.\d+)*\s+", "", title).strip()
            key = title.lower()
            if title and key not in seen:
                areas.append(title.lower())
                seen.add(key)

    return areas


def _extract_section_titles(chunks: list) -> list[str]:
    titles = []
    seen = set()
    for chunk in chunks:
        title = re.sub(r"^\d+(\.\d+)*\s+", "", (chunk.get("title") or "")).strip()
        if not title:
            continue
        key = title.lower()
        if key not in seen and not _has_ocr_noise(title):
            titles.append(title.lower())
            seen.add(key)
    return titles


def _rewrite_point(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    replacements = [
        (r"\bBilden visar\b", "Det beskrivs att"),
        (r"\bBilden kan ses som\b", "Det framgår också att"),
        (r"\bBeskrivning av området\b", "Området beskrivs så att"),
        (r"\bFrågor\b", "Det betonas också att"),
        (r"\bDet är viktigt att\b", "Materialet betonar att"),
        (r"\bEn förutsättning för\b", "En viktig förutsättning är att"),
        (r"\bSystemet testat att det fungerar\b", "Målet är att säkerställa att systemet fungerar"),
        (r"\bUtbilda personal\b", "Det innebär också att utbilda personal"),
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)

    if not text.endswith("."):
        text += "."

    text = text[0].upper() + text[1:]
    return text


def _score_candidate(sentence: str, query_terms: set[str], chunk: dict) -> int:
    sentence_terms = _terms(sentence)
    overlap = len(query_terms & sentence_terms)
    if overlap == 0:
        return 0

    score = overlap * 3
    score += len(query_terms & _terms(chunk.get("title", ""))) * 2

    positive_markers = [
        "innebär", "syfte", "omfattar", "beskriv", "viktigt", "förutsättning",
        "mål", "säkerställ", "verifiera", "utbild", "stöd", "plan", "krav"
    ]
    if any(marker in sentence.lower() for marker in positive_markers):
        score += 2

    if _looks_human_readable(sentence):
        score += 2

    return score


def _split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []

    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _is_metadata_line(line):
            continue
        parts.extend(re.split(r"(?<=[.!?])\s+|(?<=:)\s+", line))

    return [part.strip(" -") for part in parts if part.strip(" -")]


def _clean_sentence(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence = re.sub(r"\s*[_]{2,}\s*", " ", sentence)
    sentence = re.sub(r"\b\d+\(\d+\)\b", "", sentence)
    sentence = sentence.strip(" -")

    if len(sentence) < 45 or len(sentence) > 260:
        return ""
    if _is_metadata_line(sentence):
        return ""
    if _has_ocr_noise(sentence):
        return ""

    return sentence


def _best_title(chunks: list) -> str:
    for chunk in chunks:
        title = (chunk.get("title") or "").strip()
        if not title:
            continue
        if any(pattern in title.lower() for pattern in BAD_TITLE_PATTERNS):
            continue
        if _has_ocr_noise(title):
            continue
        return title
    return ""


def _looks_human_readable(text: str) -> bool:
    words = re.findall(r"\w+", text)
    if len(words) < 7:
        return False
    if sum(1 for word in words if len(word) > 18) >= 2:
        return False
    if text.count('"') > 2:
        return False
    return True


def _join_list(items: list[str]) -> str:
    cleaned = [item.strip() for item in items if item.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:-1]) + f" och {cleaned[-1]}"


def _is_metadata_line(text: str) -> bool:
    lowered = text.lower()
    if any(pattern in lowered for pattern in BAD_TITLE_PATTERNS):
        return True
    if re.fullmatch(r"[\d\W_]+", text):
        return True
    if re.fullmatch(r"[A-ZÅÄÖa-zåäö\s]{1,20}", text) and len(text.split()) <= 3:
        return True
    return False


def _has_ocr_noise(text: str) -> bool:
    if "___" in text:
        return True
    if re.search(r"\b[a-zåäö]{1,2}\s+[A-ZÅÄÖa-zåäö]{1,2}\b", text):
        return True
    if re.search(r"(.)\1{4,}", text):
        return True
    if text.count("•") > 2 or text.count("") > 1:
        return True
    return False


def _terms(text: str) -> set[str]:
    return {
        term for term in re.findall(r"\w+", text.lower())
        if term not in STOPWORDS and len(term) > 2
    }
