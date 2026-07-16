from __future__ import annotations

from typing import Any, Iterable
from urllib.parse import urlparse

ALLOWED_SOURCE_TYPES = {"pdf", "web"}
ALLOWED_WEB_HOSTS = {"systeminforande.se", "www.systeminforande.se"}
INSUFFICIENT_EVIDENCE_ANSWER = (
    "Jag kan inte verifiera svaret utifrån de tillgängliga källorna "
    "(uppladdade PDF:er och hemsidans innehåll)."
)


def is_allowed_chunk_source(chunk: dict[str, Any]) -> bool:
    source_type = (chunk.get("source_type") or "").strip().lower()
    source = (chunk.get("source") or "").strip()

    if source_type not in ALLOWED_SOURCE_TYPES or not source:
        return False

    if source_type == "pdf":
        return source.lower().endswith(".pdf")

    parsed = urlparse(source)
    hostname = (parsed.hostname or "").lower()
    scheme = (parsed.scheme or "").lower()
    return scheme in {"http", "https"} and hostname in ALLOWED_WEB_HOSTS


def filter_allowed_chunks(chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [chunk for chunk in chunks if is_allowed_chunk_source(chunk)]


def filter_allowed_results(results: Iterable[tuple[float, dict[str, Any]]]) -> list[tuple[float, dict[str, Any]]]:
    return [(score, chunk) for score, chunk in results if is_allowed_chunk_source(chunk)]


def grounded_answer_or_fallback(structured_answer: str | None) -> str:
    answer = (structured_answer or "").strip()
    if answer:
        return answer
    return INSUFFICIENT_EVIDENCE_ANSWER
