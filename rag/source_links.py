from __future__ import annotations

from typing import Any
from urllib.parse import quote

GITHUB_PAGES_BASE_URL = "https://tomashelmfridsson.github.io/systeminforande"
GITHUB_PAGES_PDF_BASE_URL = f"{GITHUB_PAGES_BASE_URL}/pdfs"

_HOMEPAGE_LINKS = {
    "inforandekrav": {
        "label": "Införandekrav",
        "url": "https://www.systeminforande.se/infrandekrav-1",
    },
    "implementering": {
        "label": "Implementering",
        "url": "https://www.systeminforande.se/implementering2",
    },
    "arbetsmodell": {
        "label": "Arbetsmodell",
        "url": "https://www.systeminforande.se/arbetsmodell",
    },
    "verktyg": {
        "label": "Verktyg",
        "url": "https://www.systeminforande.se/verktyg",
    },
    "mallar": {
        "label": "Checklistor och mallar",
        "url": "https://www.systeminforande.se/checklistor-och-mallar-till-verktyget-1",
    },
}


def _ascii_fold(text: str) -> str:
    return (
        text.replace("å", "a")
        .replace("ä", "a")
        .replace("ö", "o")
        .replace("Å", "A")
        .replace("Ä", "A")
        .replace("Ö", "O")
    )


def _homepage_keys_for_source(chunk: dict[str, Any]) -> list[str]:
    source = (chunk.get("source") or "").strip().lower()
    title = _ascii_fold((chunk.get("title") or "").strip().lower())
    if not source:
        return []

    keys: list[str] = []

    if source.startswith("inforandekrav_"):
        keys.append("inforandekrav")
        if "checklista" in source or "kravmallen" in source:
            keys.append("mallar")

    if source.startswith("arbetsomraden_"):
        keys.extend(["implementering", "verktyg"])

    if source.startswith("verktyget_och_systeminforandet"):
        keys.append("arbetsmodell")

    if source.startswith("verktyget_projektstyrning") and (
        "projektstyrningsmodell" in title or "arbetsmodell" in title
    ):
        keys.append("arbetsmodell")

    if source.startswith("verktyget_"):
        keys.append("verktyg")

    if source.startswith("mallar_"):
        keys.append("mallar")

    # Preserve order while removing duplicates.
    deduped: list[str] = []
    seen = set()
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped



def homepage_links_for_chunk(chunk: dict[str, Any]) -> list[dict[str, str]]:
    links = []
    for key in _homepage_keys_for_source(chunk):
        link = _HOMEPAGE_LINKS.get(key)
        if link:
            links.append(link)
    return links



def collect_homepage_links(results) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for _, chunk in results:
        for link in homepage_links_for_chunk(chunk):
            url = link["url"]
            if url in seen_urls:
                continue
            links.append(link)
            seen_urls.add(url)

    return links



def format_source_link(chunk: dict[str, Any]) -> str:
    source = chunk.get("source", "Okänd källa")
    source_type = chunk.get("source_type")

    if source_type == "pdf":
        encoded_source = quote(source)
        return f"📄 [{source}]({GITHUB_PAGES_PDF_BASE_URL}/{encoded_source})"

    if source_type == "web":
        return f"🌐 [{source}]({source})"

    return source



def format_source_url(chunk: dict[str, Any]) -> str | None:
    source = chunk.get("source")
    source_type = chunk.get("source_type")

    if not source:
        return None

    if source_type == "pdf":
        return f"{GITHUB_PAGES_PDF_BASE_URL}/{quote(source)}"

    if source_type == "web":
        return source

    return None



def build_sources_md(results) -> str:
    used_sources = {}
    for _, chunk in results:
        used_sources[chunk["source"]] = chunk

    homepage_links = collect_homepage_links(results)

    if not used_sources and not homepage_links:
        return ""

    sections = ["\n\n---\n\n### Källor"]
    for chunk in used_sources.values():
        sections.append(f"- {format_source_link(chunk)}")

    if homepage_links:
        sections.append("\n### Relaterade hemsidor")
        for link in homepage_links:
            sections.append(f"- 🏠 [{link['label']}]({link['url']})")

    return "\n".join(sections)



def serialize_homepage_links(results) -> list[dict[str, str]]:
    return collect_homepage_links(results)
