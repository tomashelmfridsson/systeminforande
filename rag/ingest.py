# rag/ingest.py

import os
import re
import json
import time
import fitz  # PyMuPDF
from urllib.parse import urlparse


# =========================
# KONFIG
# =========================

PDF_DIR = "docs/pdfs"
WEB_SOURCE_FILE = "rag/web/source.txt"


# =========================
# REGEX FÖR RUBRIKER
# =========================

SECTION_RE = re.compile(r"^(\d+(\.\d+)+)\s+(.+)$")
ALL_CAPS_RE = re.compile(r"^[A-ZÅÄÖ][A-ZÅÄÖ\s\-]{5,}$")

def ingest_all():
    """
    Publikt API för app.py.
    Ingestar både PDF och web-källor.
    """
    return ingest_pdfs_and_web()

# =========================
# FILTER
# =========================

def is_useful_chunk(text: str) -> bool:
    text = text.strip()

    if len(text) < 80:
        return False

    if text.count("\\") > 3:
        return False

    if re.fullmatch(r"[0-9\s\-/.:()]+", text):
        return False

    if not re.search(r"\b(är|ska|syfte|beskriv|genomför|använd)\b", text.lower()):
        return False

    return True


# =========================
# PDF → TEXT
# =========================

def extract_pdf_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        pages.append((i + 1, page.get_text("text")))
    return pages


# =========================
# WEB → TEXT
# =========================

def extract_web_page(url: str):
    import requests
    from bs4 import BeautifulSoup

    r = requests.get(url, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # ta bort skräp
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text("\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return [(1, "\n".join(lines))]


# =========================
# CHUNK PER RUBRIK
# =========================

def chunk_by_headings(pages, source_name, source_type, source_ref):
    chunks = []

    current = {
        "title": None,
        "section": None,
        "content": [],
        "pages": []
    }

    def flush():
        if current["title"] and current["content"]:
            text = "\n".join(current["content"]).strip()
            if not is_useful_chunk(text):
                return

            chunks.append({
                "id": f"{source_name}_{current['section']}",
                "title": current["title"],
                "section": current["section"],
                "text": text,
                "pages": sorted(set(current["pages"])),
                "source": source_ref,
                "source_type": source_type,
            })

    for page_no, text in pages:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            m = SECTION_RE.match(line)
            is_caps = ALL_CAPS_RE.match(line)

            if m or is_caps:
                flush()

                if m:
                    section = m.group(1)
                    title = f"{section} {m.group(3)}"
                else:
                    section = line
                    title = line

                current = {
                    "title": title,
                    "section": section,
                    "content": [],
                    "pages": [page_no]
                }
                continue

            current["content"].append(line)
            current["pages"].append(page_no)

    flush()
    return chunks


# =========================
# INGEST PDF + WEB
# =========================

def ingest_pdfs_and_web():
    all_chunks = []
    start = time.perf_counter()

    # -------- PDF --------
    for file in sorted(os.listdir(PDF_DIR)):
        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(PDF_DIR, file)
        name = os.path.splitext(file)[0]

        pages = extract_pdf_pages(path)
        chunks = chunk_by_headings(
            pages,
            source_name=name,
            source_type="pdf",
            source_ref=file
        )
        all_chunks.extend(chunks)

    # -------- WEB --------
    if os.path.exists(WEB_SOURCE_FILE):
        with open(WEB_SOURCE_FILE, encoding="utf-8") as f:
            urls = [l.strip() for l in f if l.strip()]

        for url in urls:
            domain = urlparse(url).netloc.replace(".", "_")

            try:
                pages = extract_web_page(url)
            except ImportError:
                print(f"⚠️ Hoppar över web-källa utan parserstöd: {url}")
                continue
            except Exception as exc:
                print(f"⚠️ Hoppar över web-källa efter fel: {url} ({exc})")
                continue

            chunks = chunk_by_headings(
                pages,
                source_name=domain,
                source_type="web",
                source_ref=url
            )
            all_chunks.extend(chunks)

    elapsed = time.perf_counter() - start
    print(f"⏱️ Chunking klar på {elapsed:.2f} sekunder")
    print(f"📦 Totalt {len(all_chunks)} chunkar")

    return all_chunks


# =========================
# SPARA
# =========================

def save_chunks(chunks, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "chunks.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"💾 Sparade {len(chunks)} chunkar → {out_path}")
