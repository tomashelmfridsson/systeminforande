# rag/search.py

import json
import os
import math
import re


DATA_DIR = "rag/data"
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")


# =========================
# LADDNING
# =========================

def load_chunks():
    if not os.path.exists(CHUNKS_FILE):
        return []

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return json.load(f)

# =========================
# ENKEL SEMANTISK SCORING
# =========================
def score_chunk(query: str, chunk: dict) -> float:
    query = query.lower()

    query_terms = set(re.findall(r"\w+", query))
    text = (chunk.get("title", "") + " " + chunk.get("text", "")).lower()
    text_terms = set(re.findall(r"\w+", text))

    overlap = len(query_terms & text_terms)

    boost = 0

    # 🔹 Definition-fråga
    if query.startswith("vad är"):
        if any(k in chunk["title"].lower() for k in ["inledning", "syfte", "omfattning"]):
            boost += 3

    return overlap + boost

# =========================
# SEARCH
# =========================

def search(query: str, top_k: int = 5):
    chunks = load_chunks()   # <-- LADDAS HÄR
    print(f"🔍 Search: laddade {len(chunks)} chunkar")
    if not chunks:
        return []

    scored = []
    for chunk in chunks:
        score = score_chunk(query, chunk)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]