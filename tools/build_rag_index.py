import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.ingest import ingest_pdfs_and_web, save_chunks


DATA_DIR = "rag/data"


def main():
    print("🔄 Startar RAG-ingest")
    start_time = time.perf_counter()
    chunks = ingest_pdfs_and_web()
    save_chunks(chunks, out_dir=DATA_DIR)
    elapsed = time.perf_counter() - start_time
    print(f"✅ Ingest klar – {len(chunks)} chunkar skapade")
    print(f"⏱️ Ingest-tid: {elapsed:.2f} sekunder")


if __name__ == "__main__":
    main()
