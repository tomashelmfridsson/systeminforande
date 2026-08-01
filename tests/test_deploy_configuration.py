from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hugging_face_docker_defaults_to_agentic_rag():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "SYSTEMINFORANDE_ENABLE_AGENTIC_RAG=true" in dockerfile
    assert "SYSTEMINFORANDE_ENABLE_AGENTIC_RAG=false" not in dockerfile
