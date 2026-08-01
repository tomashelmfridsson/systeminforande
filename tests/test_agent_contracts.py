from rag.agent_contracts import (
    normalize_answer_draft,
    normalize_correction,
    normalize_retrieval_rewrite,
    normalize_review,
)


def test_agent1_contract_restores_original_question_and_query():
    result = normalize_retrieval_rewrite({"status": "recovered"}, "Vilken fråga?")
    assert result["original_question"] == "Vilken fråga?"
    assert result["retrieval_queries"][0]["query"] == "Vilken fråga?"


def test_agent2_contract_keeps_draft_uncertainty():
    result = normalize_answer_draft({"status": "ok", "answer": "Ett utkast."}, "Vad?")
    assert result["original_question"] == "Vad?"
    assert result["evidence_ids_used"] == []


def test_agent3_invalid_status_becomes_review_unavailable():
    result = normalize_review({"status": "invalid_json"}, "Ett utkast.")
    assert result["status"] == "unavailable"
    assert result["draft_answer"] == "Ett utkast."


def test_agent4_contract_keeps_question_and_evidence_ids():
    result = normalize_correction({"status": "ok"}, "Vad?")
    assert result["original_question"] == "Vad?"
    assert result["evidence_ids_used"] == []
