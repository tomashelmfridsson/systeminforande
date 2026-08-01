"""Shared, runtime-light contracts for the Agentic RAG pipeline.

These contracts describe the hand-off between agents. They deliberately do not
decide answer quality; Agent 2 drafts, Agent 3 reviews, and Agent 4 revises.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


Agent1Status = Literal["ok", "recovered", "fallback"]
Agent2Status = Literal["ok", "fallback"]
Agent3Status = Literal["approved", "revision", "rejected", "unavailable", "not_run"]
Agent4Status = Literal["ok", "fallback", "not_run"]


class RetrievalRewriteContract(TypedDict, total=False):
    status: Agent1Status
    original_question: str
    retrieval_queries: list[dict[str, Any]]
    semantic_terms: list[dict[str, Any]]
    negative_constraints: list[str]
    debug: dict[str, Any]


class AnswerDraftContract(TypedDict, total=False):
    status: Agent2Status
    original_question: str
    answer: str
    answer_scope: str
    evidence_used: list[dict[str, Any]]
    evidence_ids_used: list[str]
    unsupported_or_uncertain: list[str]
    debug: dict[str, Any]


class ReviewContract(TypedDict, total=False):
    status: Agent3Status
    draft_answer: str
    reason: str
    revision: str
    evidence_ids_used: list[str]
    debug: dict[str, Any]


class CorrectionContract(TypedDict, total=False):
    status: Agent4Status
    original_question: str
    answer: str
    evidence_ids_used: list[str]
    debug: dict[str, Any]


def normalize_retrieval_rewrite(result: dict[str, Any] | None, question: str) -> dict[str, Any]:
    """Normalize Agent 1 hand-off without judging retrieval quality."""
    normalized = dict(result or {})
    normalized["original_question"] = question
    if not isinstance(normalized.get("retrieval_queries"), list) or not normalized["retrieval_queries"]:
        normalized["retrieval_queries"] = [{"query": question, "purpose": "literal", "weight": 1.0}]
    return normalized


def normalize_answer_draft(result: dict[str, Any] | None, question: str) -> dict[str, Any]:
    """Normalize Agent 2 hand-off; never turn content uncertainty into rejection."""
    normalized = dict(result or {})
    normalized["original_question"] = question
    normalized.setdefault("evidence_ids_used", [])
    normalized.setdefault("unsupported_or_uncertain", [])
    return normalized


def normalize_review(result: dict[str, Any] | None, draft_answer: str) -> dict[str, Any]:
    """Normalize Agent 3 hand-off; format failures remain review-unavailable."""
    normalized = dict(result or {})
    normalized.setdefault("draft_answer", draft_answer)
    if normalized.get("status") not in {"approved", "revision", "rejected", "unavailable", "not_run"}:
        normalized["status"] = "unavailable"
    return normalized


def normalize_correction(result: dict[str, Any] | None, question: str) -> dict[str, Any]:
    normalized = dict(result or {})
    normalized["original_question"] = question
    normalized.setdefault("evidence_ids_used", [])
    return normalized
