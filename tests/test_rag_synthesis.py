from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.grounding import INSUFFICIENT_EVIDENCE_ANSWER
from rag.synthesis import (
    DEFAULT_SYNTHESIS_MODEL,
    SUPPORTED_EXPERIMENT_MODELS,
    build_final_grounded_answer,
    resolve_synthesis_settings,
)


CHUNKS = [
    {
        "source": "Mallar_utbildningsstrategi.pdf",
        "source_type": "pdf",
        "title": "Utbildningsstrategi",
        "text": (
            "Utbildningsstrategin ska beskriva syftet med dokumentet, utbildningsstrategins huvudresultat, "
            "målgrupper, utbildningarnas innehåll, utbildningsmål och ett grovt uppskattat utbildningsbehov. "
            "Den ska också beskriva bakgrund, förutsättningar och hur utbildningen ska genomföras."
        ),
        "pages": [1],
    }
]


def test_synthesis_stage_is_off_by_default():
    result = build_final_grounded_answer(
        "Vad ska en utbildningsstrategi innehålla",
        CHUNKS,
        enable_synthesis=False,
    )

    assert result["synthesis_enabled"] is False
    assert result["synthesis_used"] is False
    assert result["final_answer"] == result["extractive_answer"]
    assert "utbildningsstrategi" in result["final_answer"].lower()


@pytest.mark.parametrize("model_name", ["openai/gpt-oss-120b", "zai-org/GLM-5.2"])
def test_synthesis_stage_can_replace_extractive_answer_when_feature_flag_is_enabled(model_name: str):
    def llm_rewrite(prompt: str, model: str | None = None) -> str:
        assert "Utbildningsstrategi" in prompt
        assert model == model_name
        return (
            "Materialet visar att utbildningsstrategin bör beskriva syftet och huvudresultatet, "
            "vilka målgrupper som omfattas, vilket utbildningsinnehåll och vilka utbildningsmål som gäller, "
            "samt ett grovt uppskattat utbildningsbehov. Den bör också ta upp bakgrund, förutsättningar "
            "och hur utbildningen ska genomföras."
        )

    result = build_final_grounded_answer(
        "Vad ska en utbildningsstrategi innehålla",
        CHUNKS,
        enable_synthesis=True,
        llm_model=model_name,
        llm_rewrite=llm_rewrite,
    )

    assert result["synthesis_enabled"] is True
    assert result["synthesis_used"] is True
    assert result["llm_model"] == model_name
    assert result["final_answer"] != result["extractive_answer"]
    assert "målgrupper" in result["final_answer"].lower()
    assert "utbildningsbehov" in result["final_answer"].lower()


def test_synthesis_stage_falls_back_to_extractive_answer_when_rewrite_is_unsupported():
    def llm_rewrite(prompt: str, model: str | None = None) -> str:
        return "Projektet bör ha en styrgrupp och veckovisa statusmöten för att lyckas."

    result = build_final_grounded_answer(
        "Vad ska en utbildningsstrategi innehålla",
        CHUNKS,
        enable_synthesis=True,
        llm_rewrite=llm_rewrite,
    )

    assert result["synthesis_enabled"] is True
    assert result["synthesis_used"] is False
    assert result["final_answer"] == result["extractive_answer"]
    assert result["llm_status"] == "fallback_to_extractive_due_to_grounding_check"


def test_synthesis_stage_does_not_call_llm_when_extractive_path_has_insufficient_evidence():
    called = False

    def llm_rewrite(prompt: str, model: str | None = None) -> str:
        nonlocal called
        called = True
        return "Det här borde aldrig användas."

    result = build_final_grounded_answer(
        "Vilken färg har månen i projektmodellen?",
        [],
        enable_synthesis=True,
        llm_rewrite=llm_rewrite,
    )

    assert result["extractive_answer"] == INSUFFICIENT_EVIDENCE_ANSWER
    assert result["final_answer"] == INSUFFICIENT_EVIDENCE_ANSWER
    assert result["synthesis_used"] is False
    assert called is False


def test_resolve_synthesis_settings_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SYSTEMINFORANDE_ENABLE_LLM_SYNTHESIS", raising=False)
    monkeypatch.delenv("SYSTEMINFORANDE_LLM_SYNTHESIS_MODEL", raising=False)

    settings = resolve_synthesis_settings()

    assert settings["enabled"] is False
    assert settings["model"] == DEFAULT_SYNTHESIS_MODEL
    assert SUPPORTED_EXPERIMENT_MODELS == ("openai/gpt-oss-120b", "zai-org/GLM-5.2")


def test_resolve_synthesis_settings_supports_environment_default_model(monkeypatch):
    monkeypatch.setenv("SYSTEMINFORANDE_ENABLE_LLM_SYNTHESIS", "true")
    monkeypatch.setenv("SYSTEMINFORANDE_LLM_SYNTHESIS_MODEL", "zai-org/GLM-5.2")

    settings = resolve_synthesis_settings()

    assert settings["enabled"] is True
    assert settings["model"] == "zai-org/GLM-5.2"
    assert settings["enabled_source"] == "environment"


def test_resolve_synthesis_settings_supports_per_request_override(monkeypatch):
    monkeypatch.setenv("SYSTEMINFORANDE_ENABLE_LLM_SYNTHESIS", "false")
    monkeypatch.setenv("SYSTEMINFORANDE_LLM_SYNTHESIS_MODEL", "openai/gpt-oss-120b")

    settings = resolve_synthesis_settings(
        enable_synthesis=True,
        llm_model="zai-org/GLM-5.2",
        default_model="openai/gpt-oss-120b",
    )

    assert settings["enabled"] is True
    assert settings["model"] == "zai-org/GLM-5.2"
    assert settings["requested_model"] == "zai-org/GLM-5.2"
    assert settings["enabled_source"] == "override"
