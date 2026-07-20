from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.reasoning import extract_hf_usage
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
        assert "4 till 8 meningar" in prompt
        assert "Svara självständigt" in prompt
        assert "Använd inte ett förbyggt bassvar som struktur" in prompt
        assert "Fallback-svar om LLM-svaret inte blir källbundet" in prompt
        assert "Bassvar:" not in prompt
        return (
            "En utbildningsstrategi behöver göra det tydligt varför dokumentet finns och vilket huvudresultat "
            "strategin ska leda till. Den ska också beskriva vilka målgrupper som omfattas, vilket innehåll "
            "utbildningarna har och vilka mål utbildningen ska uppfylla. Utifrån underlaget behöver strategin dessutom "
            "ta upp ett grovt uppskattat utbildningsbehov, bakgrund, förutsättningar och hur utbildningen ska genomföras."
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


def test_synthesis_stage_rejects_formulaic_template_phrases():
    def llm_rewrite(prompt: str, model: str | None = None) -> str:
        return (
            "Materialet visar att utbildningsstrategin bör beskriva syfte, huvudresultat, målgrupper, "
            "utbildningarnas innehåll, utbildningsmål och utbildningsbehov. Den bör också beskriva "
            "bakgrund, förutsättningar och genomförande."
        )

    result = build_final_grounded_answer(
        "Vad ska en utbildningsstrategi innehålla",
        CHUNKS,
        enable_synthesis=True,
        llm_rewrite=llm_rewrite,
    )

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


def test_resolve_synthesis_settings_uses_llm_synthesis_by_default(monkeypatch):
    monkeypatch.delenv("SYSTEMINFORANDE_ENABLE_LLM_SYNTHESIS", raising=False)
    monkeypatch.delenv("SYSTEMINFORANDE_LLM_SYNTHESIS_MODEL", raising=False)

    settings = resolve_synthesis_settings()

    assert settings["enabled"] is True
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


def test_extract_hf_usage_reads_huggingface_usage_fields_defensively():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens="12",
            completion_tokens=5,
            total_tokens=17,
        )
    )

    assert extract_hf_usage(response) == {
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
    }


def test_extract_hf_usage_tolerates_missing_usage():
    assert extract_hf_usage(SimpleNamespace()) == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def test_huggingface_provider_call_returns_usage_when_present(monkeypatch):
    from llm import reasoning

    class FakeClient:
        def chat_completion(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message={"content": "Källbundet svar"})],
                usage=SimpleNamespace(
                    prompt_tokens=13,
                    completion_tokens=8,
                    total_tokens=21,
                ),
            )

    monkeypatch.setattr(reasoning, "get_llm_client", lambda model=None: FakeClient())

    result = reasoning.generate_reasoning_from_prompt_with_usage(
        "Svara bara utifrån källmaterialet.",
        model="Qwen/Qwen3-32B",
    )

    assert result.text == "Källbundet svar"
    assert result.usage == {
        "prompt_tokens": 13,
        "completion_tokens": 8,
        "total_tokens": 21,
    }


def test_huggingface_provider_call_does_not_fail_when_usage_is_missing(monkeypatch):
    from llm import reasoning

    class FakeClient:
        def chat_completion(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message={"content": "Svar utan usage-metadata"})]
            )

    monkeypatch.setattr(reasoning, "get_llm_client", lambda model=None: FakeClient())

    result = reasoning.generate_reasoning_from_prompt_with_usage(
        "Svara bara utifrån källmaterialet.",
        model="Qwen/Qwen3-32B",
    )

    assert result.text == "Svar utan usage-metadata"
    assert result.usage == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
