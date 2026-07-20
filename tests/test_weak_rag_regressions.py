from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.extractive import build_extractive_reasoning
from rag.grounding import grounded_answer_or_fallback
from rag.search import search

FALLBACK_PREFIX = "Det finns inte tillräckligt tydligt underlag"


def _answer_for(question: str) -> str:
    results = search(question, top_k=5)
    chunks = [chunk for _, chunk in results]
    return grounded_answer_or_fallback(build_extractive_reasoning(question, chunks))


def _sources_for(question: str) -> list[str]:
    return [chunk["source"] for _, chunk in search(question, top_k=5)]


def test_education_strategy_answer_should_stay_on_training_strategy_topic():
    answer = _answer_for("Vad ska en utbildningsstrategi innehålla")
    answer_lower = answer.lower()
    sources = _sources_for("Vad ska en utbildningsstrategi innehålla")

    assert sources[0] == "Mallar_utbildningsstrategi.pdf"
    assert "utbild" in answer_lower
    assert "testregister" not in answer_lower
    assert "acceptanstest_testplan" not in answer_lower


def test_compound_business_and_conversion_question_should_cover_both_subquestions():
    answer = _answer_for(
        "Hur kolla att systemet fungerar i verksamheten?Hur många konverteringar behöver genomföras?"
    )
    answer_lower = answer.lower()

    assert "verksam" in answer_lower
    assert "konverter" in answer_lower


def test_four_part_operational_question_should_cover_all_subquestions():
    answer = _answer_for(
        "Hur kolla att säkerheten är tillräcklig?\n\n"
        "Hur ska systemet driftsättas?\n\n"
        "Hur många IT-miljöer ska finnas?\n\n"
        "Hur ska överlämning till drift och förvaltning gå till?"
    )
    answer_lower = answer.lower()

    assert "säker" in answer_lower
    assert "driftsätt" in answer_lower
    assert "it-miljö" in answer_lower or "it-miljo" in answer_lower
    assert "förvalt" in answer_lower


def test_education_strategy_existence_question_should_not_fallback_when_pdf_exists():
    answer = _answer_for("Finns det en utbildningsstrategi")
    answer_lower = answer.lower()
    sources = _sources_for("Finns det en utbildningsstrategi")

    assert sources
    assert all(source == "Mallar_utbildningsstrategi.pdf" for source in sources)
    assert not answer.startswith(FALLBACK_PREFIX)
    assert "ja" in answer_lower or "finns" in answer_lower


def test_system_setup_question_should_not_be_routed_to_training_content():
    answer = _answer_for("Hur ska systemet/applikationen sättas upp?")
    answer_lower = answer.lower()

    assert any(keyword in answer_lower for keyword in ["sätta upp", "konfigur", "install", "driftsätt"])
    assert "utbildning" not in answer_lower
    assert "acceptanstest" not in answer_lower


def test_system_dependencies_answer_should_be_clean_narrative_not_fragment_dump():
    answer = _answer_for("Fungerar sambanden med omgivande system?")
    answer_lower = answer.lower()

    assert "omgivande system" in answer_lower
    assert "finns det en beskrivning som visar samband med omgivande system" not in answer_lower
    assert "[ta fram en beskrivning" not in answer_lower
    assert "?." not in answer


def test_response_time_answer_should_not_return_broken_fragment_dump():
    answer = _answer_for("Hur kolla att svarstider och bearbetningstider uppfyller ställda krav?")
    answer_lower = answer.lower()

    assert "svarstid" in answer_lower or "bearbetningstid" in answer_lower
    assert "•" not in answer
    assert "tillfredställande i." not in answer_lower
    assert "tekniska." not in answer_lower


def test_system_implementation_obstacles_answer_should_not_use_template_intro():
    answer = _answer_for("Vilka hinder finns i systeminförande?")
    answer_lower = answer.lower()

    assert "systeminförande" in answer_lower
    assert "frågan verkar beröra" not in answer_lower
    assert "de som framträder tydligast här" not in answer_lower
