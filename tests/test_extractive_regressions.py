from rag.extractive import build_extractive_reasoning
from rag.search import search
from rag.source_links import build_sources_md


def _answer_for(question: str) -> str:
    results = search(question, top_k=5)
    chunks = [chunk for _, chunk in results]
    return build_extractive_reasoning(question, chunks).lower()


def test_requirement_checklist_question_answers_yes_from_local_chunks():
    answer = _answer_for("Finns det en checklista för införandekrav")

    assert "ja" in answer
    assert "checklista" in answer
    assert "bedömningen bygger" not in answer
    assert "s. " not in answer


def test_work_area_definition_mentions_grouping_and_examples():
    answer = _answer_for("Vad är ett arbetsområde?")

    assert answer.startswith("ett arbetsområde är")
    assert "avgränsat delområde" in answer
    assert "aktiviteter" in answer
    assert any(keyword in answer for keyword in ["acceptanstest", "utbildning", "it-miljoer", "driftsattning"])


def test_project_library_definition_uses_project_library_section():
    answer = _answer_for("Vad är ett projektbibliotek")

    assert answer.startswith("ett projektbibliotek är")
    assert "arbetsresultat" in answer
    assert "filer" in answer
    assert "dokument" in answer


def test_project_organization_definition_uses_project_organization_section():
    answer = _answer_for("Vad är en projektorganisation")

    assert answer.startswith("en projektorganisation är")
    assert "vid sidan av linjeorganisationen" in answer
    assert any(keyword in answer for keyword in ["styrgrupp", "projektledning", "delprojekt"])


def test_acceptance_test_process_answer_mentions_key_phases():
    answer = _answer_for("Hur används acceptanstest i införandet?")

    assert "processen" in answer
    assert "planeringen" in answer
    assert "förberedelserna" in answer
    assert "genomförandet" in answer
    assert "uppföljningen" in answer


def test_decision_points_purpose_answer_mentions_control_and_steering():
    answer = _answer_for("Vad är syftet med beslutspunkter?")

    assert "kontrollera" in answer
    assert "styra" in answer
    assert "rätt organisatorisk nivå" in answer


def test_go_live_timing_answer_is_honest_about_missing_exact_date():
    answer = _answer_for("När ska driftsättning ske?")

    assert "inte ett exakt datum" in answer
    assert "fastställas" in answer
    assert "driftsättningen" in answer


def test_delivery_approval_timing_answer_mentions_planned_but_not_exact_time():
    answer = _answer_for("När fattas beslut om leveransgodkännande?")

    assert "inte någon konkret tidpunkt" in answer
    assert "kriterier" in answer
    assert "beslutsfattare" in answer


def test_stage_question_is_honest_about_missing_full_stage_list():
    answer = _answer_for("Vilka etapper finns det")

    assert "etappindelat" in answer
    assert "inte" in answer
    assert "räcker" in answer


def test_requirement_areas_question_lists_multiple_requirement_areas():
    answer = _answer_for("Vilka kravområden ingår i införandekrav?")

    assert "införandekrav delas in i flera kravområden" in answer
    assert any(keyword in answer for keyword in ["övergripande krav", "arbetsrutiner", "systemsamband"])
    assert "acceptanstest" in answer


def test_work_areas_are_used_for_planning_structure_and_followup():
    answer = _answer_for("Hur används arbetsområden i planeringen?")

    assert any(keyword in answer for keyword in ["planeringen", "strukturera", "ansvar", "uppföljning"])


def test_performance_answer_uses_flowing_body_without_inline_source_references():
    question = "Hur kolla att svarstider och bearbetningstider uppfyller ställda krav?"
    results = search(question, top_k=5)
    chunks = [chunk for _, chunk in results]

    answer = build_extractive_reasoning(question, chunks).lower()
    sources_md = build_sources_md(results)

    assert answer.startswith("för att kontrollera att svarstider och bearbetningstider")
    assert "acceptanstest" in answer
    assert "leveransgodkännande" in answer
    assert "bedömningen bygger" not in answer
    assert ".pdf" not in answer
    assert "s. " not in answer
    assert "### Källor" in sources_md
    assert ".pdf" in sources_md
