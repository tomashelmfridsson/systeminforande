from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.extractive import build_extractive_reasoning
from rag.grounding import grounded_answer_or_fallback
from rag.ingest import chunk_by_headings, is_noise_line
from rag.search import search
from rag.synthesis import build_final_grounded_answer, build_synthesis_prompt

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


def test_tomas_metadata_boilerplate_is_not_used_as_answer_content():
    chunks = [
        {
            "source": "Verktyget_projektstyrning.pdf",
            "source_type": "pdf",
            "title": "Projektstyrning",
            "text": (
                "Citrus Projektstyrning Verktyg för systeminförande Hans Johansson "
                "Version 1 © Citrus i Stockholm\n"
                "Planera införandet genom att beskriva mål, aktiviteter, tidplan, "
                "resurser och uppföljning."
            ),
            "pages": [1],
        }
    ]

    answer = build_extractive_reasoning("Hur planerar man implementation av system?", chunks)
    answer_lower = answer.lower()

    assert "mål" in answer_lower
    assert "aktiviteter" in answer_lower
    assert "hans johansson" not in answer_lower
    assert "©" not in answer
    assert "version 1" not in answer_lower
    assert "citrus projektstyrning" not in answer_lower


def test_tomas_figure_only_context_is_not_sufficient_evidence():
    chunks = [
        {
            "source": "Verktyget_och_systeminforandet.pdf",
            "source_type": "pdf",
            "title": "Systemfaser",
            "text": (
                "Nedanstående figur visar olika faser under ett systems livscykel "
                "med översiktlig beskrivning för varje fas samt exempel på frågor "
                "och arbetsuppgifter kring ett system under dess livslängd."
            ),
            "pages": [1],
        }
    ]

    answer = build_extractive_reasoning("Hur planerar man implementation av system?", chunks)

    assert answer.startswith(FALLBACK_PREFIX)
    assert "nedanstående figur" not in answer.lower()


def test_tomas_q01_planning_answer_stays_on_planning_not_organising():
    answer = _answer_for("Hur planerar man implementation av system?").lower()

    assert "mål" in answer
    assert "aktivit" in answer
    assert any(term in answer for term in ["tidplan", "tidpunkt", "när"])
    assert any(term in answer for term in ["resurs", "bemann"])
    assert any(term in answer for term in ["uppfölj", "överläm"])
    assert "projektorganisation" not in answer
    assert "styrgrupp" not in answer


def test_tomas_q02_organising_answer_stays_on_roles_structure_and_responsibility():
    answer = _answer_for("Hur organiserar man ett införande av ett system?").lower()

    assert any(term in answer for term in ["roll", "roller"])
    assert "ansvar" in answer
    assert any(term in answer for term in ["projektorganisation", "organisation", "bemanning"])
    assert any(term in answer for term in ["styrgrupp", "samverkan", "projektledning", "delprojekt"])
    assert "tidplan" not in answer
    assert "milstol" not in answer


def test_tomas_q05_technical_requirements_answer_stays_in_technical_scope():
    answer = _answer_for(
        "Vilka tekniska krav behöver vara uppfyllda innan vi börjar införa ett system?"
    ).lower()

    assert "teknisk plattform" in answer
    assert any(term in answer for term in ["it-miljö", "it-miljo", "driftmiljö"])
    assert any(term in answer for term in ["säkerhet", "behörighet", "svarstid", "koppling"])
    assert "kompetenser" not in answer
    assert "systembyte = verksamhetsförändring" not in answer


def test_tomas_q06_acceptance_test_responsibility_answers_directly_without_counter_questions():
    answer = _answer_for("Vem ansvarar för att acceptanstesta system?").lower()

    assert "testorganisation" in answer
    assert "roller" in answer
    assert "ansvar" in answer
    assert "resurs" in answer
    assert "?" not in answer
    assert "vem som ansvarar" not in answer


def test_tomas_q07_work_model_answer_explains_the_model_when_supported():
    answer = _answer_for("Finns det en arbetsmodell för att införa system?").lower()

    assert "ja" in answer
    assert "arbetsmodell" in answer
    assert any(term in answer for term in ["process", "cykel", "etapp"])
    assert "planera" in answer
    assert any(term in answer for term in ["godkännande", "uppfölj", "beslut"])
    assert "det framgår av dokumentet" not in answer
    assert "hans johansson" not in answer


def test_tomas_synthesis_prompt_contains_quality_guardrails():
    prompt = build_synthesis_prompt(
        "Hur planerar man implementation av system?",
        [
            {
                "source": "Verktyget_och_systeminforandet.pdf",
                "source_type": "pdf",
                "title": "Systemfaser",
                "text": "Planera aktiviteter, tidpunkter, resurser och uppföljning.",
                "pages": [1],
            }
        ],
        "Planera aktiviteter, tidpunkter, resurser och uppföljning.",
    ).lower()

    assert "pdf-metadata" in prompt or "pdf metadata" in prompt
    assert "inte" in prompt and "fakta" in prompt
    assert "håll dig till frågans fokus" in prompt
    assert "nedanstående figur" in prompt
    assert "inte räcker som sakstöd" in prompt or "inte som sakstöd" in prompt
    assert "källbundet" in prompt or "källunderlaget" in prompt
    assert "utvecklat" in prompt


def test_tomas_synthesis_prompt_contains_reasoning_and_scope_contract():
    prompt = build_synthesis_prompt(
        "Finns det en arbetsmodell för att införa system?",
        [
            {
                "source": "Verktyget_och_systeminforandet.pdf",
                "source_type": "pdf",
                "title": "Systeminförande",
                "text": (
                    "Arbetsmodellen för systeminförande innehåller planering, "
                    "genomförande, godkännande och uppföljning."
                ),
                "pages": [2],
            }
        ],
        "Ja, det finns en arbetsmodell med planering, genomförande, godkännande och uppföljning.",
    ).lower()

    assert "identifiera vad användaren faktiskt frågar" in prompt
    assert "börja med ett direkt svar" in prompt
    assert "beskriv modellens delar" in prompt
    assert "stanna inte vid" in prompt and "ja" in prompt
    assert "motfrågor" in prompt
    assert "källor redovisas separat" in prompt


def test_tomas_synthesis_prompt_warns_against_missing_figures_and_tables():
    prompt = build_synthesis_prompt(
        "Vilka tekniska krav behöver vara uppfyllda innan vi börjar införa ett system?",
        [
            {
                "source": "Tekniska_krav.pdf",
                "source_type": "pdf",
                "title": "Tekniska krav",
                "text": "Tabellen nedan visar krav på teknisk plattform, IT-miljöer och säkerhet.",
                "pages": [4],
            }
        ],
        "Det finns inte tillräckligt tydligt underlag för ett säkert svar.",
    ).lower()

    assert "figur" in prompt
    assert "tabell" in prompt
    assert "faktiska informationen" in prompt
    assert "inte finns i textutdraget" in prompt


def test_tomas_synthesis_rejects_rewrite_that_leaks_pdf_metadata():
    result = build_final_grounded_answer(
        "Hur planerar man implementation av system?",
        [
            {
                "source": "Verktyget_projektstyrning.pdf",
                "source_type": "pdf",
                "title": "Planering och uppföljning",
                "text": "Planera införandet med mål, aktiviteter, tidpunkter, resurser och uppföljning.",
                "pages": [1],
            }
        ],
        enable_synthesis=True,
        llm_rewrite=lambda prompt, model=None: (
            "Citrus Projektstyrning Hans Johansson Version 1 anger att införandet ska "
            "planeras med mål, aktiviteter, tidpunkter, resurser och uppföljning."
        ),
    )

    assert result["synthesis_used"] is False
    assert result["llm_status"] == "fallback_to_extractive_due_to_grounding_check"


def test_tomas_ingest_suppresses_generic_pdf_metadata_and_repeated_headers():
    pages = [
        (
            1,
            "\n".join(
                [
                    "Repeated PDF Header",
                    "Hans Johansson",
                    "Version 1",
                    "2024-01-01",
                    "1.1 Planering",
                    "Planeringen ska beskriva mål, aktiviteter, tidplan, resurser och uppföljning. "
                    "Den ska också visa hur genomförandet följs, vilka beslut som behövs och hur "
                    "ansvar för varje aktivitet dokumenteras.",
                ]
            ),
        ),
        (
            2,
            "\n".join(
                [
                    "Repeated PDF Header",
                    "Hans Johansson",
                    "Sida 2 av 3",
                    "1.2 Genomförande",
                    "Genomförandet ska beskriva ansvar, beslut och hur arbetet följs upp. "
                    "Texten ska bevara faktiska arbetsmoment, roller och kontroller även när "
                    "sidhuvuden och sidfötter tas bort.",
                ]
            ),
        ),
        (
            3,
            "\n".join(
                [
                    "Repeated PDF Header",
                    "Hans Johansson",
                    "3/3",
                    "1.3 Uppföljning",
                    "Uppföljningen ska beskriva resultat, avvikelser och fortsatt ansvar. "
                    "Den ska ge sakligt underlag om vad som genomförts och vilka nästa steg "
                    "som återstår efter införandet.",
                ]
            ),
        ),
    ]

    chunks = chunk_by_headings(pages, "synthetic", "pdf", "synthetic.pdf")

    assert chunks
    combined_text = "\n".join(chunk["text"] for chunk in chunks)
    assert "Repeated PDF Header" not in combined_text
    assert "Hans Johansson" not in combined_text
    assert "Version 1" not in combined_text
    assert "2024-01-01" not in combined_text
    assert "Sida 2 av 3" not in combined_text
    assert "3/3" not in combined_text
    assert "mål, aktiviteter, tidplan" in combined_text
    assert "ansvar, beslut" in combined_text


def test_tomas_ingest_noise_line_filter_is_generic_not_question_specific():
    assert is_noise_line("© Citrus i Stockholm")
    assert is_noise_line("Version nr xx.xx")
    assert is_noise_line("Författare: Anna Andersson")
    assert is_noise_line("Page 4 of 12")
    assert is_noise_line("2024/02/01")
    assert is_noise_line("a b c d e f g")

    assert not is_noise_line(
        "Planeringen ska beskriva mål, aktiviteter, tidplan, resurser och uppföljning."
    )


def test_tomas_ingest_preserves_numbered_heading_title_even_if_title_repeats():
    pages = [
        (
            1,
            "\n".join(
                [
                    "Kvalitetssäkring",
                    "1.1 Planering",
                    "Planeringen ska beskriva mål, aktiviteter, tidplan, resurser och uppföljning. "
                    "Den ska också visa hur genomförandet följs, vilka beslut som behövs och hur "
                    "ansvar för varje aktivitet dokumenteras.",
                ]
            ),
        ),
        (
            2,
            "\n".join(
                [
                    "Kvalitetssäkring",
                    "2.6",
                    "Kvalitetssäkring",
                    "Kvalitetssäkringen ska granska projektarbetet och resultatet vid viktiga "
                    "tidpunkter. Den ska vara ett stöd för projektet och ge sakliga förslag "
                    "till åtgärder.",
                ]
            ),
        ),
        (
            3,
            "\n".join(
                [
                    "Kvalitetssäkring",
                    "1.2 Uppföljning",
                    "Uppföljningen ska beskriva resultat, avvikelser och fortsatt ansvar. "
                    "Den ska ge sakligt underlag om vad som genomförts och vilka nästa steg "
                    "som återstår efter införandet.",
                ]
            ),
        ),
    ]

    chunks = chunk_by_headings(pages, "synthetic", "pdf", "synthetic.pdf")

    assert any(chunk["title"] == "2.6 Kvalitetssäkring" for chunk in chunks)
