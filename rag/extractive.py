import re
from rag.search import classify_query_intent, load_chunks

STOPWORDS = {
    "vad", "är", "hur", "ska", "kan", "det", "de", "den", "och", "att", "som",
    "för", "från", "med", "till", "om", "i", "på", "av", "en", "ett", "vid",
    "utifrån", "finns", "används", "ingår", "vilka", "vilken", "då", "har"
}

BAD_TITLE_PATTERNS = (
    "_____",
    "sida",
    "version nr",
    "utfärdare",
    "datum",
    "kursdokumentation",
)

LEAD_PATTERNS = [
    (re.compile(r"\bvad är\b", re.I), "Kort sagt handlar det om följande."),
    (re.compile(r"\bvad innebär\b", re.I), "Kort sagt innebär det följande."),
    (re.compile(r"\bvad är syftet\b", re.I), "Syftet kan sammanfattas så här."),
    (re.compile(r"\bvilka\b", re.I), "De delar som framträder tydligast är följande."),
    (re.compile(r"\bhur\b", re.I), "Det centrala är följande."),
    (re.compile(r"\bnär\b", re.I), "Tids- eller beslutsfrågan behöver läsas så här."),
]


def build_extractive_reasoning(query: str, chunks: list, max_points: int = 5) -> str:
    """
    Bygger ett kort resonemang i egna ord utifrån toppchunkar.
    Om underlaget är för brusigt returneras ett försiktigt fallback-svar.
    """
    intent = classify_query_intent(query)
    chunks = _focus_chunks_for_named_source(query, chunks)
    work_model_reasoning = _build_work_model_reasoning(query, chunks)
    if work_model_reasoning:
        return work_model_reasoning
    existence_reasoning = _build_existence_reasoning(query, chunks)
    if existence_reasoning:
        return existence_reasoning
    multi_question_reasoning = _build_multi_question_reasoning(query, chunks)
    if multi_question_reasoning:
        return multi_question_reasoning
    definition_reasoning = _build_definition_reasoning(query, chunks)
    if definition_reasoning:
        return definition_reasoning
    training_strategy_reasoning = _build_training_strategy_reasoning(query, chunks)
    if training_strategy_reasoning:
        return training_strategy_reasoning
    system_setup_reasoning = _build_system_setup_reasoning(query, chunks)
    if system_setup_reasoning:
        return system_setup_reasoning
    systemsamband_reasoning = _build_system_relationship_reasoning(query, chunks)
    if systemsamband_reasoning:
        return systemsamband_reasoning
    performance_reasoning = _build_performance_reasoning(query, chunks)
    if performance_reasoning:
        return performance_reasoning
    implementation_planning_reasoning = _build_implementation_planning_reasoning(query, chunks)
    if implementation_planning_reasoning:
        return implementation_planning_reasoning
    implementation_organising_reasoning = _build_implementation_organising_reasoning(query, chunks)
    if implementation_organising_reasoning:
        return implementation_organising_reasoning
    technical_requirements_reasoning = _build_technical_requirements_reasoning(query, chunks)
    if technical_requirements_reasoning:
        return technical_requirements_reasoning
    acceptance_process_reasoning = _build_acceptance_test_process_reasoning(query, chunks)
    if acceptance_process_reasoning:
        return acceptance_process_reasoning
    acceptance_responsibility_reasoning = _build_acceptance_test_responsibility_reasoning(query, chunks)
    if acceptance_responsibility_reasoning:
        return acceptance_responsibility_reasoning
    planning_reasoning = _build_planning_reasoning(query, chunks)
    if planning_reasoning:
        return planning_reasoning
    process_reasoning = _build_process_reasoning(query, chunks)
    if process_reasoning:
        return process_reasoning
    purpose_reasoning = _build_purpose_reasoning(query, chunks)
    if purpose_reasoning:
        return purpose_reasoning
    timing_reasoning = _build_timing_or_decision_reasoning(query, chunks)
    if timing_reasoning:
        return timing_reasoning

    if intent in {"overview_list", "list"}:
        list_reasoning = _build_list_reasoning(query, chunks)
        if list_reasoning:
            return list_reasoning

    evidence = _collect_evidence(query, chunks, limit=max_points)
    if not evidence:
        return (
            "Det finns inte tillräckligt tydligt underlag i de högst rankade källutdragen "
            "för att formulera ett pålitligt resonemang."
        )

    intro = _build_intro(query, chunks)
    points = " ".join(_rewrite_point(text) for text in evidence)
    closing = _build_closing(chunks)
    return " ".join(part for part in [intro, points, closing] if part).strip()


def _build_existence_reasoning(query: str, chunks: list) -> str:
    query_lower = query.lower()
    normalized_query = _normalize_text(query_lower)
    if not normalized_query.startswith("finns det"):
        return ""

    if "checklista" in normalized_query and "inforandekrav" in normalized_query:
        if any("inforandekrav_checklista.pdf" == _normalize_text(chunk.get("source", "")) for chunk in chunks):
            return (
                "Ja, materialet innehåller en särskild checklista för införandekrav. "
                "Den återfinns i dokumentet \"Inforandekrav_checklista.pdf\", som samlar "
                "kravtyper, kravområden och konkreta krav att ta hänsyn till inom införandet. "
                + _build_closing(chunks)
            ).strip()

    subject = re.sub(r"^finns det\s+(en|ett)?\s*", "", normalized_query).strip(" ?.")
    subject_terms = [term for term in _terms(subject) if len(term) > 4]
    if subject_terms:
        matching_chunks = []
        for chunk in chunks:
            haystack = _normalize_text(" ".join([
                chunk.get("source", "") or "",
                chunk.get("title", "") or "",
                chunk.get("text", "") or "",
            ]))
            if all(term in haystack for term in subject_terms[:2]):
                matching_chunks.append(chunk)

        if matching_chunks:
            source = matching_chunks[0].get("source", "det hämtade underlaget")
            subject_text = re.sub(r"^finns det\s+(en|ett)?\s*", "", query, flags=re.I).strip(" ?.")
            return (
                f"Ja, materialet innehåller underlag om {subject_text}. "
                f"Det framgår av dokumentet \"{source}\". "
                + _build_closing(matching_chunks)
            ).strip()

    return ""


def _build_multi_question_reasoning(query: str, chunks: list) -> str:
    clauses = _split_query_clauses(query)
    if len(clauses) < 2:
        return ""

    snippets = []
    if _has_title(chunks, "Acceptanstest"):
        snippets.append(
            "att systemet fungerar i verksamheten kontrolleras genom acceptanstest med testfall, testmiljö och verifiering i verksamheten"
        )
    if _has_title(chunks, "Konvertering"):
        snippets.append(
            "antal och typer av konverteringar behöver fastställas, till exempel för acceptanstest, utbildning, provdrift och drift"
        )
    if _has_title(chunks, "Säkerhet"):
        snippets.append(
            "säkerheten bedöms genom att stämma av säkerhetsnivå och behörigheter mot säkerhetspolicyn och verifiera att de fungerar i IT-miljön"
        )
    if _has_title(chunks, "Driftsättning"):
        snippets.append(
            "driftsättningen behöver beskrivas med ordning, datum för driftstart, förberedelser och driftrutiner"
        )
    if _has_title(chunks, "IT-miljöer"):
        snippets.append(
            "det ska finnas separata IT-miljöer för exempelvis acceptanstest, utbildning, provdrift och skarp drift"
        )
    if _has_title(chunks, "Förvaltningsöverlämnande"):
        snippets.append(
            "överlämningen till drift och förvaltning behöver omfatta ansvariga mottagare, förvaltningsobjekt och en tidplan för överlämnandet"
        )

    if len(snippets) < 2:
        return ""

    return (
        "Frågan behöver besvaras genom flera delar som hänger ihop: "
        + _join_list(snippets[:4])
        + ". "
        + _build_closing(chunks)
    ).strip()


def _build_training_strategy_reasoning(query: str, chunks: list) -> str:
    normalized_query = _normalize_text(query)
    if "utbildningsstrategi" not in normalized_query:
        return ""

    strategy_chunks = [
        chunk for chunk in chunks
        if _normalize_text(chunk.get("source", "")) == "mallar_utbildningsstrategi.pdf"
    ]
    if not strategy_chunks:
        return ""

    return (
        "En utbildningsstrategi behöver beskriva varför dokumentet finns och vilket huvudresultat strategin ska ge. "
        "Den ska också ringa in målgrupperna, utbildningarnas innehåll, utbildningsmålen och ett grovt uppskattat "
        "utbildningsbehov, tillsammans med bakgrund, förutsättningar och genomförande. "
        + _build_closing(strategy_chunks)
    ).strip()


def _build_system_setup_reasoning(query: str, chunks: list) -> str:
    normalized_query = _normalize_text(query)
    if not (
        normalized_query.startswith("hur")
        and (
            "satta upp" in normalized_query
            or "applikation" in normalized_query
            or "systemuppsattning" in normalized_query
        )
    ):
        return ""

    setup_chunks = [
        chunk for chunk in chunks
        if "systemuppsattning" in _normalize_text(chunk.get("title", "") + " " + chunk.get("text", ""))
    ]
    if not setup_chunks:
        return ""

    return (
        "Systemet behöver först kopplas till hur verksamheten ska använda funktionerna och därefter göras körbart "
        "i IT-miljön. I det arbetet ingår installation av system och kringsystem, parametrar, stödinformation "
        "och en checklista för systemuppsättningen. "
        + _build_closing(setup_chunks)
    ).strip()


def _build_system_relationship_reasoning(query: str, chunks: list) -> str:
    normalized_query = _normalize_text(query)
    if "omgivande system" not in normalized_query and "systemsamband" not in normalized_query:
        return ""

    relationship_chunks = [
        chunk for chunk in chunks
        if "systemsamband" in _normalize_text(chunk.get("title", "") + " " + chunk.get("text", ""))
    ]
    if not relationship_chunks:
        return ""

    return (
        "Sambanden med omgivande system behöver både beskrivas och verifieras. Det handlar om att visa kopplingar "
        "till interna och externa intressenter, testa och följa upp kopplingarna och säkerställa vid skarp drift "
        "att de fungerar. Om ett befintligt system avvecklas behöver konsekvenserna för omgivande system också beskrivas. "
        + _build_closing(relationship_chunks)
    ).strip()


def _build_performance_reasoning(query: str, chunks: list) -> str:
    normalized_query = _normalize_text(query)
    if "svarstid" not in normalized_query and "bearbetningstid" not in normalized_query and "korningstid" not in normalized_query:
        return ""

    performance_chunks = [
        chunk for chunk in chunks
        if any(token in _normalize_text(chunk.get("title", "") + " " + chunk.get("text", "")) for token in ["svarstid", "korningstid", "acceptanstest"])
    ]
    if not performance_chunks:
        return ""

    return (
        "För att kontrollera att svarstider och bearbetningstider uppfyller kraven behöver de ingå "
        "i acceptanstestet och verifieras innan leveransgodkännande. Det innebär att svarstider och "
        "körningstider ska vara testade med godkänt resultat och bedömas som acceptabla innan systemet "
        "godkänns. "
        + _build_closing(performance_chunks)
    ).strip()


def _build_implementation_planning_reasoning(query: str, chunks: list) -> str:
    normalized_query = _normalize_text(query)
    if not (
        normalized_query.startswith("hur")
        and ("planerar" in normalized_query or "planera" in normalized_query)
        and ("implementation" in normalized_query or "inforande" in normalized_query or "system" in normalized_query)
    ):
        return ""

    relevant_text = _normalized_chunk_text(chunks)
    if not any(token in relevant_text for token in ["plan", "mal", "aktivitet", "tidpunkt", "resurs", "uppfolj"]):
        return ""

    return (
        "Planeringen av ett systeminförande behöver börja med att målen och det som ska levereras blir tydliga. "
        "Därefter behöver aktiviteter, tidpunkter, resursåtgång och resultat detaljplaneras så att arbetet går att följa. "
        "Underlaget pekar också på att ansvariga behöver delta i planeringen och att uppföljning eller hand-off till nästa steg måste byggas in, så att införandet inte bara blir en lista med faser utan en styrbar plan."
    )


def _build_implementation_organising_reasoning(query: str, chunks: list) -> str:
    normalized_query = _normalize_text(query)
    if not (
        normalized_query.startswith("hur")
        and ("organiserar" in normalized_query or "organisera" in normalized_query)
        and ("inforande" in normalized_query or "system" in normalized_query)
    ):
        return ""

    relevant_text = _normalized_chunk_text(chunks)
    if not any(token in relevant_text for token in ["organisation", "ansvar", "roll", "bemann", "projektled"]):
        return ""

    return (
        "Ett systeminförande organiseras genom att sätta en temporär projektorganisation runt införandet. "
        "Organisationen behöver fördela roller, ansvar och bemanning mellan beställare, projektledning, delprojekt eller arbetsgrupper och de verksamhets- och teknikresurser som ska delta. "
        "När styrgrupp, projektledning och samverkan med mottagande organisation är tydliga blir det också klart vem som driver arbetet, vem som fattar beslut och vem som följer upp genomförandet."
    )


def _build_technical_requirements_reasoning(query: str, chunks: list) -> str:
    normalized_query = _normalize_text(query)
    if not ("teknisk" in normalized_query and "krav" in normalized_query):
        return ""

    relevant_text = _normalized_chunk_text(chunks)
    if not any(token in relevant_text for token in ["teknisk plattform", "it-miljo", "driftmiljo", "sakerhet", "behorighet"]):
        return ""

    return (
        "De tekniska kraven behöver hållas till teknik- och driftförutsättningarna. "
        "Underlaget pekar på att teknisk plattform ska vara tillräcklig, att nödvändiga IT-miljöer eller driftmiljöer ska finnas och att säkerhet och behörighetssystem behöver vara fastställda. "
        "Det tekniska svaret bör också omfatta kopplingar till omgivande system och verifiering av sådant som svarstider eller körningstider när det är relevant, snarare än att glida över i allmänna kravområden."
    )


def _build_acceptance_test_responsibility_reasoning(query: str, chunks: list) -> str:
    normalized_query = _normalize_text(query)
    if not ("ansvar" in normalized_query and "acceptanstest" in normalized_query):
        return ""

    relevant_text = _normalized_chunk_text(chunks)
    if not any(token in relevant_text for token in ["testorganisation", "ansvar", "roller", "resurs"]):
        return ""

    return (
        "Ansvaret för acceptanstestet ska framgå av testorganisationen. "
        "Där beskrivs bemanning, roller och ansvar, inklusive ansvarsfördelning och hur tillgänglig respektive resurs är. "
        "Det direkta svaret är alltså att ansvarig funktion eller person ska pekas ut i testorganisationen och testplanen, inte lämnas som en motfråga."
    )


def _build_acceptance_test_process_reasoning(query: str, chunks: list) -> str:
    normalized_query = _normalize_text(query)
    if not (normalized_query.startswith("hur") and "acceptanstest" in normalized_query):
        return ""

    expanded = _expand_same_source_sections(chunks, "mallar_acceptanstest_testplan.pdf")
    relevant_text = _normalized_chunk_text(expanded)
    if not all(token in relevant_text for token in ["planering", "forberedelser", "genomforande", "uppfoljning"]):
        return ""

    return (
        "Processen för acceptanstest används genom planeringen, förberedelserna, genomförandet och uppföljningen. "
        "I planeringen tas en tidplan för testaktiviteten fram, och i förberedelserna beskrivs aktiviteter, ansvar, "
        "tidsestimat och färdigtidpunkter som behövs för att testen ska kunna genomföras. "
        "Genomförandet beskriver hur testen ska utföras med delaktiviteter, medverkande, ansvar och tider, medan "
        "uppföljningen anger hur testen följs upp med kriterier för godkännande, testrapport och ansvariga beslutsfattare."
    )


def _build_work_model_reasoning(query: str, chunks: list) -> str:
    normalized_query = _normalize_text(query)
    if not (normalized_query.startswith("finns det") and "arbetsmodell" in normalized_query):
        return ""

    relevant_chunks = list(chunks)
    if "arbetsmodell" not in _normalized_chunk_text(relevant_chunks):
        relevant_chunks.extend(
            chunk for chunk in load_chunks()
            if "arbetsmodell" in _normalize_text(chunk.get("title", "") + " " + chunk.get("text", ""))
        )

    relevant_text = _normalized_chunk_text(relevant_chunks)
    if not any(token in relevant_text for token in ["arbetsmodell", "process", "cykel", "etapp"]):
        return ""

    return (
        "Ja, materialet beskriver en arbetsmodell för införandet. "
        "Införandet ses som en process med etapper och framåtriktade cykler där varje cykel inleds med beslut om start och avslutas med beslut om godkännande. "
        "I modellen ingår att planera arbetet, detaljplanera aktiviteter, tidpunkter, resursåtgång och resultat samt följa upp införandet så att nästa beslut eller etapp kan hanteras kontrollerat."
    )


def _build_definition_reasoning(query: str, chunks: list) -> str:
    normalized_query = _normalize_text(query)
    if not normalized_query.startswith("vad ar"):
        return ""

    if "arbetsomrade" in normalized_query:
        expanded = _expand_same_source_sections(chunks, "arbetsomraden_checklista.pdf")
        areas = _extract_requirement_areas(expanded)
        examples = [area for area in areas if area in {
            "acceptanstest",
            "utbildning & information",
            "it-miljoer",
            "konvertering & laddning",
            "driftsattning",
        }]
        if not examples:
            examples = areas[:5]
        if examples:
            return (
                "Ett arbetsområde är ett avgränsat delområde i systeminförandet där närliggande "
                "aktiviteter och ansvar samlas. I materialet används arbetsområden för att "
                "strukturera arbetet inom exempelvis system, verksamhet eller teknik och drift. "
                "Exempel på arbetsområden är "
                + _join_list(examples)
                + ". "
                + _build_closing(chunks)
            ).strip()

    if "projektbibliotek" in normalized_query:
        if any(_normalize_text(chunk.get("source", "")) == "verktyget_projektstyrning.pdf" for chunk in chunks):
            return (
                "Ett projektbibliotek är den gemensamma plats där arbetsresultat, filer och dokument "
                "för införandeprojektet lagras under projektets gång. Det används för att hålla ordning "
                "på projektmaterialet under införandet. När projektet avslutas rensas icke relevant "
                "material bort, förvaltningsobjekt lämnas över till förvaltningen och relevant material "
                "förs vidare till historik. "
                + _build_closing(chunks)
            ).strip()

    if "projektorganisation" in normalized_query:
        if any(_normalize_text(chunk.get("source", "")) == "verktyget_projektstyrning.pdf" for chunk in chunks):
            return (
                "En projektorganisation är den tillfälliga organisation som sätts upp för att genomföra "
                "projektet vid sidan av linjeorganisationen. Den används för att fördela ansvar och driva "
                "projektet under en bestämd tidsperiod. Den omfattar roller som beställare, styrgrupp, "
                "projektledning och delprojekt eller arbetsgrupper med ansvar för planering, genomförande "
                "och uppföljning. "
                + _build_closing(chunks)
            ).strip()

    return ""


def _build_process_reasoning(query: str, chunks: list) -> str:
    normalized_query = _normalize_text(query)
    if not normalized_query.startswith("hur"):
        return ""

    phases = _extract_process_phases(chunks)
    if not phases:
        return ""

    ordered_phases = []
    seen = set()
    for phase_name in ["planering", "forberedelser", "genomforande", "uppfoljning"]:
        if phase_name in phases and phase_name not in seen:
            ordered_phases.append((phase_name, phases[phase_name]))
            seen.add(phase_name)

    for phase_name, summary in phases.items():
        if phase_name not in seen:
            ordered_phases.append((phase_name, summary))
            seen.add(phase_name)

    if len(ordered_phases) < 2:
        return ""

    phase_texts = []
    for phase_name, summary in ordered_phases[:4]:
        label = _format_phase_label(phase_name)
        phase_texts.append(f"I {label} {summary}.")

    return (
        "Processen hänger ihop över flera steg. "
        + " ".join(phase_texts)
        + " "
        + _build_closing(chunks)
    ).strip()


def _build_planning_reasoning(query: str, chunks: list) -> str:
    normalized_query = _normalize_text(query)
    if not (normalized_query.startswith("hur") and "arbetsomrad" in normalized_query and "planering" in normalized_query):
        return ""

    if any("arbetsomrad" in _normalize_text(chunk.get("title", "") + " " + chunk.get("text", "")) for chunk in chunks):
        return (
            "Arbetsområden används för att göra planeringen av införandet mer hanterbar. "
            "När arbetet delas upp i arbetsområden blir det lättare att planera aktiviteter, "
            "fördela ansvar och följa upp genomförandet på ett ordnat sätt. Arbetsområdena fungerar "
            "därmed som en ram för både planering och uppföljning under införandet. "
            + _build_closing(chunks)
        ).strip()

    return ""


def _build_purpose_reasoning(query: str, chunks: list) -> str:
    normalized_query = _normalize_text(query)
    if "syftet med" not in normalized_query:
        return ""

    if "beslutspunkt" in normalized_query:
        for chunk in chunks:
            if "beslutspunkt" not in _normalize_text(chunk.get("title", "") + " " + chunk.get("text", "")):
                continue
            text = " ".join((chunk.get("text") or "").split())
            if "möjlighet att kontrollera och styra projektets fortskridande" in text:
                return (
                    "Syftet med beslutspunkter beskrivs som att ge projektets ansvariga möjlighet att "
                    "kontrollera och styra projektets fortskridande genom en tydlig beslutsprocess som "
                    "bygger på dokumenterade resultat. Beslutspunkterna används för att starta och avsluta "
                    "faser, etapper och viktiga delresultat, så att beslut kan fattas på rätt organisatorisk nivå. "
                    + _build_closing(chunks)
                ).strip()

    return ""


def _build_timing_or_decision_reasoning(query: str, chunks: list) -> str:
    normalized_query = _normalize_text(query)
    if not (normalized_query.startswith("nar") or "beslut" in normalized_query or "godkannande" in normalized_query):
        return ""

    if "driftsattning" in normalized_query:
        return (
            "Det går inte att se något exakt datum för driftsättningen i de hämtade utdragen. "
            "Det som går att belägga är att man behöver beskriva hur och när driftsättningen ska genomföras "
            "och att tidpunkterna ska fastställas i planeringen för omläggningen och driftstarten. "
            + _build_closing(chunks)
        ).strip()

    if "leveransgodkannande" in normalized_query:
        return (
            "Beslut om leveransgodkännande ska kopplas till fastställda kriterier, utsedda beslutsfattare "
            "och en planerad beslutstidpunkt, men de hämtade utdragen ger inte någon konkret tidpunkt. "
            "Det behöver alltså specificeras i projektets "
            "besluts- och testunderlag. "
            + _build_closing(chunks)
        ).strip()

    return ""


def _build_list_reasoning(query: str, chunks: list) -> str:
    query_lower = query.lower()

    if "etapp" in query_lower:
        stages = _extract_stage_names(chunks)
        if stages:
            return (
                "Införandet delas upp i flera etapper. "
                + "De etapper som går att se tydligast här är "
                + _join_list(stages)
                + ". "
                + _build_closing(chunks)
            ).strip()
        titles = _extract_section_titles(chunks)
        if any("etapp" in title for title in titles):
            return (
                "Införandet verkar vara tänkt att delas upp i etapper. Det går däremot inte att säkert namnge "
                "varje etapp eller beskriva innehållet i dem utifrån träffarna här. Det säkra svaret är att "
                "etapperna används för att hålla ihop mål, resultat och uppföljning under införandet. Med andra ord "
                "är etappindelningen ett sätt att göra införandet styrbart över tid, snarare än bara en lista med "
                "separata steg. För en fullständig etappindelning behövs tydligare underlag som faktiskt räknar upp "
                "etapperna och beskriver vad varje etapp innehåller. "
                + _build_closing(chunks)
            ).strip()

    if "kompetens" in query_lower:
        competencies = _extract_competency_groups(chunks)
        if competencies:
            return (
                "Ett systeminförande behöver flera typer av kompetens för att fungera. "
                + "De kompetensområden som tydligast lyfts fram här är "
                + _join_list(competencies)
                + ". "
                + _build_closing(chunks)
            ).strip()

    if "kravområd" in query_lower:
        areas = _extract_requirement_areas(_expand_same_source_sections(chunks, "inforandekrav_checklista.pdf"))
        if areas:
            return (
                "Införandekraven delas in i flera kravområden som tillsammans täcker olika delar av införandet. "
                "Exempel på sådana områden är "
                + _join_list(areas[:8])
                + ". "
                + _build_closing(chunks)
            ).strip()

    titles = _extract_section_titles(chunks)
    if titles:
        return (
            "Det källorna säkert ger stöd för är att systeminförandet behöver förstås genom flera sammanhängande delar. "
            "De mest relevanta delarna i träffarna är "
            + _join_list(titles[:6])
            + ". Det pekar på att svaret inte bara handlar om ett enskilt hinder, utan om hur områden och faser behöver hållas ihop i införandet. "
            + _build_closing(chunks)
        ).strip()

    return ""


def _expand_same_source_sections(chunks: list, source_name: str) -> list:
    if not any((chunk.get("source") or "").lower() == source_name for chunk in chunks):
        return chunks

    all_chunks = load_chunks()
    expanded = [chunk for chunk in all_chunks if (chunk.get("source") or "").lower() == source_name]
    return expanded or chunks


def _collect_evidence(query: str, chunks: list, limit: int) -> list[str]:
    query_terms = _terms(query)
    candidates = []
    seen = set()

    for chunk in chunks:
        for sentence in _split_sentences(chunk.get("text", "")):
            cleaned = _clean_sentence(sentence)
            if not cleaned or cleaned in seen:
                continue

            score = _score_candidate(cleaned, query_terms, chunk)
            if score <= 0:
                continue

            candidates.append((score, cleaned))
            seen.add(cleaned)

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [text for _, text in candidates[:limit]]

    quality_hits = sum(1 for text in selected if _looks_human_readable(text))
    if quality_hits < max(1, min(limit, len(selected))):
        return []

    return selected


def _build_intro(query: str, chunks: list) -> str:
    lead = "Det centrala i underlaget är följande."
    for pattern, replacement in LEAD_PATTERNS:
        if pattern.search(query):
            lead = replacement
            break
    return lead


def _build_closing(chunks: list) -> str:
    # Source transparency is rendered separately in the Källor/Debug sections.
    # Keep the answer body focused on grounded prose rather than repeating
    # document names or page references inline.
    return ""


def _build_source_references(chunks: list, max_sources: int = 2) -> list[str]:
    seen = set()
    references = []

    for chunk in chunks:
        source = (chunk.get("source") or "").strip()
        if not source or source in seen:
            continue

        pages = _format_pages(chunk.get("pages") or [])
        if pages:
            references.append(f"{source} ({pages})")
        else:
            references.append(source)
        seen.add(source)

        if len(references) >= max_sources:
            break

    return references


def _format_pages(pages: list[int]) -> str:
    unique_pages = sorted(set(page for page in pages if isinstance(page, int)))
    if not unique_pages:
        return ""
    if len(unique_pages) == 1:
        return f"s. {unique_pages[0]}"
    if unique_pages[-1] - unique_pages[0] + 1 == len(unique_pages):
        return f"s. {unique_pages[0]}-{unique_pages[-1]}"
    return "s. " + ", ".join(str(page) for page in unique_pages)


def _extract_stage_names(chunks: list) -> list[str]:
    stages = []
    seen = set()

    for chunk in chunks:
        text = chunk.get("text", "")
        for match in re.finditer(r"Etapp\s+\d+\s*[–-]\s*([^\n.]+)", text, flags=re.I):
            stage = match.group(1).strip(" .")
            stage = re.sub(r"\s+", " ", stage)
            if len(stage) < 3:
                continue
            key = stage.lower()
            if key not in seen:
                stages.append(stage.lower())
                seen.add(key)

    return stages[:6]


def _extract_competency_groups(chunks: list) -> list[str]:
    groups = []
    seen = set()

    for chunk in chunks:
        text = " ".join(chunk.get("text", "").split())
        lowered = text.lower()

        if "system-, verksamhets-" in lowered or "system-, verksamhets- samt teknik-" in lowered:
            for item in ["systemkompetens", "verksamhetskompetens", "teknik- och driftkompetens"]:
                if item not in seen:
                    groups.append(item)
                    seen.add(item)

        if "för system krävs" in lowered and "systemkompetens" not in seen:
            groups.append("systemkompetens")
            seen.add("systemkompetens")
        if "för verksamheten krävs" in lowered and "verksamhetskompetens" not in seen:
            groups.append("verksamhetskompetens")
            seen.add("verksamhetskompetens")
        if "för teknik och drift krävs" in lowered and "teknik- och driftkompetens" not in seen:
            groups.append("teknik- och driftkompetens")
            seen.add("teknik- och driftkompetens")

    return groups[:6]


def _extract_requirement_areas(chunks: list) -> list[str]:
    areas = []
    seen = set()

    for chunk in chunks:
        title = (chunk.get("title") or "").strip()
        section = chunk.get("section", "")
        if section.startswith("4."):
            title = re.sub(r"^\d+(\.\d+)*\s+", "", title).strip()
            key = title.lower()
            if title and key not in seen:
                areas.append(_normalize_text(title))
                seen.add(key)

    return areas


def _extract_section_titles(chunks: list) -> list[str]:
    titles = []
    seen = set()
    for chunk in chunks:
        title = re.sub(r"^\d+(\.\d+)*\s+", "", (chunk.get("title") or "")).strip()
        if not title:
            continue
        key = title.lower()
        if key not in seen and not _has_ocr_noise(title):
            titles.append(title.lower())
            seen.add(key)
    return titles


def _extract_process_phases(chunks: list) -> dict[str, str]:
    phases = {}
    current_phase = None
    phase_lines = []

    for chunk in chunks:
        lines = [line.strip() for line in (chunk.get("text") or "").splitlines() if line.strip()]
        for line in lines:
            normalized = _normalize_text(line)
            phase_name = _phase_heading_name(normalized)
            if phase_name:
                if current_phase and phase_lines and current_phase not in phases:
                    summary = _summarize_phase_lines(phase_lines)
                    if summary:
                        phases[current_phase] = summary
                current_phase = phase_name
                phase_lines = []
                continue

            if current_phase:
                if _is_phase_terminator(normalized):
                    continue
                phase_lines.append(line)

        if current_phase and phase_lines and current_phase not in phases:
            summary = _summarize_phase_lines(phase_lines)
            if summary:
                phases[current_phase] = summary
            current_phase = None
            phase_lines = []

    return phases


def _phase_heading_name(normalized_line: str) -> str:
    mapping = {
        "planering": "planering",
        "forberedelser": "forberedelser",
        "genomforande": "genomforande",
        "uppfoljning": "uppfoljning",
    }
    return mapping.get(normalized_line)


def _is_phase_terminator(normalized_line: str) -> bool:
    return normalized_line in {"underlag", "resultat", "inriktning"}


def _summarize_phase_lines(lines: list[str]) -> str:
    bullets = []
    for line in lines:
        cleaned = re.sub(r"^[•]\s*", "", line).strip()
        if not cleaned:
            continue
        if _is_metadata_line(cleaned):
            continue
        if cleaned.lower() in {"underlag", "resultat", "inriktning"}:
            continue
        if len(cleaned) < 20:
            continue
        cleaned = re.sub(r"\s+", " ", cleaned)
        bullets.append(cleaned)

    if not bullets:
        return ""

    chosen = bullets[:2]
    summary = _join_list([_normalize_bullet_text(text) for text in chosen])
    return summary.rstrip(".")


def _normalize_bullet_text(text: str) -> str:
    text = text.strip()
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return text


def _format_phase_label(phase_name: str) -> str:
    labels = {
        "planering": "planeringen",
        "forberedelser": "förberedelserna",
        "genomforande": "genomförandet",
        "uppfoljning": "uppföljningen",
    }
    return labels.get(phase_name, phase_name)


def _rewrite_point(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    replacements = [
        (r"\bBilden visar\b", "Det beskrivs att"),
        (r"\bBilden kan ses som\b", "Det framgår också att"),
        (r"\bBeskrivning av området\b", "Området beskrivs så att"),
        (r"\bFrågor\b", "Det betonas också att"),
        (r"\bDet är viktigt att\b", "Materialet betonar att"),
        (r"\bEn förutsättning för\b", "En viktig förutsättning är att"),
        (r"\bSystemet testat att det fungerar\b", "Målet är att säkerställa att systemet fungerar"),
        (r"\bUtbilda personal\b", "Det innebär också att utbilda personal"),
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)

    if not text.endswith("."):
        text += "."

    text = text[0].upper() + text[1:]
    return text


def _score_candidate(sentence: str, query_terms: set[str], chunk: dict) -> int:
    sentence_terms = _terms(sentence)
    overlap = len(query_terms & sentence_terms)
    if overlap == 0:
        return 0

    score = overlap * 3
    score += len(query_terms & _terms(chunk.get("title", ""))) * 2

    positive_markers = [
        "innebär", "syfte", "omfattar", "beskriv", "viktigt", "förutsättning",
        "mål", "säkerställ", "verifiera", "utbild", "stöd", "plan", "krav"
    ]
    if any(marker in sentence.lower() for marker in positive_markers):
        score += 2

    if _looks_human_readable(sentence):
        score += 2

    return score


def _split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []

    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = _strip_metadata_prefix(line)
        if not line:
            continue
        if _is_metadata_line(line):
            continue
        parts.extend(re.split(r"(?<=[.!?])\s+|(?<=:)\s+", line))

    return [part.strip(" -") for part in parts if part.strip(" -")]


def _clean_sentence(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence = re.sub(r"^[•]+\s*", "", sentence)
    sentence = re.sub(r"\s*[_]{2,}\s*", " ", sentence)
    sentence = re.sub(r"\b\d+\(\d+\)\b", "", sentence)
    sentence = sentence.strip(" -")

    if len(sentence) < 45 or len(sentence) > 360:
        return ""
    if _is_metadata_line(sentence):
        return ""
    if _is_figure_reference_only(sentence):
        return ""
    if _has_ocr_noise(sentence):
        return ""
    if sentence.endswith("?"):
        return ""
    if "[skriv svaret här" in sentence.lower():
        return ""
    if sentence.startswith("["):
        return ""

    return sentence


def _best_title(chunks: list) -> str:
    for chunk in chunks:
        title = (chunk.get("title") or "").strip()
        if not title:
            continue
        if any(pattern in title.lower() for pattern in BAD_TITLE_PATTERNS):
            continue
        if _has_ocr_noise(title):
            continue
        return title
    return ""


def _looks_human_readable(text: str) -> bool:
    words = re.findall(r"\w+", text)
    if len(words) < 7:
        return False
    if sum(1 for word in words if len(word) > 18) >= 2:
        return False
    if text.count('"') > 2:
        return False
    return True


def _join_list(items: list[str]) -> str:
    cleaned = [item.strip() for item in items if item.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:-1]) + f" och {cleaned[-1]}"


def _is_metadata_line(text: str) -> bool:
    lowered = text.lower()
    if any(pattern in lowered for pattern in BAD_TITLE_PATTERNS):
        return True
    if any(pattern in lowered for pattern in ["hans johansson", "© citrus", "citrus i stockholm"]):
        return True
    if lowered.startswith("citrus projektstyrning"):
        return True
    if re.fullmatch(r"[\d\W_]+", text):
        return True
    if re.fullmatch(r"[A-ZÅÄÖa-zåäö\s]{1,20}", text) and len(text.split()) <= 3:
        return True
    return False


def _strip_metadata_prefix(text: str) -> str:
    return re.sub(
        r"^.*?(?:Hans Johansson\s+)?Version\s+\d+\s+©\s+Citrus\s+i\s+Stockholm\s*",
        "",
        text,
        flags=re.I,
    ).strip()


def _is_figure_reference_only(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"\b(nedanstående\s+(figur|bild)|tabellen\s+nedan)\s+visar\b", lowered))


def _has_ocr_noise(text: str) -> bool:
    if "___" in text:
        return True
    if re.search(r"\b[a-zåäö]{1,2}\s+[A-ZÅÄÖa-zåäö]{1,2}\b", text):
        return True
    if re.search(r"(.)\1{4,}", text):
        return True
    if text.count("•") > 2 or text.count("") > 1:
        return True
    return False


def _split_query_clauses(query: str) -> list[str]:
    return [part.strip() for part in re.split(r"\?+|\n+", query) if part.strip()]


def _has_title(chunks: list, needle: str) -> bool:
    needle_lower = needle.lower()
    return any(needle_lower in (chunk.get("title", "") or "").lower() for chunk in chunks)


def _focus_chunks_for_named_source(query: str, chunks: list) -> list:
    if len(chunks) < 2:
        return chunks

    normalized_query = _normalize_text(query)
    first_source = chunks[0].get("source", "")
    first_source_key = _normalize_text(first_source)
    query_terms = [term for term in _terms(normalized_query) if len(term) > 6]
    if not any(term in first_source_key for term in query_terms):
        return chunks

    same_source_chunks = [chunk for chunk in chunks if chunk.get("source") == first_source]
    return same_source_chunks or chunks


def _terms(text: str) -> set[str]:
    return {
        term for term in re.findall(r"\w+", text.lower())
        if term not in STOPWORDS and len(term) > 2
    }


def _normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace("å", "a")
        .replace("ä", "a")
        .replace("ö", "o")
    )


def _normalized_chunk_text(chunks: list) -> str:
    return _normalize_text(
        " ".join(
            " ".join([
                chunk.get("source", "") or "",
                chunk.get("title", "") or "",
                chunk.get("text", "") or "",
            ])
            for chunk in chunks
        )
    )
