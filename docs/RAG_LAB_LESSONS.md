# RAG Lab Journal

**Utvecklingsjournal för chatbotens RAG-lösning**

**Författare**

Tomas Helmfridsson och OpenAI Codex

**Startdatum**

2026-07-10

**Senast uppdaterad**

2026-07-29

Detta är en kronologisk journal över experiment, fel, mätningar och beslut i arbetet med chatbotens RAG-lösning. Journalen beskriver hur lösningen förändrades och varför. Den ska inte läsas som dokumentation av dagens arkitektur.

Den aktuella lösningen beskrivs i [RAG Solution](./rag-solution.html).

## Så läses journalen

Varje journalpost använder så långt möjligt samma struktur:

- **Utgångsläge** – problemet eller frågan vi började med
- **Observation** – vad tester, loggar eller kodgranskning visade
- **Ändring** – vad vi gjorde
- **Resultat** – vad som kunde verifieras
- **Beslut eller lärdom** – vad observationen betyder för fortsatt arbete

Uppgifter om modeller, feature flags och standardvärden är historiska när de står i en daterad journalpost. [RAG Solution](./rag-solution.html) är alltid den auktoritativa beskrivningen av aktuellt läge.

## Aktuellt arbetsläge

Den 29 juli 2026 är den kontrollerade RAG-vägen tillfällig standard i Docker/Hugging Face:

```text
SYSTEMINFORANDE_ENABLE_AGENTIC_RAG=false
```

Den kostnadsstyrda agentkedjan finns kvar bakom feature flaggan. Agent 1–3 använder 20B och endast ett underkänt Agent 3-svar kan utlösa en kompakt 120B-korrigering.

Den lokala testsviten passerar med 173 tester. Den senaste kompletta RAGAS-körningen för Agentic RAG kunde däremot inte slutföras eftersom Hugging Face-krediterna tog slut.

## 2026-07-10 – Från PDF-samling till första RAG-lösning

### Utgångsläge

Målet var både att skapa en praktiskt användbar chatbot för systeminförande och att förstå vilka delar som faktiskt avgör kvaliteten i en RAG-lösning.

### Ändring

Den första kedjan byggdes med:

1. PDF-extraktion sida för sida
2. rubrikbaserad chunkning
3. lokalt JSON-index
4. BM25-baserad retrieval
5. modellfri svargenerering
6. klickbara källänkar via GitHub Pages
7. Gradio som användargränssnitt på Hugging Face

### Observation

Det blev snabbt tydligt att språkmodellen inte var den enda eller ens den första kvalitetsfrågan. Om fel chunkar valdes kunde ingen prompt skapa ett pålitligt svar.

### Lärdom

RAG är en kedja. Materialberedning, chunkning och retrieval måste fungera innan modellval och promptjustering kan bedömas meningsfullt.

## 2026-07-10 – Den första systematiska kvalitetsgranskningen

### Utgångsläge

Chatboten fungerade tekniskt, men enstaka klicktester räckte inte för att avgöra om den hittade rätt underlag.

### Observation

Livefrågor via Gradio API visade flera återkommande fel:

- fel dokumentfamilj rankades högst
- detaljavsnitt slog ut bättre översiktsavsnitt
- processfrågor fick fragmentariska svar
- mindre stavfel kunde ge svaga träffar
- hårdkodade dokumentnamnsboostar gjorde retrievalen skör

### Diagnos

Problemen kom huvudsakligen från retrieval:

- otillräcklig svensk normalisering
- för svag skillnad mellan rubrik, källnamn och brödtext
- ingen tydlig hantering av frågetyp
- ingen lokal reranking av toppkandidater
- för stor tillit till en enda rankningssignal

### Ändring

Retrievalen kompletterades med:

- svenska böjnings- och stavningsvarianter
- titel-, käll- och dokumentfamiljssignaler
- frågetyper för definition, syfte, lista, process och beslut
- generella egenskapsbaserade boostar
- en lokal reranker ovanpå BM25
- lokala regressionstester för verkliga svagfrågor

### Resultat

Översiktsfrågor hittade oftare översiktsavsnitt och processfrågor fick bättre sammanhängande underlag.

### Lärdom

Ett svar som ser ut som ett promptproblem är ofta ett retrievalproblem. Rankningen bör testas som en egen produkt, inte bara bedömas genom sluttexten.

## 2026-07-11–2026-07-17 – Frågetypsstyrd modellfri syntes

### Utgångsläge

Även med bättre retrieval blev det modellfria svaret ibland mekaniskt. Samma svarsmall passade inte för alla frågor.

### Observation

Fyra svarstyper behövde olika behandling:

- listfrågor behövde korta och fullständiga uppräkningar
- syftesfrågor behövde skilja mål från aktiviteter
- tids- och beslutsfrågor behövde uttrycka när något sker
- process- och planeringsfrågor behövde ett begripligt förlopp

### Ändring

Separata lokala generatorer infördes för listor, syfte, tid/beslut, process och planering.

### Resultat

Svarens form blev bättre anpassad till frågan utan att en extern modell behövde skriva ny saktext.

### Lärdom

Modellfri syntes kan vara både säker och användbar när frågetypen är tydlig. Korthet är inte alltid ett fel: frågan `Vilka etapper finns?` bör i första hand ge en korrekt lista. Mer resonemang kräver en mer detaljerad fråga.

## 2026-07-18 – Första fasta 30-frågebaslinjen

### Utgångsläge

Kvaliteten bedömdes fortfarande för mycket genom enstaka exempel och subjektiv läsning.

### Ändring

En fast uppsättning med 30 frågor infördes för jämförbara körningar. Frågorna täckte bland annat:

- definitioner
- listor
- processer
- ansvar
- planering
- beslut och tidpunkt
- svaga eller tvetydiga formuleringar

### Resultat

En tidig HF-baslinje gav:

- 10 godkända svar
- 6 underkända svar
- 14 svar som krävde manuell granskning

Många svar var källgrundade men ofullständiga, indirekta eller för mekaniska.

### Lärdom

Frågan `fungerar chatboten?` är för grov. Retrieval, grounding, relevans, svarsstil, latens och kostnad behöver följas var för sig.

## 2026-07-18–2026-07-20 – RAGAS, modelljämförelser och grounded syntes

### Utgångsläge

Den modellfria vägen var säker men kunde ge styva svar. Vi ville undersöka om en språkmodell kunde förbättra formuleringen utan att ta över faktainnehållet.

### Ändring

En valfri LLM-syntes lades efter det extraktiva svaret. Modellen fick:

- originalfrågan
- hämtade chunkar
- det modellfria fallbacksvaret
- ett tydligt kontrakt om att inte lägga till fakta

En lokal groundingkontroll fick avgöra om omskrivningen kunde användas.

Flera modeller jämfördes mot samma svagfrågor. API-metadata utökades med modell, tokenusage, revisionsinformation, retrievalstatus och fallbackorsak.

### Resultat

I livejämförelsen den 20 juli:

- 7 av 7 baselines fångades
- 28 av 28 syntesvarianter fångades
- 13 omskrivningar användes
- 15 syntesförsök föll säkert tillbaka efter groundingkontroll

### Observation

Providerdata för tokenusage var ibland ofullständig. Saknade tokenvärden kunde inte behandlas som noll förbrukning.

### Lärdom

En språkmodell kan förbättra språket, men den måste ligga bakom ett evidenskontrakt. En säker fallback är en avsedd del av lösningen, inte bara felhantering.

## 2026-07-20–2026-07-21 – Metadata får inte bli fakta

### Utgångsläge

Vissa svar använde rubriker, filnamn eller interna beskrivningar som om de vore källinnehåll.

### Observation

Modellen kunde formulera trovärdig text utifrån metadata trots att motsvarande påstående inte fanns i själva chunktexten.

### Ändring

- Prompten skärptes så att metadata bara fick användas för navigation.
- Groundingkontrollen fick skilja mellan innehåll och källmetadata.
- Regressionstester skapades för feltypen i stället för ett enskilt facitsvar.
- Deployrevision började registreras tydligare i loggar och svar.

### Lärdom

En källa kan vara relevant utan att varje ord i dess titel är evidens. Systemet måste veta skillnaden mellan att hitta ett dokument och att ha stöd för ett påstående.

## 2026-07-22 – Design av en kontrollerad treagentskedja

### Utgångsläge

Den lokala retrievalen behövde bättre stöd för svenska synonymer, böjningar och sammansättningar. Samtidigt fick en agentisk lösning inte göra svaret friare eller svårare att verifiera.

### Beslut

En kedja med tre separata roller definierades:

1. Agent 1 skapar betydelsebevarande sökvarianter.
2. Agent 2 skriver ett svar från en begränsad evidenspool.
3. Agent 3 granskar svaret mot originalfrågan och citerad evidens.

Originalfrågan är ankare genom hela kedjan. Agent 1 får påverka retrieval men inte ändra vilken fråga Agent 2 besvarar.

### Säkerhetsregler

- strikt JSON mellan stegen
- stabila chunk-ID:n
- maximalt antal sökvarianter och evidenschunkar
- timeout och fallback per agent
- lokal validering utöver modellernas egen bedömning
- frånvaro av Agent 3-review är aldrig ett godkännande

### Lärdom

Agentic RAG behöver tydligare ansvar, inte bara fler modellrop. Varje agent ska göra en liten kontrollerbar uppgift.

## 2026-07-23 – Första liveutvärderingen av Agentic RAG

### Utgångsläge

Den lokala implementationen passerade tester, men det behövde verifieras att samma beteende fanns i den deployade HF-miljön.

### Resultat

Den lokala sviten gav:

```text
143 passed, 6 warnings
```

Live smoke-testet passerade Gradio-information, åtta scenarier och två regressionsfrågor.

### Problem

`/health` och `/ready` svarade `200`, men `/api/ask` gav `404`. Den fungerande livevägen var Gradio `/submit`, vilket inte exponerade all strukturerad metadata som behövdes för en rättvis agentutvärdering.

### Lärdom

Lokala tester kan bekräfta kontrakt, men inte den deployade produktens verkliga integrationsyta. Ett separat och stabilt API behövdes.

## 2026-07-28 – FastAPI frikopplades från Gradio

### Utgångsläge

RAG-utvärderingen var beroende av Gradio-interna endpointar och privata launch-varianter.

### Ändring

FastAPI gjordes till ägare av:

- `/api/ask`
- `/health`
- `/ready`

Gradio monterades som en separat UI-applikation på `/`.

### Resultat

Revision `3691b74b5530d300ad8d4b7832dac72c65a2226f` svarade korrekt på kontroll-endpointarna samtidigt som Gradio fortsatte fungera.

Feature flaggan kunde nu överstyras per anrop. Prioriteten blev:

1. JSON-body
2. URL-parametrar
3. miljökonfiguration

### Lärdom

Gradio är ett bra användargränssnitt men ska inte vara ägare av lösningens integrationskontrakt. Ett frikopplat API gör både testning och framtida integration enklare.

## 2026-07-28 – Första livekörningen av hela agentkedjan

### Utgångsläge

Frågan om överlämning till drift och förvaltning, Q22, användes för A/B-jämförelse mellan kontrollvägen och Agentic RAG.

### Observation

Kontrollvägen svarade snabbt utan LLM-anrop. Agent 1 kördes, men Agent 2 föll på ogiltig eller ofullständig JSON. Agent 3 kunde därför inte granska något svar.

### Problem

Det första fallbacksvaret byggdes delvis från Agent 1:s breddade retrieval. Om Agent 1 hade drivit frågan mot ett närliggande ämne kunde samma drift följa med in i fallbacken.

### Ändring

Fallbacken isolerades:

1. kör retrieval på nytt med originalfrågan
2. använd inga Agent 1-varianter
3. bygg svar och källor från den nya kandidatpoolen
4. exponera `agentic_fallback_retrieval_used`

### Lärdom

En säker fallback måste vara oberoende både av den misslyckade modelltexten och av dess retrievalunderlag.

## 2026-07-29 – Tokenkapning i Agent 2

### Observation

I nästa livekörning nådde Agent 2 exakt sin outputgräns. JSON-svaret kapades innan objektet avslutades.

### Ändring

Outputgränsen höjdes och prompten gjordes kompaktare.

### Resultat

Agent 2 kunde returnera komplett JSON, men nästa valideringsfel blev `agent2_missing_evidence`.

Kontrollvägen svarade på cirka `140,02 ms`. Agentvägen tog cirka `4 314,95 ms` och använde:

- 3 211 prompttokens
- 1 808 completiontokens
- 5 019 tokens totalt

### Lärdom

`invalid_json` var ett symtom, inte hela orsaken. Outputgränser måste följas tillsammans med exakt fallbackorsak och faktisk modelloutput.

## 2026-07-29 – Evidenskontraktet förenklades

### Utgångsläge

Agent 2 behövde returnera både evidens-ID:n, källnamn, sidor och en egen `source_coverage`. Samma information ägdes redan av applikationen.

### Observation

Det redundanta kontraktet ökade risken för schemafel utan att göra svaret säkrare.

### Ändring

- Agent 2 behöver primärt ange `chunk_id` och vilket påstående chunken stöder.
- Källnamn och sidor hämtas från applikationens auktoritativa chunk.
- `source_coverage` beräknas av systemet.
- Okända ID:n och felaktiga evidensstrukturer får separata fallbackorsaker.
- Groundingkontrollen granskar använda chunkar, inte alla språkligt närliggande retrievalträffar.

### Lärdom

Modellen ska inte duplicera metadata som systemet redan känner till. Ju mindre och tydligare kontrakt, desto lättare blir det att validera.

## 2026-07-29 – Första kompletta Agent 1–2–3-passeringen

### Resultat

Efter korrigeringen gav ett riktat end-to-end-test:

- Agent 1: `ok`
- Agent 2: `ok`
- Agent 3: `approved`
- två giltiga evidens-ID:n

Agent 3 fick bara se de chunkar som Agent 2 hade citerat.

I revision `11c7a16` tog kontrollvägen cirka `99,07 ms`. Agentvägen tog cirka `4 414,9 ms` och använde:

- 3 714 prompttokens
- 1 949 completiontokens
- 5 663 tokens totalt

Den körningen föll fortfarande tillbaka efter en för strikt groundingkontroll, men ett separat riktat anrop bekräftade att den fullständiga kedjan kunde passera.

### Lärdom

Groundingkontrollen måste vara tillräckligt strikt för att stoppa nya påståenden men tillräckligt tolerant för naturlig svensk parafras.

## 2026-07-29 – Första kompletta livekedjan

### Resultat

Revision `fe32085` gav den första kompletta livepasseringen för Q22:

- Agent 1 `ok`
- Agent 2 `ok`
- Agent 3 `approved`
- ingen agentisk fallback
- svarstid `7 468,54 ms`
- 5 596 prompttokens
- 2 585 completiontokens
- 8 181 tokens totalt

Kontrollvägen svarade på `163,9 ms` utan LLM-anrop.

### Observation

Agentsvaret var mer utvecklat men tog även med närliggande aktiviteter kring driftsättning. Kedjan fungerade tekniskt, men frågefokus och kostnad behövde följas över fler frågor.

### Lärdom

En fungerande kedja är inte automatiskt en bättre lösning. Kvalitetsvinst, latens och tokenkostnad måste jämföras samtidigt.

## 2026-07-29 – Dubbla källsektioner i GUI

### Utgångsläge

Frågan `Vilka etapper finns?` gav rätt lista men visade två sektioner med rubriken `Källor`.

### Orsak

Modellen skrev en egen källsektion i svarstexten. Därefter lade applikationen till sin klickbara och auktoritativa källista.

Saneringen kände bara igen vissa rubrikformat och missade exempelvis `**Källor**`.

### Ändring

- flera varianter av modellgenererade källrubriker tas bort
- applikationen är ensam ägare av källsektionen
- agentiska svar visar bara källor från godkända evidens-ID:n
- ociterade retrievalträffar följer inte längre med

### Resultat

Både `/api/ask` och Gradio visade de fem etapperna med en enda källsektion.

### Lärdom

Källpresentation är en systemfunktion. Modellen ska formulera svaret, inte skapa en parallell källförteckning.

## 2026-07-29 – RAGAS-körningen stoppades av HF-krediter

### Utgångsläge

Den fasta 30-frågesviten kördes mot revision `d719bbe` med `enable_agentic_rag=true`.

### Resultat

Endast Q01 nådde `approved` utan fallback. Q02–Q05 utlöste olika fallbackorsaker. Från fallbackanropet i Q05 returnerade Hugging Face `402 Payment Required`.

Q06–Q30 kunde därför inte köra den avsedda agentkedjan.

Fyra frågor hade kompletta tokenrader:

- totalt 46 903 tokens
- spann 8 448–17 891 tokens
- medel 11 725,75 tokens

Med kända delanrop i Q05 observerades minst 55 120 tokens.

### RAGAS-liknande resultat

Den deterministiska scorern kunde tekniskt poängsätta slutsvaren:

- faithfulness: `0,6277`
- answer relevance: `0,7700`
- context precision: `0,3514`
- context recall: `0,5657`

### Bedömning

Resultatet är inte en giltig mätning av Agentic RAG. 29 av 30 svar kom från fallback och merparten kördes efter providerfelet.

### Lärdom

Ett svar kan vara poängsättningsbart trots att den avsedda arkitekturen aldrig kördes. Kvalitetsmätning måste därför alltid kombineras med agentstatus, fallbackorsak, tokens och providerfel.

## 2026-07-29 – Kostnadsstyrd agentkedja

### Utgångsläge

`openai/gpt-oss-120b` användes både för Agent 2 och för en stor generell fallbacksyntes. Ett misslyckat flöde kunde därmed bli dyrare än ett godkänt flöde.

### Ändring

Kedjan gjordes om:

1. Agent 1 använder 20B.
2. Agent 2 använder 20B.
3. Agent 3 använder 20B.
4. Endast `rejected` från Agent 3 kan utlösa exakt en kompakt 120B-korrigering.
5. En misslyckad korrigering går direkt till baseline-retrieval och extraktivt svar.
6. Den tidigare stora `agentic_fallback_synthesis` används inte längre.

### Observerbarhet

API-metadata skiljer nu på:

- `review_status`
- `final_status`
- `escalation`
- fallbackorsak
- usage per agent och totalt

### Lärdom

Den stora modellen ska användas där den har ett tydligt avgränsat värde, inte som generell räddning efter varje fel.

## 2026-07-29 – Domänfrämmande frågor stoppas före Agent 2

### Utgångsläge

Frågan `Vilket bodtennis gummi är bäst?` gav ett konstruerat svar om systeminförande med irrelevanta PDF-källor.

### Orsak

- Agent 1 kunde behålla det generiska ordet `bäst` och samtidigt driva sökningen till systeminförandedomenen.
- Relevansgrinden accepterade en hög score trots att originalfrågans ämne saknade stöd.

### Ändring

- Generiska rekommendationsord räknas inte som semantisk brygga.
- Hög score räcker inte utan täckning av originalfrågans ämnesord.
- Träffar från enbart omskrivna frågor underkänns om originalfrågan saknar ämnesstöd.
- Irrelevanta frågor stoppas före Agent 2.
- Svaret får inga PDF-källor eller relaterade hemsidor.

### Resultat

Regressionstestet använder en avsiktligt driftad sökvariant med höga men irrelevanta träffar. Frågan måste ändå få det explicita källbegränsade svaret.

### Lärdom

Agentisk retrieval får förbättra recall men får aldrig omdefiniera vilket ämne användaren frågade om.

## 2026-07-29 – Tillfällig driftprofil

### Utgångsläge

Månadens inkluderade Hugging Face-krediter var slut samtidigt som den kostnadsstyrda agentkedjan ännu inte hade fått en fullständig jämförbar mätning.

### Beslut

Docker-konfigurationen sattes till:

```text
SYSTEMINFORANDE_ENABLE_AGENTIC_RAG=false
```

Den kontrollerade RAG-vägen är därmed standard för Gradio och `/api/ask` när anropet saknar override.

Den agentiska vägen finns kvar och kan väljas med:

```text
enable_agentic_rag=true
```

### Lärdom

Feature flaggan och driftstandarden har olika ansvar:

- feature flaggan väljer körväg för ett anrop
- Docker-värdet väljer den ekonomiskt säkra standarden

## 2026-08-01 – Agentic RAG fick göras om stegvis

### Utgångsläge

Agentic RAG gav ibland generiska svar och vissa anrop slutade i fallback. Samtidigt visade HF-loggarna HTTP 402 när den tidigare 120B-konfigurationen hade förbrukat tillgänglig provider-kvot.

### Observation

120B var kostnadsdrivaren i den äldre syntes-/correction-vägen. När ett 402-fel eller ett för tidigt kvalitetsstopp inträffade kunde systemet i stället ge ett extraktivt svar som började med ”Kort sagt handlar det om följande …”. Ett live-test utan 402 visade dessutom att Agent 2 själv stoppade frågan med `thin_evidence`, så Agent 3 fick aldrig granska utkastet.

### Ändring

Agentkedjan tydliggjordes och fick denna ansvarsfördelning:

1. Agent 1 – retrieval rewrite: förbättrar sökfrågan.
2. Agent 2 – evidence answer: skriver ett första, generöst utkast.
3. Agent 3 – grounded review: granskar om utkastet i huvudsak håller.
4. Agent 4 – answer correction: försöker korrigera ett underkänt svar.

120B togs bort ur correction-konfigurationen. Ett 402-fel stoppar nu anropet med ett tydligt tjänstefel i stället för fallback. Agent 2 skickar nu giltiga men osäkra utkast vidare till Agent 3. Agent 3 gjordes samtidigt mer tolerant: mindre luckor ska normalt leda till `revision` eller `approved`, medan `rejected` reserveras för helt fel ämne, centrala påhittade fakta, tydliga motsägelser eller helt saknat stöd.

### Resultat

Den första ändringen finns i commit `73eee44`, Agent 2-ändringen i `9884c73` och den tolerantare Agent 3-granskningen i `42da677`. Tester för Agent 2/3 passerade efter ändringarna. Live-testet med ”Vad är dyrast med ett systeminförande?” bekräftade att HF och 20B fungerade utan 402; felet var i stället att Agent 2 stoppade för tidigt.

### Beslut eller lärdom

Ansvar ska ligga så sent i kedjan som möjligt: Agent 1 söker, Agent 2 formulerar, Agent 3 granskar och Agent 4 korrigerar vid behov. Tekniska kvotfel ska inte skickas mellan agenter. Extraktiv fallback ska inte presenteras som ett bra agentiskt svar när Agentic RAG har misslyckats.

## Samlade lärdomar

Arbetet hittills har gett några återkommande slutsatser:

1. Retrievalproblemet kommer före promptproblemet.
2. Rubrikbaserad chunkning är en del av lösningens kvalitet, inte bara preprocessing.
3. BM25 är fortfarande användbart för ett avgränsat material med tydliga verksamhetsbegrepp.
4. Modellfri syntes är en stark säkerhets- och kostnadsbaseline.
5. En LLM-omskrivning måste granskas mot evidensen.
6. Originalfrågan måste vara ankare genom hela agentkedjan.
7. Fallback-retrieval får inte ärva en misslyckad agents ämnesdrift.
8. Modellen ska referera till chunk-ID:n; applikationen ska äga källmetadata.
9. Kvalitetspoäng utan drifttelemetri kan ge en falsk bild av arkitekturen.
10. En fungerande agentkedja måste motivera sin extra latens och kostnad med mätbar kvalitetsvinst.

## Relaterade dokument och artefakter

- [RAG Solution](./rag-solution.html) – aktuell lösningsbeskrivning
- [Agentic RAG contracts and architecture](./agentic-rag-contracts.md) – ursprungligt designkontrakt
- [RAGAS evaluation proposal](./ragas-evaluation-proposal.md)
- [RAGAS HF evaluation spec](./ragas-hf-evaluation-spec.md)
- [Weak RAG regression notes](./WEAK_RAG_REGRESSION_NOTES.md)
- `tests/results/` – captures, jämförelser och mätresultat
