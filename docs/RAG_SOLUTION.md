# RAG Solution

**Lösningen bakom chatboten för systeminförande**

**Författare**

Tomas Helmfridsson och OpenAI Codex

**Senast uppdaterad**

2026-08-03

**Publik chatbot**

https://www.systeminforande.se/chatt-bot

Detta dokument beskriver hur den aktuella RAG-lösningen fungerar. Fokus ligger på systemets uppbyggnad, hur en fråga behandlas, hur svar hålls förankrade i källmaterialet och hur lösningen kan användas via Gradio och API.

Utvecklingshistorik, misslyckade försök och mätobservationer finns i [RAG Lab Journal](./rag-lab-lessons.html).

## 1. Syfte

Chatboten hjälper användaren att hitta och förstå information om systeminförande. Den använder ett avgränsat material av PDF-dokument och utvalda webbsidor som kunskapskälla.

Lösningen ska:

- hitta relevanta avsnitt i källmaterialet
- svara direkt på användarens fråga
- hålla svaret inom det underlag som faktiskt hittats
- visa en tydlig och klickbar källista
- avstå från att hitta på ett svar när materialet inte räcker
- kunna utvärderas och felsökas med strukturerad metadata

RAG står för Retrieval-Augmented Generation. I den här lösningen betyder det att systemet först hämtar relevanta textutdrag och därefter bygger ett svar från just dessa utdrag.

## 2. Lösningen i korthet

Den normala kedjan kan sammanfattas så här:

```text
PDF-filer och webbsidor
        ↓
Textextraktion och rubrikbaserad chunkning
        ↓
Lokalt index i rag/data/chunks.json
        ↓
Användarfråga
        ↓
Normalisering, frågetyp och retrieval
        ↓
Relevans- och groundingkontroll
        ↓
Källgrundat svar
        ↓
En auktoritativ lista med källor och relaterade hemsidor
```

Lösningen har två valbara svarsvägar:

- en kontrollerad RAG-väg med lokal retrieval och modellfri eller valfri grounded syntes
- en agentisk RAG-väg där tre avgränsade agenter hjälper till med sökning, svar och verifiering

Feature flaggan `enable_agentic_rag` avgör vilken väg som används. Agentkedjan är standard i Docker/Hugging Face genom `SYSTEMINFORANDE_ENABLE_AGENTIC_RAG=true`. Den äldre kontrollerade vägen finns kvar för rollback och reproducerbara jämförelser.

## 3. Källmaterial och indexering

### 3.1 Källor

Kunskapsunderlaget består huvudsakligen av PDF-filer i `docs/pdfs/`. Det kan även innehålla uttryckligen tillåtna webbsidor. Källorna handlar bland annat om:

- arbetsmodell och projektstyrning
- införandekrav
- arbetsområden
- acceptanstest
- driftförberedelser och driftsättning
- utbildning, konvertering och dokumentation
- checklistor och mallar

Systemet filtrerar bort källtyper som inte är godkända för användarvända svar.

### 3.2 Textextraktion

PDF-filerna läses sida för sida. Återkommande sidhuvuden, sidfötter, sidnummer och andra typiska dokumentartefakter rensas bort innan texten delas upp.

För varje chunk sparas bland annat:

- ett stabilt ID
- källfil
- källtyp
- rubrik och avsnitt
- sidnummer
- textinnehåll

### 3.3 Chunkning

Texten delas huvudsakligen vid dokumentens rubriker och avsnitt, inte med ett blint fast teckenfönster. Det gör att en chunk oftare motsvarar en sammanhängande del av dokumentet.

Rubrikbaserad chunkning ger två viktiga fördelar:

- retrieval kan väga in både rubrik och brödtext
- källhänvisningen kan peka på ett begripligt avsnitt och rätt sidor

Indexet byggs med `tools/build_rag_index.py` och sparas i `rag/data/chunks.json`.

## 4. Retrieval

Retrieval är den del som väljer vilka chunkar som får användas som svarunderlag.

### 4.1 Normalisering

Frågan delas upp i söktermer. Svenska tecken normaliseras för jämförelse och vanliga funktionsord filtreras bort. Systemet hanterar även:

- enklare svenska böjningar
- kända stavvarianter och mindre stavfel
- sammansättningar
- kontrollerade synonymer

Originalfrågan sparas alltid och är ankaret för slutsvaret.

### 4.2 BM25

Grundrankningen använder BM25. Metoden värderar hur väl frågans termer matchar varje chunk och tar hänsyn till hur ovanlig en term är i hela materialet.

BM25 passar lösningen eftersom:

- materialet innehåller tydliga verksamhetsbegrepp
- många frågor använder ord som också förekommer i dokumenten
- rankningen kan köras snabbt och lokalt
- resultatet är relativt lätt att förklara och felsöka

### 4.3 Boostar och frågetyp

BM25 kompletteras med generella signaler. Exempel är träff i rubrik, källnamn, dokumentfamilj och avsnittsnivå.

Frågan klassificeras också i en enkel frågetyp:

- definition
- syfte
- lista eller översiktslista
- process
- tidpunkt eller beslut
- allmän fråga

Frågetypen hjälper rankningen och den modellfria svargenereringen att välja en lämplig form. En listfråga kan exempelvis besvaras kort, medan en processfråga behöver ett mer sammanhängande förlopp.

### 4.4 Lokal reranking

De främsta BM25-kandidaterna rankas en andra gång med en lokal textlikhetssignal. BM25 väger tyngst och likhetssignalen används för att förbättra ordningen inom kandidatpoolen.

### 4.5 Relevansgrind

En hög retrievalscore betyder inte automatiskt att materialet kan besvara frågan. Innan svargenerering kontrollerar systemet att träffarna faktiskt täcker originalfrågans meningsbärande ämnesord.

Generiska ord som `bäst`, `bra` och `rekommendation` räcker inte som ämnesstöd. En domänfrämmande fråga, exempelvis om bordtennisutrustning, ska därför inte kunna få ett konstruerat svar om systeminförande bara för att en omskriven sökfråga råkar hitta höga träffar.

Om relevansgrinden underkänner underlaget returneras ett tydligt källbegränsat svar utan PDF-källor eller relaterade hemsidor.

## 5. Den kontrollerade RAG-vägen

När Agentic RAG är avstängd används lokal retrieval följd av en kontrollerad svarsgenerator.

### 5.1 Modellfri syntes

Den modellfria syntesen bygger svaret med egen kod. Den väljer och sammanfogar meningar från de hämtade chunkarna och anpassar formen efter frågetypen.

Fördelarna är:

- ingen extern modellkostnad
- låg latens
- reproducerbart beteende
- liten risk att lägga till nya sakpåståenden

Nackdelen är att svaret ibland kan bli mer mekaniskt eller mindre resonerande än ett välgrundat LLM-svar.

### 5.2 Valfri LLM-syntes

Den äldre vägen kan även använda en extern modell för att skriva om det extraktiva svaret till naturligare svenska. Omskrivningen får bara använda de hämtade chunkarna och det modellfria svaret.

Efter omskrivningen körs en lokal groundingkontroll. Om modellen lägger till innehåll som saknar stöd används det modellfria svaret i stället.

Syntesen styrs med:

- `SYSTEMINFORANDE_ENABLE_LLM_SYNTHESIS`
- `SYSTEMINFORANDE_LLM_SYNTHESIS_MODEL`
- `enable_synthesis` per API-anrop

Den kostnadseffektiva standardmodellen för denna syntesväg är `openai/gpt-oss-20b`.

## 6. Agentic RAG

Den agentiska vägen är en kontrollerad kedja med separata ansvarsområden. Agenterna får inte fritt diskutera sig fram till ett svar; varje steg har ett strukturerat kontrakt och lokala kontroller.

### 6.1 Agent 1 – retrieval rewrite

Agent 1 får inte svara på användarens fråga. Den skapar upp till fem betydelsebevarande sökvarianter, exempelvis:

- böjningsformer
- synonymer
- svenska sammansättningar
- sannolikt ordval i källdokumenten

Originalfrågan körs alltid som den första sökningen. Varianter som ser ut som svar eller driver till ett annat ämne avvisas.

### 6.2 Sammanfogad retrieval

Lokal retrieval körs för de accepterade sökvarianterna. Resultaten dedupliceras och slås samman till en kandidatlista.

En agentisk sökvariant får inte ensam bevisa att frågan hör till materialet. Originalfrågans ämne måste fortfarande ha stöd i träffarna.

### 6.3 Agent 2 – evidensbaserat svar

Agent 2 får:

- originalfrågan
- kompakt rewrite-metadata
- högst ett begränsat antal chunkar med stabila ID:n

Agenten ska svara på originalfrågan och returnera strikt JSON. Varje central del av svaret ska kopplas till ett känt chunk-ID. Applikationen, inte modellen, är ägare av källnamn och sidmetadata.

### 6.4 Agent 3 – verifiering

Agent 3 granskar:

- om svaret besvarar originalfrågan
- om påståendena stöds av citerade chunkar
- om svaret har glidit mot en omskriven sökfråga
- om intern metadata har läckt in som fakta

Agent 3 ser bara de chunkar som Agent 2 faktiskt har citerat. Resultatet är `approved`, `revision` eller `rejected`.

En mindre källstödd revision kan publiceras direkt. Ett godkänt svar visar endast de källor vars chunk-ID har godkänts.

### 6.5 Begränsad korrigering och fallback

Agent 1, Agent 2 och Agent 3 använder normalt `openai/gpt-oss-20b`.

Om Agent 3 returnerar `rejected` får systemet göra exakt ett kompakt korrigeringsanrop med `openai/gpt-oss-20b`. Detta är Agent 4:s enda uppgift: att förbättra ett svar som redan har hämtats och granskats, inte att vara en generell reservväg. Korrigeringen får samma originalfråga, det underkända svaret, granskningsorsaken och den auktoritativa evidensen. `openai/gpt-oss-120b` tillåts inte som korrigeringsmodell; ett gammalt sådant miljövärde ersätts automatiskt med 20B-standarden.

Om korrigeringen inte passerar kontraktet görs inga fler LLM-anrop. Systemet kör då retrieval på nytt med endast originalfrågan och returnerar ett säkert extraktivt svar.

### 6.6 Outputgränser per agent

Varje agent har ett maximalt antal outputtokens för ett enskilt modellanrop:

| Agent | Uppgift | Max outputtokens |
| --- | --- | ---: |
| Agent 1 | Retrieval rewrite | 2 400 |
| Agent 2 | Evidensbaserat svar | 3 600 |
| Agent 3 | Grounding- och frågefokusgranskning | 2 000 |
| Agent 4 | En korrigering efter ett avslag | 2 000 |

Gränserna gäller modellens genererade output, inte prompten eller den totala tokenförbrukningen. De är tak och reserverar inte tokens; ett normalt svar kan därför bli betydligt kortare.

Taken finns för att begränsa kostnad och svarstid, stoppa okontrollerat långa modellutdata och göra agentkedjans drift mer förutsägbar. Samtidigt måste de lämna tillräckligt utrymme för agenternas JSON-kontrakt. Om ett svar kapas mitt i ett JSON-objekt kan kontraktet inte valideras. För Agent 3 markeras detta som `unavailable`, varefter Agent 2:s utkast behålls utan en lyckad review.

Ett modellanrop med `status="ok"` betyder bara att leverantören returnerade ett svar. Det betyder inte att svaret passerade det lokala JSON-kontraktet. Om `completion_tokens` upprepade gånger är exakt lika med agentens outputgräns samtidigt som `invalid_json`, schemafel eller `unavailable` förekommer ska tokenkapning misstänkas. Prompt och kontrakt bör då först göras så kompakta som möjligt; gränsen kan höjas när agenten fortfarande behöver mer legitimt utrymme.

## 7. Källor och transparens

Applikationen äger den användarvända källistan. Modellgenererade källrubriker och källmetadata tas bort från svarstexten innan den riktiga källsektionen läggs till.

Det ger en enda auktoritativ sektion:

- `Källor` med klickbara PDF-länkar
- `Relaterade hemsidor` när en godkänd koppling finns

På den agentiska vägen begränsas källistan till Agent 2:s och Agent 3:s validerade evidens-ID:n. På fallbackvägen byggs den från den nya retrievalen på originalfrågan.

## 8. API och Gradio

### 8.1 Gradio

Gradio är det publika användargränssnittet. Det använder samma centrala frågerouter som API:t.

Agentic RAG kan överstyras i GUI-adressen:

```text
?enable_agentic_rag=true
?enable_agentic_rag=false
```

Debug kan styras på motsvarande sätt med `debug=true` eller `debug=false`.

### 8.2 `/api/ask`

`POST /api/ask` är den strukturerade integrations- och utvärderingsytan. Ett grundanrop innehåller:

```json
{
  "question": "Vilka etapper finns?",
  "debug_mode": false,
  "enable_agentic_rag": false
}
```

API:t returnerar bland annat:

- det formaterade svaret
- route och vald modell
- källor och relaterade hemsidor
- retrievalresultat och relevansstatus
- agentstatus och fallbackorsak
- tokenusage och svarstid

Feature flags kan anges både i JSON-body och URL. Prioritetsordningen är:

1. JSON-body
2. URL-parametrar
3. miljökonfiguration

Ogiltiga booleska URL-värden returnerar `400 Bad Request`.

### 8.3 Driftstatus

Följande endpointar används för kontroll:

- `/health` visar att processen svarar samt deployrevision och viktiga driftvärden
- `/ready` visar att applikationen är redo att ta emot frågor

FastAPI äger dessa endpointar och `/api/ask`. Gradio är monterat som användargränssnitt på samma applikation.

## 9. Feature flags och modeller

Viktigaste miljövariablerna är:

```text
SYSTEMINFORANDE_ENABLE_AGENTIC_RAG=true
SYSTEMINFORANDE_ENABLE_LLM_SYNTHESIS=false
SYSTEMINFORANDE_LLM_SYNTHESIS_MODEL=openai/gpt-oss-20b
SYSTEMINFORANDE_AGENT1_MODEL=openai/gpt-oss-20b
SYSTEMINFORANDE_AGENT2_MODEL=openai/gpt-oss-20b
SYSTEMINFORANDE_AGENT3_MODEL=openai/gpt-oss-20b
SYSTEMINFORANDE_AGENT_CORRECTION_MODEL=openai/gpt-oss-20b
```

Temperatur styrs per agentens uppgift. Agent 1 (retrieval rewrite) och Agent 3 (verifiering) körs med `temperature=0.0` för stabilare struktur och bedömning. Agent 2 (svar) och Agent 4 (korrigering) körs med `temperature=0.2` för viss språklig flexibilitet. Alla agentväxlingar valideras genom gemensamma kontrakt i `rag/agent_contracts.py`; formatfel ska inte i sig utlösa en kvalitetsmässig fallback.

Agenternas aktuella outputgränser och motivet till dem beskrivs i avsnitt 6.6. Gränserna är kodkonfiguration i `app.py` och ska följas upp med `completion_tokens`, agentstatus och fallbackorsak i loggarna.

Docker-konfigurationen sätter Agentic RAG till `true`, vilket gör agentkedjan till standard i den deployade miljön. En miljövariabel som konfigureras direkt i Hugging Face Space överstyr Docker-värdet och måste därför också vara `true` eller tas bort inför deployen.

Feature flaggan ändrar körväg men tar inte bort någon funktion. Samma deploy kan därför användas för kontrollerade jämförelser med `enable_agentic_rag=false` och `enable_agentic_rag=true`.

## 10. Observerbarhet

Varje strukturerat API-svar kan innehålla information om:

- deployrevision
- frågetyp och söktermer
- valda retrievalresultat
- om relevansgrinden godkände underlaget
- vilka agentsteg som kördes
- agenternas modeller och status
- `review_status`
- `final_status`
- eventuellt korrigeringssteg
- fallbackorsak
- tokens per anrop och totalt
- total svarstid

`final_status` visar om agentsvaret blev:

- `approved`
- `revised`
- `corrected`
- `fallback`

Denna metadata gör det möjligt att skilja svarskvalitet från driftproblem. Ett välformulerat fallback-svar ska exempelvis inte räknas som en lyckad agentkörning.

## 11. Test och utvärdering

Lösningen verifieras på flera nivåer.

### 11.1 Enhets- och regressionstester

Den lokala testsviten kontrollerar bland annat:

- chunkning och retrieval
- svenska språkvariationer och stavfel
- frågetypsstyrda svar
- grounding och metadatahygien
- agenternas JSON-kontrakt
- fallback vid timeout eller ogiltig modelloutput
- feature flags i API och Gradio
- att domänfrämmande frågor inte får konstruerade svar
- att endast en källsektion visas

### 11.2 Live-tester

Smoke-tester körs mot den deployade Hugging Face-applikationen. De verifierar att rätt revision körs och att `/health`, `/ready`, `/api/ask` och Gradio fungerar tillsammans.

### 11.3 RAGAS

Den fasta frågesviten kan bedömas med fyra RAGAS-relaterade kvalitetsmått:

- faithfulness
- answer relevance
- context precision
- context recall

Mätningen kompletteras med:

- latens
- tokenförbrukning
- fallbackfrekvens
- andel godkända agentkörningar
- manuell bedömning av tydlighet och användbarhet

RAGAS-resultat ska bara jämföras när körningarna verkligen har använt den avsedda arkitekturen. Fallback-svar och providerfel måste redovisas separat.

#### Senaste Agentic RAG-baslinje

Efter deploy av temperaturändringen kördes den fasta sviten med 30 frågor mot Hugging Face. Den offline-deterministiska RAGAS-aligned mätningen gav:

| Mått | Resultat |
|---|---:|
| Faithfulness | 0,5177 |
| Answer relevance | 0,8423 |
| Context precision | 0,3396 |
| Context recall | 0,5780 |

Körningen använde 236 293 tokens, i genomsnitt 7 876 tokens per fråga, och hade 91 lyckade LLM-anrop. En fråga gick till fallback och Agent 4 användes endast en gång. Resultatet är en jämförbar teknisk baseline; det är inte en garanti för användbarhet eller språkkvalitet. Därför kompletteras den med Human-in-the-Loop-granskning av direkthet, svenska, slutsatskvalitet, källstöd och korrekt avstående vid domänfrämmande frågor.

## 12. Deployment

Koden ligger i GitHub. Ett GitHub Actions-flöde deployar den aktuella revisionen till Hugging Face Space.

Deploykedjan är:

```text
Lokal ändring
    ↓
Git commit och push till GitHub
    ↓
GitHub Actions
    ↓
Hugging Face Space bygger Docker-imagen
    ↓
Health- och ready-kontroll
    ↓
Live-verifiering via API och Gradio
```

Git LFS används för PDF-filerna. GitHub Pages publicerar PDF-källorna och dokumentationssidorna så att chatbotens källänkar är öppna och klickbara.

## 13. Begränsningar

Lösningen har några medvetna begränsningar:

- BM25 och den lokala rerankern bygger främst på lexikal likhet och fångar inte alla semantiska relationer.
- Modellfri syntes är säker och snabb men kan bli mindre naturlig än ett bra LLM-svar.
- LLM- och agentvägen är långsammare och förbrukar externa inference-credits.
- Kvaliteten är beroende av PDF-filers rubrikstruktur och extraherbara text.
- Ett kort korrekt svar är ibland avsiktligt; en bredare förklaring kräver en mer detaljerad fråga.
- Systemet kan bara svara utifrån det material som har indexerats.

## 14. Sammanfattning

RAG-lösningen är byggd kring en enkel princip: retrieval och källkontroll kommer före fri språkmodellsgenerering.

Den kontrollerade vägen ger snabba och reproducerbara svar. Den agentiska vägen kan förbättra sökning och formulering, men får bara publicera svar som passerar strukturerade evidens- och groundingkontroller. När någon kontroll faller används ett säkert svar från originalfrågan och det lokala källmaterialet.

Resultatet är en lösning där svar, källor, agentbeslut, tokenförbrukning och fallback går att följa och utvärdera var för sig.
