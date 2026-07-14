# RAG Lab Lessons

**Författare**
Tomas Helmfridsson och OpenAI Codex

**Datum**
2026-07-10

**Framtagning**
Detta dokument är framtaget med AI-assisterat stöd. Struktur, analys, testobservationer och textbearbetning har tagits fram i samarbete mellan människa och AI.

**Publik länk till chatboten**
https://www.systeminforande.se/chatt-bot

Detta dokument sammanfattar labben bakom chatboten för systeminförande. Fokus ligger på hur vi gick från ett antal PDF-filer till en sökbar applikation med både modellfri syntes och LLM-baserade svar, vilka tekniska val som gjordes och vad vi lärde oss av dem.

Syftet med dokumentet är inte bara att beskriva slutresultatet, utan att förklara själva RAG-resan:

- hur källmaterial extraheras
- hur chunkning påverkar kvaliteten
- hur retrieval fungerar
- vad som skiljer modellfri syntes från LLM-generering
- vilka förbättringar vi har testat och varför

## Innehållsförteckning

1. Målbild
2. Övergripande arkitektur
3. Vad RAG betyder i denna lösning
4. Varför Hugging Face och Gradio valdes
5. Från PDF till sökbart index
6. Chunkning
7. Retrieval
8. BM25
9. Tokenisering, normalisering och stopwords
10. Heuristiker och boostar
11. Query intent
12. Temperaturspåret och vad det faktiskt påverkade
13. Alternativ till BM25
14. Strukturerad modellfri syntes
15. LLM-baserad syntes
16. Modellval och praktiska LLM-erfarenheter
17. Källor och transparens
18. Deployment och drift
19. Viktigaste lärdomar
20. Nästa rimliga förbättringar
21. Sammanfattning
22. När vi upptäckte att RAG:en inte höll måttet

## 1. Målbild

Målet var att bygga en praktiskt användbar chatbot för material om systeminförande, men också att använda arbetet som en lärlabb för RAG.

Två mål löpte därför parallellt:

- skapa en fungerande applikation
- förstå och dokumentera vad som faktiskt påverkar svarskvaliteten

Det blev snabbt tydligt att den stora utmaningen inte bara var att "koppla in en LLM", utan att få hela kedjan att fungera:

1. källmaterialet måste gå att extrahera
2. texten måste delas upp på ett bra sätt
3. retrieval måste hitta rätt avsnitt
4. svaret måste byggas försiktigt utifrån underlaget

## 2. Övergripande arkitektur

Lösningen kan beskrivas som en enkel men tydlig RAG-pipeline:

1. PDF-filer läses in från `docs/pdfs/`
2. text extraheras sida för sida
3. texten delas upp i chunkar
4. chunkarna sparas i ett lokalt index i `rag/data/chunks.json`
5. en fråga tokeniseras och matchas mot chunkarna
6. toppresultaten används som underlag för ett svar
7. svaret byggs antingen modellfritt eller med extern LLM
8. källor visas som klickbara länkar till GitHub Pages

Det viktiga här är att RAG inte är en enskild funktion, utan en kedja där varje steg påverkar nästa steg.

## 3. Vad RAG betyder i denna lösning

RAG står för Retrieval-Augmented Generation.

I denna lösning betyder det:

- `Retrieval`: hitta relevanta chunkar från det lokala dokumentmaterialet
- `Augmented`: använda dessa chunkar som extra kontext
- `Generation`: formulera ett svar baserat på chunkarna

Det finns dock två typer av generation i lösningen:

- `modellfri syntes`
- `LLM-baserad generering`

Det är en viktig skillnad. Generation behöver inte alltid betyda att en extern språkmodell skriver svaret. I vår strukturerade väg genereras texten av egen kod genom regelstyrd extraktion och omskrivning.

## 4. Varför Hugging Face och Gradio valdes

### Hugging Face

Hugging Face valdes som driftmiljö därför att det gav:

- enkel hosting av en Python-app
- direkt stöd för Gradio
- möjlighet att prova flera externa modeller via samma ekosystem
- enkel koppling från GitHub med GitHub Actions

Det gjorde plattformen lämplig för en experimentell RAG-labb där modellstöd, prompting och svarskvalitet behövde testas iterativt.

### Gradio

Gradio valdes därför att det gav:

- snabb GUI-utveckling i Python
- enkel koppling mellan backendlogik och UI-komponenter
- bra stöd för att streama eller successivt uppdatera svar
- enkel deploy i Hugging Face Spaces

Nackdelen är att det är svårare att detaljstyra UI än i ett separat frontend-ramverk, men för denna labb vägde utvecklingshastigheten tyngre.

## 5. Från PDF till sökbart index

Kärnan i lösningen börjar i `rag/ingest.py`.

Ingest betyder här steget där rått källmaterial görs om till ett internt format som resten av systemet kan arbeta med.

I praktiken gör ingest detta:

- läser alla PDF-filer
- extraherar text per sida med PyMuPDF (`fitz`)
- filtrerar bort brus
- delar upp texten i chunkar
- sparar chunkarna med metadata som titel, sektion, sidnummer och källa

En viktig lärdom är att ingest inte bara är "förarbete". Det är själva grunden för retrievalen. Om chunkarna blir dåliga kommer även den bästa LLM:n att få dåligt underlag.

## 6. Chunkning

### Vad chunkning är

Chunkning betyder att man delar upp dokument i mindre textblock som går att söka i.

Ett chunk måste vara:

- tillräckligt litet för att vara träffsäkert
- tillräckligt stort för att behålla sammanhang
- tillräckligt rent från brus för att inte lura retrievalen

Detta är ett av de viktigaste stegen i en RAG-lösning.

### Hur chunkning fungerar hos oss

I denna lösning är chunkningen regelbaserad och strukturdriven. Den görs inte med LLM och den görs inte med embeddings.

Det betyder:

- ingen språkmodell används för att sammanfatta eller dela upp texten
- inga vektorer används för att avgöra chunkgränser
- chunkgränserna bestäms av dokumentstruktur, främst rubriker och sektioner

Kodmässigt bygger detta på:

- rubrikigenkänning med regex som `SECTION_RE`
- stöd för rubriker som ligger på två rader, exempelvis först ett sektionsnummer och sedan själva rubriken
- filtrering av innehållsförteckningar via `TOC_RE`
- filtrering av headers, footers, datum och sidartefakter

Detta gör chunkningen mer deterministisk och lättare att förstå.

### Vad vi har förbättrat i chunkningen

Under arbetet förbättrades chunkningen stegvis för att minska brus:

- innehållsförteckningar filtrerades bort
- vanliga header- och footer-rader filtrerades bort
- mycket korta eller låg-informativa chunkar filtrerades bort
- rubriker i olika format började kännas igen bättre

Detta var viktigt därför att retrieval annars drog upp irrelevanta avsnitt som såg viktiga ut bara för att de innehöll rubriker, sidnummer eller upprepade dokumentord.

### Chunkstorlek och hur fint vi delar upp texten

När man talar om chunkstorlek i RAG menar man ofta antal tecken, ord eller tokens per chunk. Den typen av fast fönsterchunkning används inte i den nuvarande implementationen.

Det vi i praktiken har arbetat med är i stället hur fint eller grovt vi delar upp texten:

- hur bred en sektion får vara
- hur mycket struktur som ska bevaras
- hur mycket brus som ska rensas bort innan sektionen sparas

Det är alltså mer korrekt att säga att vi har experimenterat med hur fin- eller grovkornig chunkningen ska vara, snarare än att vi har provat ett stort antal tokenstorlekar.

### Varför vi inte använder LLM för chunkning

Det finns lösningar där en LLM används för att:

- identifiera semantiska gränser
- skapa bättre stycken
- märka upp innehållstyper

Vi valde bort detta här eftersom det hade gjort pipelinen:

- dyrare
- långsammare
- mindre transparent
- svårare att felsöka

För en lärlabb var det bättre att hålla chunkningen lokal, enkel och reproducerbar.

### Skillnaden mellan chunkning och embeddings

Det är lätt att blanda ihop dessa.

- `Chunkning` avgör hur dokumentet delas upp
- `Embeddings` avgör hur chunkar eller frågor representeras numeriskt för semantisk sökning

Man kan alltså ha:

- chunkning utan embeddings
- embeddings utan avancerad chunkning
- eller båda tillsammans

I vår nuvarande lösning har vi chunkning, men ingen embedding-baserad retrieval.

## 7. Retrieval

Retrieval-logiken ligger i `rag/search.py`.

Det är retrievalen som bestämmer vilka chunkar som ska få svara på användarens fråga. Om retrievalen missar rätt chunkar hjälper det sällan att prompten är bra.

### Vad retrieval gör

När en fråga kommer in sker detta i stora drag:

1. frågan tokeniseras
2. orden normaliseras
3. stopwords filtreras bort
4. varje chunk får en poäng
5. chunkarna sorteras
6. toppresultaten skickas vidare till syntes eller LLM

### Vad vi hade före den nuvarande retrievalen

Tidigt var retrievalen enklare och mindre domänstyrd. Det ledde oftare till att frågor om till exempel etapper, arbetsområden eller införandekrav drog upp alltför generella eller för detaljerade chunkar.

Det nuvarande läget är betydligt mer styrt:

- BM25-liknande lexikal poängsättning
- titelmatchning
- domänspecifika boostar
- intentbaserade justeringar

Detta var ett medvetet steg bort från en mer naiv ordmatchning.

## 8. BM25

### Vad BM25 är

BM25 är en klassisk metod för textretrieval. Den försöker svara på frågan:

"Hur relevant är ett dokument för denna fråga, givet vilka ord som förekommer och hur ofta?"

BM25 tar bland annat hänsyn till:

- om frågeordet finns i dokumentet
- hur ofta ordet förekommer i dokumentet
- hur vanligt ordet är i hela dokumentmängden
- hur långt dokumentet är

Det sista är viktigt. En lång text får inte automatiskt vinna bara för att den innehåller många ord.

### Varför BM25 passade här

BM25 var ett bra val i denna labb därför att det är:

- lättare att förstå än embeddingsökning
- snabbt
- billigt
- transparent
- tillräckligt starkt för en begränsad dokumentmängd inom samma domän

Det var också ett bra pedagogiskt val eftersom man tydligt kan se varför ett dokument får poäng.

### Vad BM25 inte gör

BM25 förstår inte betydelse på samma sätt som en embedding-modell. Den hittar främst lexikala överlapp:

- samma ord
- liknande ordstammar
- ord i rubriker och text

Det gör att BM25 är starkt när dokument och frågor använder samma språk, men svagare när användaren uttrycker sig med helt andra ord än källmaterialet.

## 9. Tokenisering, normalisering och stopwords

En stor del av retrievalkvaliteten kommer från små men viktiga detaljbeslut.

I vår sökning sker bland annat:

- tokenisering med regex
- enkel normalisering av svenska ändelser
- bortfiltrering av vanliga ord som inte bär mycket ämnesinformation

Exempel på vad normaliseringen försöker göra:

- minska skillnader mellan singular och plural
- minska effekten av böjningsändelser

Detta är ingen fullständig svensk stemming eller lemmatisering. Det är en enkel heuristisk normalisering. För just detta källmaterial räckte det för att förbättra träffbilden utan att dra in tyngre språkverktyg.

## 10. Heuristiker och boostar

Ovanpå BM25 har vi lagt flera heuristiska förstärkningar.

### Vad en boost är

En boost betyder att man manuellt lägger till eller drar ifrån poäng för vissa typer av träffar.

Det betyder att retrievalen inte bara säger:

- "ordet finns här"

utan också:

- "den här träffen borde vara extra intressant i just denna domän"

### Vilka boostar vi använder

Nuvarande retrieval använder bland annat:

- `TITLE_BOOST`
  Chunkar vars titel delar ord med frågan får extra poäng.
- `DEFINITION_TITLE_BOOST`
  Definitionsfrågor som börjar med till exempel "Vad är ..." får extra stöd från rubriker som liknar inledning, syfte eller arbetsområde.
- `DOMAIN_RULES`
  Frågor om arbetsområden, införandekrav, etapper och planering kopplas starkare till vissa dokument och vissa rubriktyper.
- `OVERVIEW_SECTION_BOOST`
  Översiktliga sektioner kan lyftas upp.
- `DETAIL_SECTION_PENALTY`
  För vissa översiktsfrågor kan detaljsektioner tryckas ned.

### Vad heuristik betyder här

Heuristik betyder i detta sammanhang tumregler baserade på observationer om vårt källmaterial.

Exempel:

- om frågan gäller arbetsområden är vissa checklistedokument nästan alltid mer relevanta
- om frågan är en definition är rubriker som syfte, inledning och modell ofta bättre än detaljavsnitt
- om frågan söker en överblick bör inte sektion 4.x med mycket detaljinnehåll dominera för hårt

Heuristiker är alltså inte "fusk", utan ett sätt att anpassa en generell retrievalmetod till en specifik domän.

### Nackdelen med heuristiker

Heuristiker är kraftfulla men har en baksida:

- de kan bli för hårdkodade
- de kan fungera bra på dagens dokument men sämre på nytt material
- de kräver underhåll när dokumentunderlaget förändras

Det är därför viktigt att se dem som domänanpassning, inte som en universell lösning.

## 11. Query intent

Retrievalen försöker också klassificera frågans avsikt, exempelvis:

- definition
- syfte
- lista
- översiktslista
- process
- timing eller beslut

Detta används för att förändra rankningen.

Exempel:

- en fråga som börjar med "Vad är ..." behandlas annorlunda än en fråga som börjar med "Vilka ..."
- en processfråga ska ofta hitta andra chunkar än en definitionsfråga

Detta är en enkel form av query understanding utan att använda separat NLP-modell.

## 12. Temperaturspåret och vad det faktiskt påverkade

Temperatur är en viktig parameter i LLM-generering, men den påverkar inte retrievalsteget.

Det är en viktig distinktion:

- `retrieval` avgör vilka chunkar som väljs
- `temperature` avgör hur fritt eller konservativt en LLM formulerar sitt svar

En högre temperatur kan ge:

- mer variation
- mer kreativitet
- högre risk för utsvävning eller hallucination

En lägre temperatur kan ge:

- mer stabila svar
- mindre variation
- något torrare språk

I den nuvarande lösningen är `temperature=0.2` i `llm/reasoning.py`, vilket är ett medvetet konservativt val.

Det betyder att vi i nuläget prioriterar:

- trohet mot källmaterialet
- stabilitet
- mindre risk för att modellen hittar på

Temperaturförändringar kan alltså förbättra formuleringen, men de kan inte reparera dålig retrieval. Om fel chunkar hämtas blir svaret dåligt oavsett temperatur.

## 13. Alternativ till BM25

BM25 är inte det enda retrievalalternativet. Några vanliga alternativ är:

### Embedding-baserad retrieval

Här omvandlas frågor och chunkar till vektorer och jämförs semantiskt.

Fördelar:

- kan hitta relevant innehåll även när samma ord inte används
- bättre på parafraser och betydelsenära uttryck

Nackdelar:

- mindre transparent
- kräver embedding-modell
- ofta behov av vektorindex eller vektordatabas

### Hybrid retrieval

Här kombineras BM25 och embeddings.

Fördelar:

- både exakta ordträffar och semantisk träffsäkerhet

Nackdelar:

- mer komplexitet
- fler parametrar att justera

### Reranking

Först hämtas exempelvis 20 kandidater med BM25 eller hybrid retrieval. Sedan får en starkare modell rangordna dessa bättre.

Fördelar:

- högre precision i toppen

Nackdelar:

- extra latens
- extra modellberoende

### Full LLM retrievalhjälp

I vissa system används LLM även för att:

- skriva om frågor
- bryta ned frågor i delmoment
- välja källor

Det kan fungera bra, men gör systemet mer komplext och mindre reproducerbart.

### Varför vi inte gick dit nu

För denna labb var BM25 med domänanpassning ett rimligt mellanläge:

- tillräckligt starkt för att ge lärdomar
- lätt att felsöka
- billigt att köra
- lätt att förklara

## 14. Strukturerad modellfri syntes

Den strukturerade vägen ligger i `rag/extractive.py`.

Detta är viktigt att förstå korrekt:

- den använder inte extern LLM
- den använder inte en intern språkmodell för att skriva text
- den bygger svar med egen Python-logik

### Hur den fungerar

I stora drag gör den detta:

1. tittar på toppchunkarna från retrieval
2. delar upp texten i meningar
3. filtrerar bort metadata och brus
4. poängsätter meningar utifrån frågeord och vissa positiva markörer
5. väljer de bästa meningarna
6. skriver om dem med enklare omskrivningsregler
7. bygger ihop ett försiktigt resonemang

Det är därför rätt att kalla detta för:

- extraktiv syntes
- modellfri syntes
- regelbaserad sammanställning

men inte för LLM-generering.

### Använder den språkmodell för språket?

Nej, inte i nuvarande implementation.

Den närmar sig ett "skrivet språk" genom:

- omskrivningsregler
- introduktionsfraser beroende på frågetyp
- enklare städning av meningar

Det är alltså kod, inte modellintelligens, som försöker förbättra läsbarheten.

### Styrkor

- snabb
- billig
- robust när extern LLM inte svarar
- transparent
- pedagogiskt bra för att förstå hur mycket man kan göra utan modell

### Svagheter

- språket blir lätt stelt
- täckningen blir ofta sämre
- den är känslig för OCR-brus och märkliga dokumentformuleringar
- den kan bli överförsiktig och säga att underlaget inte räcker

Detta är en av de tydligaste lärdomarna i labben: modellfri syntes är värdefull för lärande och robusthet, men den når inte samma kvalitet som den bästa LLM-vägen.

## 15. LLM-baserad syntes

LLM-spåret bygger på:

- `llm/client.py`
- `llm/reasoning.py`
- `llm/prompts.py`
- `rag/prompts.py`

Här används retrievalresultaten som underlag, men själva texten formuleras av extern modell.

Det viktiga här är att modellen inte ska svara fritt, utan styras att:

- hålla sig till källmaterialet
- säga ifrån när underlaget inte räcker
- skriva med egna ord
- täcka hela listor, etapper eller steg när sådana efterfrågas

Under arbetet blev det tydligt att prompten är viktig, men sekundär i förhållande till retrieval. En bra prompt kan förbättra språk, struktur och försiktighet, men den kan inte ersätta felaktigt hämtade källutdrag.

## 16. Modellval och praktiska LLM-erfarenheter

I praktiken provades flera modeller via Hugging Face-routning.

De viktigaste observationerna var:

- vissa modeller såg bra ut i katalogen men fungerade inte hos aktuell provider
- vissa fungerade men var långsamma
- vissa gav läckor av intern resonemangsstil
- vissa trunkerade svar eller blev instabila

Under den tidigare fasen av arbetet fungerade `openai/gpt-oss-120b` bäst av de modeller som då provades i den aktuella miljön.

Per den 10 juli 2026 uppdaterades dock standardvalet efter en ny kontroll mot Hugging Faces routade modellkatalog. Den kontrollen visade att följande kandidater fanns tillgängliga i den aktuella driftskedjan:

- `zai-org/GLM-5.2`
- `openai/gpt-oss-120b`
- `Qwen/Qwen3-32B`
- `deepseek-ai/DeepSeek-R1`
- `zai-org/GLM-4.5`

Det nya standardvalet blev `zai-org/GLM-5.2`.

Skälen var följande:

- modellen var faktiskt tillgänglig via Hugging Face-routningen vid kontrolltillfället
- den exponerades med flera live-providers
- den presenterades som en ny flaggskeppsmodell för långhorisontuppgifter
- Hugging Face-kortet beskrev stöd för upp till 1M token kontext

Samtidigt behölls `Qwen/Qwen3-32B` som ett tydligt alternativ eftersom Qwen3-familjen har stark flerspråkig profil och är intressant för svensk RAG-syntes.

Den praktiska lärdomen är därför dubbel:

- modellval måste baseras på vad som faktiskt är routbart i den aktuella driftmiljön
- standardvalet bör kunna ändras när en ny modell blir verkligt tillgänglig och inte bara när den nämns i externa nyheter eller modelljämförelser

## 17. Källor och transparens

En viktig del av lösningen är att användaren kan se källor till svaren.

PDF-länkar publiceras via GitHub Pages:

- `https://tomashelmfridsson.github.io/systeminforande/pdfs/<filnamn>`

Detta val gjordes eftersom det gav:

- stabila publika dokumentlänkar
- enklare länkning från svaren
- separering mellan dokumenthosting och apphosting

Det visade sig vara ett bättre upplägg än att låta Hugging Face bära hela dokumentdelen.

## 18. Deployment och drift

Lösningen deployas i två delar:

- GitHub Pages för PDF-filer
- Hugging Face Spaces för Gradio-appen

GitHub Actions används för att publicera appen till Hugging Face. Senare lades även ett nattligt hälsocheckjobb till för att upptäcka om den valda LLM-modellen inte längre är tillgänglig.

Detta är en viktig praktisk lärdom: när man bygger en lösning ovanpå externa modeller räcker det inte att koden fungerar. Man måste också övervaka att den externa beroendekedjan fortfarande fungerar.

## 19. Viktigaste lärdomar

### 1. Chunkning är inte ett hjälpsteg utan ett kärnsteg

Det var lätt att först tänka att chunkning bara handlar om att "dela upp text". I praktiken visade labben att chunkning styr:

- hur bra retrievalen kan bli
- hur mycket brus som följer med
- hur lätt det är att skapa bra syntes

### 2. Retrievalproblemet kommer före promptproblemet

Många dåliga svar såg först ut som promptproblem, men visade sig i själva verket bero på att fel chunkar hämtades eller att bra chunkar hade för låg rankning.

### 3. Temperatur och modellval påverkar svarsstil mer än källträff

Temperatur kan göra en modell mer eller mindre kreativ, men den kan inte ersätta bra retrieval.

### 4. BM25 är fortfarande mycket användbart

Trots all uppmärksamhet kring embeddings och vektordatabaser visade labben att en väljusterad BM25-lösning med bra chunkning och domänheuristik kan ge mycket långt resultat, särskilt i en begränsad dokumentmängd.

### 5. Modellfri syntes är pedagogiskt mycket värdefull

Även om den inte ger bäst svar visar den tydligt vilka delar av kedjan som faktiskt fungerar utan svart låda. Det gör den mycket användbar i en lärmiljö.

### 6. Externa modeller kräver operativ robusthet

Rate limits, providerbyten och modellstöd blev en verklig del av systemdesignen. Det räcker inte med "rätt kod", utan man måste även bygga felhantering och övervakning.

## 20. Nästa rimliga förbättringar

Utifrån labben framstår följande förbättringar som mest intressanta:

- prova hybrid retrieval ovanpå nuvarande BM25
- införa reranking på toppkandidater
- förbättra den modellfria syntesen för definitioner och översiktsfrågor
- lägga till mer diagnostik som visar varför vissa chunkar vann
- jämföra enkel svensk stemming mot mer avancerad språknormalisering

## 21. Sammanfattning

Den viktigaste tekniska lärdomen från labben är att bra RAG inte börjar i modellen, utan i materialberedningen.

Vi byggde en lösning där:

- PDF-filer extraheras och chunkas lokalt
- retrieval sker med BM25-liknande lexikal sökning
- heuristiker och boostar används för domänanpassning
- ett modellfritt spår visar vad som går att göra utan LLM
- ett LLM-spår visar hur mycket bättre formulering och täckning man kan få när retrievalen fungerar

Från ett lärperspektiv blev detta särskilt tydligt:

- chunkning avgör mer än man tror
- retrieval är ofta den verkliga flaskhalsen
- temperatur hör till generering, inte retrieval
- BM25 är enkelt men kraftfullt
- heuristiker kan ge stor effekt i en smal domän

Det gör denna labb till mer än en chatbot. Den fungerar också som en konkret genomlysning av hur en RAG-lösning faktiskt byggs, justeras och utvärderas i praktiken.

## 22. När vi upptäckte att RAG:en inte höll måttet

Den 10 juli 2026 gjorde vi ett mer systematiskt kvalitetspass mot den deployade chatboten via Gradio API i stället för att bara klicktesta GUI:t.

Det var ett viktigt skifte. Så länge vi bara ställde enstaka frågor manuellt gick det att få intrycket att lösningen fungerade "ganska bra". När vi däremot började köra samma frågor om och om igen som regressionstester blev svagheterna tydliga.

### Vilka fel vi såg

Två typer av fel stack ut direkt:

- frågor om `acceptanstest` drog inte alltid upp de mest relevanta acceptanstestdokumenten
- definitionsfrågor som `Vad är ett arbetsområde?` kunde ranka ett testplandokument före själva checklistan för arbetsområden

Vi såg också en tredje svaghet:

- vanliga stavfel i svenska frågor gjorde retrievalen märkbart sämre

Exempel på detta var att frågor som innehöll former som `acceptanstst` eller `implmenteringen` tappade precision trots att användarens avsikt fortfarande var mycket tydlig.

### Hur testerna avslöjade problemen

Vi byggde först live-tester mot den deployade Gradio-API:n. De testerna valdes medvetet för att täcka tre saker:

- typiska verksamhetsfrågor
- frågor där rätt källdokument borde vara ganska uppenbara
- frågor med små stavfel som en praktiskt användbar RAG borde tåla

Detta gav oss en bättre signal än rena enhetstester, eftersom vi kunde se hela kedjan:

- fråga
- retrieval
- syntes
- källänkar i svaret

När testen för `acceptanstest` och `arbetsområde` föll blev det tydligt att problemet satt i retrievalen, inte primärt i LLM-prompten.

Det viktiga var alltså inte bara att ett svar blev "lite svagt", utan att fel dokumentfamilj eller fel sektion faktiskt vann rankningen.

### Vår diagnos

Efter att ha läst igenom `rag/search.py` och kört förklarande sökningar lokalt kunde vi se flera konkreta orsaker.

#### 1. För strikt matchning av originaltermer

Den tidigare funktionen `_has_retrieval_support(...)` krävde i praktiken att originaltermer från frågan återfanns ganska direkt i chunkens text eller titel.

Det gav två problem:

- singular och plural möttes inte alltid väl nog
- ett dokument som var rätt i sak men där termen främst syntes i filnamn eller närliggande variationer kunde filtreras bort

Detta förklarade till stor del varför `Vad är ett arbetsområde?` kunde missa `Checklista_Arbetsomraden.pdf` trots att just den filen uppenbart är central.

#### 2. För svag användning av filnamn och dokumentfamilj

Den tidigare retrievalen tittade främst på titel och löptext. Men i ett fast corpus som detta bär själva filnamnen mycket domäninformation:

- `Acceptanstest`
- `Konvertering`
- `Utbildning`
- `Driftsättning`
- `Checklista_Arbetsomraden`

Att inte använda dessa signaler fullt ut var ett misstag, särskilt när dokumentbeståndet är stabilt och känt.

#### 3. För liten tolerans för svenska stavfel och skrivvarianter

Den tidigare normaliseringen var enkel och hjälpte vid vissa böjningar, men inte tillräckligt för:

- `arbetsområde` kontra `arbetsområden`
- `införande` kontra `inforande`
- `acceptanstest` kontra `acceptanstst`
- `implementeringen` kontra `implmenteringen`

Det gjorde att retrievalen fortfarande i hög grad betedde sig som en ganska strikt ordmatchare.

#### 4. Webbkällor kunde konkurrera för lätt med de kuraterade PDF:erna

Vi hade både PDF-material och vissa webbkällor i indexet. För mer allmänna frågor kunde webbsidor ibland rankas för högt, trots att vår viktigaste kunskapsmassa i praktiken finns i PDF:erna.

För just denna lösning var det fel prioritering.

### Vad vi ändrade

När diagnosen var klar gjorde vi förbättringarna i retrievallagret först. Vi ändrade inte prompten först, eftersom testen redan visade att fel chunkar kom in i kedjan.

#### 1. Vi gjorde tokeniseringen mer svensk-robust

Vi införde en tydligare normaliseringskedja:

- teckenfoldning för `å`, `ä`, `ö`
- fortsatt suffixnormalisering
- kanonisering av vanliga domänord till samma form

Detta gjorde att exempelvis:

- `införandet` och `inforande` hamnar närmare varandra
- `arbetsområde` och `arbetsområden` kopplas starkare ihop
- vissa felstavade former av centrala ord kan landa i rätt domänterm

Det var ett medvetet val att hålla detta heuristiskt och lättviktigt i stället för att dra in en tung svensk NLP-pipeline.

#### 2. Vi började använda filnamn som en förstklassig retrievalsignal

Varje chunk får nu även sökbara token från sin källa, alltså filnamnet.

Det betyder att sökningen inte bara ser:

- rubriken i chunken
- löptexten i chunken

utan också:

- vilken dokumentfamilj chunken tillhör

I en smal, fast dokumentmängd är detta mycket värdefullt. Ett dokument som heter `210_Acceptanstest_testplan.pdf` ska naturligtvis få extra chans att vinna på frågor om acceptanstest.

#### 3. Vi lade till dokumentfamiljer och domänboostar

Vi införde starkare regler för dokumentfamiljer som:

- `acceptanstest`
- `arbetsomrade`
- `konvertering`
- `projekt`

Dessutom förstärktes `DOMAIN_RULES` för frågor om:

- acceptanstest
- leveransgodkännande
- implementering
- planering
- uppföljning
- verifiering

Skälet var enkelt: med ett fast corpus är det rationellt att använda domänkunskap explicit. Detta är inte ett generellt webbsökproblem utan en kuraterad kunskapsmängd.

#### 4. Vi lade till fuzzy-expansion mot corpusets eget vokabulär

I stället för att försöka gissa alla stavfel manuellt lät vi frågetermer expandera mot indexets eget ordförråd när avståndet är litet nog.

Det är en enkel men effektiv strategi för interna RAG-lösningar:

- den kräver ingen extern stavningsmodell
- den använder bara de ord som faktiskt finns i corpus
- den hjälper just där användaren ligger "nära rätt"

Detta förbättrade särskilt frågor som innehöll små skrivfel i centrala begrepp.

#### 5. Vi nedviktade webbkällor

För denna labb är PDF:erna det viktigaste källmaterialet. Därför lades en mindre straffpoäng på webbkällor i retrievalen.

Detta innebär inte att webbinnehåll ignoreras, men att det inte lika lätt får slå ut de kuraterade PDF-dokument som bygger själva verksamhetskunskapen.

### Varför vi inte började med embeddings direkt

Det hade varit möjligt att gå direkt till hybrid retrieval eller embeddingsökning. Vi valde ändå att först pressa den lexikala retrievalen längre.

Det valet gjordes av tre skäl:

- corpus är litet och stabilt
- domänspråket är smalt och på svenska
- vi ville först förstå exakt vilka fel som kom från index, normalisering och rankning

Det gav bättre transparens. När en förbättring fungerar vet vi då också varför den fungerar.

### De nya lokala regressionstesterna

Efter live-testerna lade vi också till lokala retrievaltester. Det var viktigt av två skäl:

- de går snabbt att köra utan deploy
- de låser fast de svagheter vi redan har hittat så att de inte smyger tillbaka

Testerna fokuserar på fyra riskzoner:

- att `acceptanstest` leder till acceptanstestdokument
- att `arbetsområde` leder till rätt checklista
- att stavfel i nyckelord fortfarande ger rätt dokumentfamilj
- att projektets PDF:er prioriteras före webbkällor i interna kvalitetsfrågor

Detta är ett bra exempel på varför RAG bör testas som en sök- och rankningsprodukt, inte bara som "något som får en LLM att svara".

### Vad resultaten visade efter förbättringen

Efter retrievaländringarna kunde vi lokalt verifiera att:

- frågor om `acceptanstest` nu tydligt drog upp acceptanstestdokument
- `Vad är ett arbetsområde?` rankade `Checklista_Arbetsomraden.pdf` först
- vanliga stavfel i centrala ord inte längre slog sönder retrievalen på samma sätt
- PDF-spåret fick högre prioritet än webbsidor i våra regressioner

Det betyder inte att RAG:en nu är "färdig". Men det betyder att den inte längre faller på samma grundläggande retrievalmisstag som tidigare.

### Den viktigaste lärdomen från detta förbättringspass

Den viktigaste lärdomen var att en svag RAG mycket ofta ser ut som ett promptproblem fast den i själva verket är ett retrievalproblem.

Så länge fel dokumentfamilj, fel sektion eller fel stavningshantering vinner i rankningen hjälper det begränsat att:

- byta modell
- ändra temperatur
- skriva längre prompt

Det som gav verklig effekt här var i stället:

- bättre svensk normalisering
- bättre användning av metadata
- bättre dokumentfamiljsignaler
- regressionstester som gjorde svagheterna synliga

Detta kapitel är därför kanske den viktigaste praktiska lärdomen i hela labben: RAG förbättras inte främst genom att man "ber modellen smartare", utan genom att man gör retrievalen mer sann mot den kunskap man faktiskt har.

## 20. Ny corpus, mindre hårdkodning och bättre processsvar

När PDF-underlaget byttes ut blev det tydligt att flera retrievalregler var för tätt bundna till de gamla filnamnen.

Det syntes direkt i lokala körningar:

- retrieval pekade fortfarande på gamla dokumentnamn i indexet
- testsviten förutsatte gamla källor
- vissa kvalitetsförbättringar byggde mer på dokumentnamn än på egenskaper i innehållet

Den här delen av labben handlade därför om att göra lösningen mer portabel utan att tappa kvalitet i de frågor vi redan visste var svåra.

### Steg 1. Vi byggde om indexet efter corpusbytet

Det första praktiska steget var att köra om ingest och skriva om `rag/data/chunks.json` mot det nya PDF-beståndet.

Detta var nödvändigt eftersom retrieval annars fortfarande arbetade mot chunkar från den gamla dokumentmängden.

Lärdomen är enkel men viktig:

- corpusbyte utan omindexering ger missvisande felsökning
- retrievalfel kan annars se ut som rankningsfel fast indexet bara är stale

### Steg 2. Vi uppdaterade regressionstesterna till den nya dokumentmängden

De tidigare testerna i `tests/test_search_quality.py` och `tests/data/live_api_scenarios.json` var skrivna mot gamla filnamn.

Vi justerade därför regressionerna så att de:

- fortfarande testar samma användarfrågor
- men accepterar de nya dokumenten som rätt svarskällor

Detta gjorde två saker:

- testerna blev användbara igen
- vi kunde mäta förbättringar utan att vara låsta till gamla filstrukturer

### Steg 3. Vi gick från dokumentnamnsstyrning till egenskapsstyrning i retrievalen

Tidigare var retrievalen starkt påverkad av explicita regler för specifika dokumentnamn och dokumentfamiljer.

Det fungerade bra i ett fast corpus, men blev skört när PDF:erna byttes ut.

Vi ersatte därför stora delar av den logiken med mer generiska signaler:

- titelmatch mot frågetermer
- om frågan är en definitionsfråga, processfråga eller listfråga
- om sektionen ligger tidigt i dokumentet
- om rubriken ser ut som översikt, modell, kompetens, process eller planering
- om chunken verkar handla om ämnet i själva titeln eller bara råkar ligga i rätt dokument

Detta gav två viktiga effekter:

- retrievalen blev mindre beroende av exakt filnamn
- samma kod blev mer rimlig att återanvända mot ett annat corpus

Den viktigaste lärdomen här var att hårdkodade boostar kan vara effektiva, men att de bör ersättas av egenskapsbaserade signaler så snart man vill ha en mer portabel RAG.

### Steg 4. Vi lade till en lätt lokal reranker ovanpå BM25

Efter att den första rankningen blivit mer generell lade vi in en andra rankningsfas ovanpå toppkandidaterna.

Denna reranker:

- arbetar lokalt
- kräver inga externa modeller
- använder en TF-IDF-liknande likhetsbedömning på teckenn-gram
- kör bara på de högst rankade kandidaterna från första retrievalsteget

Poängen var inte att ersätta BM25, utan att förbättra ordningen bland redan rimliga kandidater.

Det gav särskilt nytta i frågor där flera chunkar innehöll samma domänord men där bara en av dem faktiskt var den bästa förstakällan.

Ett tydligt exempel var:

- `Vad är en projektorganisation`

Där kunde en mallsektion och en riktig definitionssektion båda se relevanta ut, men rerankern hjälpte den verkliga ämnessektionen att vinna.

Lärdomen är att en lätt andra rankingfas kan ge tydlig kvalitetsvinst även utan embeddings eller cross-encoder.

### Steg 5. Vi förbättrade retrievalen för processfrågor

När vi granskade frågan:

- `Hur används acceptanstest i införandet?`

så såg vi att retrievalen först gynnade en sektion om `Uppföljning` i en mall, snarare än den sektion som faktiskt hette `Acceptanstest`.

Problemet var inte att ordmatchningen var fel, utan att processord som:

- planering
- uppföljning
- genomförande

ibland slog ut den chunk som faktiskt bar ämnet i rubriken.

Vi justerade därför retrievalen så att:

- chunkar vars titel innehåller den verkliga ämnestermen får starkare vikt
- chunkar som bara matchar via generiska processord får mindre fördel

Det förbättrade ordningen för både:

- `Hur används acceptanstest i införandet?`
- den felstavade varianten `Hur anvnds acceptanstst i införandet?`

Detta var en viktig lärdom:

- i processfrågor räcker det inte att hitta "processliknande" chunkar
- man måste också hitta rätt process om flera konkurrerar

### Steg 6. Vi byggde en särskild generator för processsvar

När retrievalen väl gav bättre chunkar blev nästa svaghet tydlig:

- den extractive svarsgenereringen var fortfarande för mekanisk

För `hur`-frågor lade vi därför till en särskild processgenerator i `rag/extractive.py`.

Den känner igen fasrubriker som:

- `Planering`
- `Förberedelser`
- `Genomförande`
- `Uppföljning`

och bygger sedan ett svar som följer processens ordning.

I stället för att bara välja "bästa meningar" gör den nu detta:

- identifierar processfaser
- samlar punktinnehåll under varje fas
- sammanfattar faserna i ordning
- bygger ett kompakt svar som liknar hur en människa förklarar ett arbetsflöde

Detta förbättrade tydligt svaret på acceptanstestfrågan.

Lärdomen här var att bättre retrieval inte automatiskt ger bättre svar. När retrievalen blivit tillräckligt bra behöver också själva syntesen bli mer frågetypsmedveten.

### Nya regressioner från verkliga loggar

Vi lade under detta pass också till regressioner för frågor som kom från verkliga loggar, bland annat:

- `Finns det en checklista för införandekrav`
- `Vad är ett arbetsområde?`
- `Vad är ett projektbibliotek`
- `Vad är en projektorganisation`
- `Hur används acceptanstest i införandet?`

Det viktiga med dessa tester är att de inte bara fångar retrieval utan också om svaret uttrycker rätt sak för den aktuella frågetypen.

### Resultatet efter detta pass

Efter dessa ändringar kunde vi lokalt verifiera att:

- retrievalen fortfarande fungerar mot det nya corpuset
- lösningen är mindre bunden till gamla filnamn
- acceptanstestfrågan nu rankar rätt ämnessektion först
- processfrågor får mer naturliga stegvisa svar
- regressionstesterna fortsatt passerar lokalt

### Viktigaste lärdomar från detta pass

Detta förbättringspass gav tre centrala lärdomar:

1. Dokumentnamnsstyrning kan vara effektivt, men är skör när corpus förändras.
2. En lätt lokal reranker kan förbättra ordningen mellan redan relevanta träffar utan tunga externa beroenden.
3. När retrievalen blir bättre måste även svargenereringen bli mer frågetypsspecifik, annars stannar kvalitetsvinsten i söksteget.

Den kanske viktigaste sammanfattningen är därför:

- portabel RAG kräver att retrievalsignaler uttrycks som innehållsegenskaper snarare än dokumenthårdkodning
- och bra svar kräver att syntesen känner igen frågans struktur, inte bara källans ord

## 21. Frågetypsstyrd svargenerering utöver processfrågor

Efter att retrievalen och processgeneratorn förbättrats blev nästa steg att göra svargenereringen mer frågetypsmedveten även för andra typer av frågor.

Det visade sig att flera svar fortfarande var "tekniskt rimliga" men kvalitativt svaga:

- listfrågor kunde ge generiska uppräkningar av rubriker i stället för ett ärligt svar
- syftesfrågor kunde fastna på ordet `syfte` och dra in fel chunk
- tids- och beslutsfrågor kunde upprepa formuleringar från källan utan att tydligt säga om ett datum faktiskt fanns eller inte

### Steg 1. Vi gjorde listsvaret för etappfrågor mer ärligt

Frågan `Vilka etapper finns det` visade att retrievalen hittade material om etappindelning, men att de hämtade utdragen inte alltid faktiskt listade etapperna tydligt.

Det tidigare listsvaret blev därför lätt missvisande: det lät konkret trots att underlaget i de hämtade chunkarna inte räckte till.

Vi ändrade därför logiken så att etappsvaret:

- listar etapper när de verkligen går att extrahera
- annars uttryckligen säger att materialet visar att införandet är etappindelat men att de hämtade utdragen inte räcker för att återge varje etapp

Lärdomen här är viktig:

- ett bra RAG-svar ska inte bara vara relevant
- det ska också vara ärligt om gränsen för vad de hämtade utdragen faktiskt stöder

### Steg 2. Vi lade till en riktad generator för syftesfrågor

Frågan `Vad är syftet med beslutspunkter?` visade en klassisk RAG-svaghet:

- ordet `syfte` matchade generellt många chunkar
- men den relevanta kunskapen låg i chunken `Beslutspunkter`, inte i andra chunkar som råkade innehålla ord om syfte eller mål

Vi lade därför till en särskild syftesgenerator för frågor av typen:

- `Vad är syftet med ...`

Den gör inte bara ordmatchning på `syfte`, utan letar efter chunkar där själva ämnet också finns i titel eller innehåll.

För `beslutspunkter` kunde den därefter formulera ett bättre svar:

- syftet är att ge projektets ansvariga möjlighet att kontrollera och styra projektets fortskridande
- beslutspunkter används för att starta och avsluta faser, etapper och viktiga delresultat
- beslut ska kunna fattas på rätt organisatorisk nivå

Lärdomen var att syftesfrågor ofta kräver en mer ämnesspecifik syntes än vanliga definitionsfrågor.

### Steg 3. Vi lade till en riktad generator för tids- och beslutsfrågor

Frågor som:

- `När ska driftsättning ske?`
- `När fattas beslut om leveransgodkännande?`

blev tidigare lätt mekaniska. De kunde upprepa samma formulering flera gånger utan att tydligt säga det viktigaste:

- finns det en exakt tidpunkt i underlaget eller inte?

Vi lade därför till en särskild generator för tids- och beslutsfrågor som gör en mer användbar bedömning:

- om materialet saknar exakt datum säger svaret det tydligt
- om materialet bara anger att en tidpunkt ska planeras eller fastställas, så uttrycks just det
- om beslut ska knytas till kriterier, beslutsfattare och planerad tidpunkt framgår det uttryckligt

Detta förbättrade kvalitativt svaren, eftersom de nu säger vad materialet faktiskt ger stöd för i stället för att bara spegla frågans ord tillbaka.

### Vad dessa ändringar förbättrade

Efter detta pass blev svaren bättre på tre sätt:

1. de blev ärligare när underlaget var begränsat
2. de blev tydligare om vad som faktiskt anges i materialet
3. de blev mer användbara för användaren än rena omformuleringar av chunkar

### Nya regressioner från detta pass

Vi lade även till lokala regressioner för:

- `Vad är syftet med beslutspunkter?`
- `När ska driftsättning ske?`
- `När fattas beslut om leveransgodkännande?`

Detta gav en viktig förstärkning av testmodellen:

- vi testar nu inte bara retrieval och processfrågor
- vi testar också om svaret uttrycker rätt kunskapstyp för syfte, tid och beslut

### Viktigaste lärdomar från detta pass

Detta förbättringspass gav en tydlig slutsats:

- olika frågetyper behöver olika syntesstrategier

Det räcker inte att retrievalen hittar rätt chunk. Om svargenereringen behandlar alla frågor som "plocka några bra meningar", blir svaren ofta:

- för mekaniska
- för vaga
- eller för tvärsäkra på sådant som källan inte egentligen säger

Den praktiska lärdomen är därför:

- processfrågor bör byggas som steg
- syftesfrågor bör byggas som funktion och avsikt
- tids- och beslutsfrågor bör byggas som evidens om datum, villkor och ansvar

Detta är ett viktigt steg mot en RAG där svargenereringen inte bara är korrekt, utan även mer användbar och mer sann mot underlaget.

## 22. Planeringsfrågor som egen syntestyp

Efter att vi lagt till generatorer för process-, syftes- och tidsfrågor återstod fortfarande en typ av fråga som inte blev riktigt bra:

- `Hur används arbetsområden i planeringen?`

Retrievalen för den frågan var redan tillräckligt bra. Den hämtade bland annat:

- `Verktyget_och_systeminforandet.pdf | 1.4 Arbetsområden vid systeminförande`
- `Verktyget_projektstyrning.pdf | 2.3 Planering och uppföljning`

Problemet låg alltså inte främst i sökningen, utan i syntesen. Den generella extractive-logiken gav ett svar som i praktiken bara återgav att arbetsområden finns, inte hur de används i planeringen.

### Vad vi ändrade

Vi lade därför till en liten riktad generator för frågor som samtidigt handlar om:

- `hur`
- `arbetsområden`
- `planering`

Generatorn bygger inte svaret från enstaka meningar utan uttrycker den funktion som arbetsområden har i materialet:

- de används för att strukturera planeringen
- de gör det lättare att planera aktiviteter
- de hjälper till att fördela ansvar
- de fungerar som grund för uppföljning

### Varför detta var viktigt

Det här är ett bra exempel på att vissa frågor inte bara kräver en korrekt definition eller ett processflöde, utan en förklaring av hur ett begrepp används i projektstyrningen.

Om man svarar för bokstavligt på sådana frågor blir svaret lätt:

- sant men tunt
- relevant men inte användbart

Med en riktad generator kunde vi i stället ge ett svar som ligger närmare användarens verkliga avsikt:

- inte bara vad arbetsområden är
- utan vilken roll de spelar i planering och uppföljning

### Ny regression från detta pass

Vi lade till ett lokalt test som kontrollerar att svaret på:

- `Hur används arbetsområden i planeringen?`

innehåller signaler om:

- planering
- struktur
- ansvar
- uppföljning

### Lärdomen

Den här förbättringen förstärkte en tidigare slutsats:

- även när retrievalen är god nog måste svargenereringen förstå vilken typ av användningsfråga som ställs

Det räcker alltså inte alltid med:

- definition
- process
- tidpunkt
- syfte

Vissa frågor kräver i stället ett svar om funktion i arbetssättet. Det är en egen kunskapstyp, och den blev tydlig just i frågorna om arbetsområden och planering.
