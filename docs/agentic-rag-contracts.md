# Agentic RAG contracts and architecture

Datum: 2026-07-22

Syftet med detta dokument är att definiera en kontrollerad 3-agentdesign för nästa RAG-steg innan implementation. Designen ska förbättra svensk retrieval och svarskvalitet utan att göra lösningen beroende av hårdkodade domänord som `överlämna` eller `förvalta`. Den ska fungera generellt för svensk grammatik, böjningar och synonymer, till exempel `undervisning`, `undervisade` och `undervisat`.

## Grundprincip

Originalfrågan från användaren är alltid ankaret för slutsvaret och granskningen. Agent 1 får skriva om frågan endast för retrieval. Agent 2 måste svara på originalfrågan, inte på Agent 1:s omskrivning. Agent 3 måste granska svaret mot originalfrågan och källunderlaget och stoppa drift om svaret glider mot den omskrivna retrievalfrågan.

Den kontrollerade kedjan är:

1. Lokal retrieval hämtar en första kandidatpool från `rag/data/chunks.json`.
2. Agent 1 (`openai/gpt-oss-20b`) skapar ett strikt JSON-objekt med retrievalvarianter, svenska termvarianter och negativa avgränsningar.
3. Lokal retrieval körs igen med Agent 1:s varianter och slås ihop med den första kandidatpoolen.
4. Agent 2 (`openai/gpt-oss-120b`) jämför evidens och skriver ett svar på originalfrågan.
5. Agent 3 (`openai/gpt-oss-20b`) granskar grounding, frågefokus och drift.
6. Slutsvaret publiceras bara om Agent 3 godkänner eller begär små, källstödda korrigeringar. Annars faller systemet tillbaka till nuvarande extractive/grounded svar.

## Modelluppdelning

- Agent 1, retrieval rewrite: `openai/gpt-oss-20b`. Liten modell räcker eftersom uppgiften är avgränsad, strukturerad och inte ska formulera sluttext.
- Agent 2, evidence comparison and answer: `openai/gpt-oss-120b`. Den starkare modellen används där den gör mest nytta: att väga källor, hantera frågefokus och skriva naturlig svensk prosa.
- Agent 3, grounding and drift review: `openai/gpt-oss-20b`. Granskningen är ett kontrollerat ja/nej-/patchbeslut och ska vara billigare än svargenereringen.

## Agent 1: retrieval rewrite contract

Agent 1 får inte svara på frågan. Den får bara skapa sökvarianter som hjälper retrieval att hitta rätt källor. Kontraktet ska valideras som JSON Schema och avvisas om det innehåller fri prosa utanför tillåtna fält.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "RetrievalRewriteResult",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "original_question",
    "retrieval_queries",
    "must_keep_focus",
    "semantic_terms",
    "negative_constraints",
    "confidence"
  ],
  "properties": {
    "original_question": {"type": "string", "minLength": 1},
    "retrieval_queries": {
      "type": "array",
      "minItems": 1,
      "maxItems": 5,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["query", "purpose"],
        "properties": {
          "query": {"type": "string", "minLength": 1, "maxLength": 180},
          "purpose": {
            "type": "string",
            "enum": ["literal", "swedish_inflection", "synonym", "compound", "broader_context"]
          }
        }
      }
    },
    "must_keep_focus": {
      "type": "array",
      "minItems": 1,
      "maxItems": 6,
      "items": {"type": "string", "maxLength": 80}
    },
    "semantic_terms": {
      "type": "array",
      "maxItems": 12,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["surface", "normalized_family", "kind"],
        "properties": {
          "surface": {"type": "string", "maxLength": 60},
          "normalized_family": {"type": "string", "maxLength": 60},
          "kind": {"type": "string", "enum": ["lemma", "inflection", "synonym", "compound", "spelling_variant"]}
        }
      }
    },
    "negative_constraints": {
      "type": "array",
      "maxItems": 6,
      "items": {"type": "string", "maxLength": 120}
    },
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
  }
}
```

Testbara regler:

- `original_question` måste vara exakt användarens originalfråga.
- `retrieval_queries` får inte innehålla föreslagna svar eller påståenden.
- `must_keep_focus` måste innehålla frågans faktiska mål, till exempel om användaren frågar efter arbetssätt, tidpunkt, ansvar, definition eller hinder.
- `semantic_terms` ska beskriva generella svenska termfamiljer. Det är tillåtet att hitta böjnings- och synonymfamiljer dynamiskt, men inte att lägga in specialregler för en enskild fråga som primär lösning.
- Om JSON-validering, timeout eller confidence < 0.55 inträffar används bara lokal retrieval.

## Agent 2: evidence comparison and answer contract

Agent 2 får skriva slutkandidaten, men bara inom källunderlaget. Den får se originalfrågan, Agent 1:s retrievalkontrakt, en deduplicerad kandidatpool och nuvarande extractive fallback. Den ska returnera strukturerat JSON, inte bara text.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "EvidenceAnswerResult",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "original_question",
    "answer",
    "answer_scope",
    "evidence_used",
    "unsupported_or_uncertain",
    "source_coverage",
    "grounding_notes"
  ],
  "properties": {
    "original_question": {"type": "string", "minLength": 1},
    "answer": {"type": "string", "minLength": 40, "maxLength": 2200},
    "answer_scope": {
      "type": "string",
      "enum": ["direct", "partial_due_to_thin_evidence", "insufficient_evidence"]
    },
    "evidence_used": {
      "type": "array",
      "minItems": 1,
      "maxItems": 8,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["chunk_id", "source", "claim_supported"],
        "properties": {
          "chunk_id": {"type": "string"},
          "source": {"type": "string"},
          "pages": {"type": "array", "items": {"type": ["integer", "string"]}},
          "claim_supported": {"type": "string", "maxLength": 220}
        }
      }
    },
    "unsupported_or_uncertain": {
      "type": "array",
      "maxItems": 6,
      "items": {"type": "string", "maxLength": 180}
    },
    "source_coverage": {
      "type": "object",
      "additionalProperties": false,
      "required": ["uses_retrieved_chunks", "answers_original_question", "ignores_metadata_as_facts"],
      "properties": {
        "uses_retrieved_chunks": {"type": "boolean"},
        "answers_original_question": {"type": "boolean"},
        "ignores_metadata_as_facts": {"type": "boolean"}
      }
    },
    "grounding_notes": {"type": "string", "maxLength": 600}
  }
}
```

Testbara regler:

- `original_question` måste vara exakt originalfrågan.
- `answer` får inte nämna Agent 1, retrievalfrågor eller interna debugfält.
- Varje central påståendemening i `answer` ska kunna kopplas till minst ett objekt i `evidence_used`.
- `source_coverage.answers_original_question` måste vara true för publicering.
- Om `answer_scope` är `insufficient_evidence` ska svaret vara ett ärligt icke-svar och inte försöka ersätta saknad evidens med generella råd.
- Om JSON-validering misslyckas, svaret är tomt, eller groundingkontrollen faller, används nuvarande extractive fallback.

## Agent 3: grounding and drift review contract

Agent 3 är en gatekeeper. Den får inte förbättra svaret fritt. Den får antingen godkänna, föreslå små källstödda korrigeringar eller stoppa svaret.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "GroundingReviewResult",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "verdict",
    "answers_original_question",
    "grounded_in_sources",
    "drift_from_original_question",
    "metadata_leakage",
    "unsupported_claims",
    "required_action",
    "review_notes"
  ],
  "properties": {
    "verdict": {"type": "string", "enum": ["approve", "revise_minor", "reject"]},
    "answers_original_question": {"type": "boolean"},
    "grounded_in_sources": {"type": "boolean"},
    "drift_from_original_question": {"type": "boolean"},
    "metadata_leakage": {"type": "boolean"},
    "unsupported_claims": {
      "type": "array",
      "maxItems": 8,
      "items": {"type": "string", "maxLength": 180}
    },
    "required_action": {
      "type": "string",
      "enum": ["publish", "publish_after_minor_revision", "fallback_to_extractive", "ask_for_more_evidence"]
    },
    "minor_revision": {"type": "string", "maxLength": 2200},
    "review_notes": {"type": "string", "maxLength": 700}
  }
}
```

Testbara regler:

- Publicering kräver `answers_original_question=true`, `grounded_in_sources=true`, `drift_from_original_question=false`, `metadata_leakage=false` och inga blockerande `unsupported_claims`.
- `revise_minor` får bara användas när revisionen tar bort eller förtydligar text. Den får inte lägga till nya sakpåståenden.
- `reject` leder till fallback, inte till ny fri generering i samma request.
- Granskningen ska uttryckligen jämföra svaret mot originalfrågan, inte mot Agent 1:s retrievalvarianter.

## Feature flags

Föreslagna miljövariabler:

- `SYSTEMINFORANDE_ENABLE_AGENTIC_RAG`: global av/på. Default `false` tills live-HF-utvärdering visar förbättring.
- `SYSTEMINFORANDE_AGENTIC_RAG_MODE`: `off`, `shadow`, `reviewed_answer`. Default `shadow` i testmiljö.
- `SYSTEMINFORANDE_AGENT1_MODEL`: default `openai/gpt-oss-20b`.
- `SYSTEMINFORANDE_AGENT2_MODEL`: default `openai/gpt-oss-120b`.
- `SYSTEMINFORANDE_AGENT3_MODEL`: default `openai/gpt-oss-20b`.
- `SYSTEMINFORANDE_AGENTIC_RAG_MAX_RETRIEVAL_QUERIES`: default `5`.
- `SYSTEMINFORANDE_AGENTIC_RAG_MAX_CONTEXT_CHUNKS`: default `8`.
- `SYSTEMINFORANDE_AGENTIC_RAG_TIMEOUT_SECONDS`: default `18` totalt för agentkedjan, med kortare per-agent timeouts.
- `SYSTEMINFORANDE_AGENTIC_RAG_REQUIRE_REVIEW`: default `true`. Om false får Agent 2 bara publiceras om befintlig lokal groundingkontroll också godkänner.
- `SYSTEMINFORANDE_AGENTIC_RAG_LOG_CONTRACTS`: default `false` i publik drift, true i kontrollerad utvärdering. Logga aldrig rå persondata utöver den fråga användaren redan skickat.

Lägen:

- `off`: nuvarande retrieval + extractive/grounded synthesis används.
- `shadow`: agentkedjan körs och loggas strukturerat, men användaren får nuvarande svarsväg. Detta är första säkra live-läget.
- `reviewed_answer`: Agent 2:s svar kan publiceras efter Agent 3-godkännande och lokal groundingkontroll.

## Fallbackbeteende

Fallback ska vara deterministiskt och mätbart:

1. Agent 1 timeout, ogiltig JSON eller låg confidence: kör lokal retrieval utan rewrite.
2. Agent 1 returnerar färre än en giltig retrievalfråga: kör lokal retrieval utan rewrite.
3. Agent 2 timeout, ogiltig JSON, tomt svar eller `answers_original_question=false`: använd extractive fallback.
4. Agent 2 svarar med unsupported claims, metadata som fakta eller synlig drift: skicka till Agent 3; om Agent 3 inte godkänner används fallback.
5. Agent 3 timeout eller ogiltig JSON: använd fallback. Frånvaro av review är aldrig ett godkännande.
6. Om källunderlaget är tunt ska systemet hellre svara kort med osäkerhet än skapa en mer utvecklad men ogrundad text.
7. Alla fallbacks ska logga en maskinläsbar orsak: `agent1_invalid_json`, `agent2_grounding_failed`, `agent3_reject`, `agent_timeout`, `thin_evidence` eller motsvarande.

## Kompakta tokenbudgetar

2026-07-20-loggen innehöll 303 rader. Den enda raden med komplett tokenusage i den tillgängliga loggen var en `Qwen/Qwen3-32B`-körning med 5 216 prompttokens, 846 completiontokens och 6 062 totalt, cirka 24,2 sekunders latens. För `openai/gpt-oss-120b` saknades tokenusage i 277 händelser, vilket ska behandlas som saknad observability, inte som noll kostnad.

Budgeten för agentic RAG ska därför vara kompaktare än den observerade cirka 6k-tokenbaselinen per tung synteskörning:

- Agent 1 prompt: max 1 200 tokens. Output: max 350 tokens. Skicka originalfråga, topprankade titlar/kortdiagnos och instruktioner, inte hela chunktexter.
- Retrieval pool efter Agent 1: max 12 kandidater internt, dedupliceras ned till max 8 chunkar för Agent 2.
- Agent 2 prompt: max 4 000 tokens. Det omfattar originalfråga, Agent 1-kontrakt, extractive fallback och 6-8 komprimerade chunkar med källa/sidor/titel/textutdrag. Output: max 900 tokens.
- Agent 3 prompt: max 2 000 tokens. Skicka originalfråga, Agent 2-svar och evidence_used med korta utdrag, inte hela retrievalpoolen. Output: max 350 tokens.
- Total publiceringsbudget: sikta på högst cirka 5 500-6 000 tokens för hela publiceringskedjan i normalfallet. I `shadow`-läge får kedjan stoppas tidigare om den riskerar att överskrida budgeten.

Praktisk konsekvens: Agent 1 och Agent 3 måste vara korta JSON-arbetare. Agent 2 är enda platsen där längre prosa får produceras. Fulla dokumentchunkar ska inte skickas till alla tre agenter.

## Observability och testbarhet

Varje körning ska kunna testas utan att läsa fri textlogg:

- Spara `original_question`, agentmodeller, mode, valideringsstatus, fallbackorsak, antal retrievalqueries, antal chunkar till Agent 2 och Agent 3-verdict i strukturerade metadatafält.
- Spara inte bara sluttexten; spara även om den kom från `extractive`, `agent2_reviewed`, `agent2_minor_revision` eller `fallback`.
- RAGAS-/HF-reruns ska kunna filtrera på dessa fält för att jämföra agentkedjan mot nuvarande väg.
- Enhetstester ska validera JSON-schema, fallback vid ogiltig JSON, att originalfrågan bevaras exakt och att Agent 3 stoppar drift.
- Regressionstester för svenska böjningar ska använda generiska exempel utanför systeminförandedomenen, till exempel `undervisning`, `undervisade`, `undervisat`, så att lösningen inte bara passerar på tidigare Q22-termer.

## Implementation boundary

Det här dokumentet är ett kontrakt, inte implementation. Första implementationen bör börja i `shadow`-läge och rapportera om Agent 1 faktiskt förbättrar retrieval innan Agent 2/3 tillåts påverka användarsvaret. Om shadow-data inte visar bättre context precision/recall ska svarsgenereringen inte aktiveras bara för att den ger snyggare prosa.
