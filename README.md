---
title: Systeminforande
emoji: 🦀
colorFrom: purple
colorTo: yellow
sdk: docker
app_port: 7860
app_file: app.py
pinned: false
license: apache-2.0
short_description: Sida för systeminforande
---

Session guidelines for coding work live in `AI_CODING_GUIDELINES.md`.

PDF-källor för publika länkar ligger under `docs/pdfs/`.
Appen bygger käll-länkar mot GitHub Pages på
`https://tomashelmfridsson.github.io/systeminforande/pdfs/<filnamn>`.
GitHub Pages bör deployas via workflow i `.github/workflows/deploy-pages.yml`
så att LFS-spårade PDF-filer publiceras som riktiga filer och inte som LFS-pekarfiler.
I GitHub-repot behöver Pages vara satt till `GitHub Actions` som build source.
Hugging Face Space kan deployas från GitHub via `.github/workflows/deploy-huggingface.yml`
med en repo-secret `HF_TOKEN`.

Experimentell grounded LLM-syntes för fria RAG-frågor styrs med:
- `SYSTEMINFORANDE_ENABLE_LLM_SYNTHESIS=false` som säker standard.
- `SYSTEMINFORANDE_ENABLE_LLM_SYNTHESIS=true` för att slå på omskrivningssteget efter extraktivt svar.
- `SYSTEMINFORANDE_LLM_SYNTHESIS_MODEL=openai/gpt-oss-120b` eller `SYSTEMINFORANDE_LLM_SYNTHESIS_MODEL=zai-org/GLM-5.2` för att välja värdmodell när syntesen är aktiverad.

API-flödet stöder även per-anrop-override via `/api/ask` med JSON-fälten `enable_synthesis` (bool) och `llm_model` (str), så att samma deploy kan testas med syntes av/på och med olika modeller utan kodändring.

## Hugging Face-tokenförbrukning i loggar

Appen loggar Hugging Face-tokenförbrukning i de vanliga JSONL-användningsloggarna när `SYSTEMINFORANDE_ENABLE_LOGGING` är aktivt.

- Loggplats: `SYSTEMINFORANDE_LOG_DIR`, med `/data/logs` som standard i Hugging Face Space.
- Filformat: en JSONL-fil per UTC-dag, `YYYY-MM-DD.jsonl`.
- Relevanta eventtyper: `chat_question`, `api_question` och `faq_question`.
- Fält i loggrad: `metadata.llm_usage`.
- `/api/ask` returnerar även samma objekt som top-level `llm_usage`; RAG-svar kan dessutom ha `retrieval.llm_usage`.

`metadata.llm_usage` är ett per-fråga-aggregat över Hugging Face `InferenceClient.chat_completion`-anrop:

- `provider`: nuvarande klient/API, normalt `huggingface_hub.InferenceClient.chat_completion`.
- `model`: resolved Hugging Face-modell-id, till exempel `openai/gpt-oss-120b`.
- `calls`: antal Hugging Face chat-completion-anrop för frågan.
- `prompt_tokens`: summerade input/prompt-tokens, i tokens, eller `null`.
- `completion_tokens`: summerade genererade/output-tokens, i tokens, eller `null`.
- `total_tokens`: summerade tokens, i tokens, eller `null`.
- `missing`: `false` bara när minst ett anrop finns och alla tokenvärden som behövs för aggregatet finns. `true` betyder att tokenvärden saknas eller är ofullständiga.
- `calls_detail`: en rad per Hugging Face-anrop med `purpose`, `provider`, `model`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `missing`, `status` och `error`.

Kända `calls_detail.purpose`-värden är `synthesis` för det valfria källbundna omskrivningssteget och `debug_comparison` för LLM-jämförelsen som bara visas i debugläge.

Tolka `null` som "ej tillgängligt", inte som noll tokens. Vanliga fall:

- Inget LLM-anrop gjordes, till exempel för fördefinierade FAQ-svar, tom fråga, syntes avstängd eller otillräckligt RAG-underlag: `calls=0`, tokenfälten är `null`, `missing=true` och `calls_detail=[]`.
- Hugging Face svarade men skickade ingen användbar `response.usage`: berörda tokenfält blir `null`, anropet får `missing=true`, och aggregatet får `missing=true`.
- Om ett av flera anrop saknar ett tokenfält blir aggregatet för det fältet `null`, så att en delsumma inte misstas för en komplett totalsumma. Kontrollera `calls_detail` för att se vilket anrop som saknade värden.
- Om ett Hugging Face-anrop misslyckas loggas det i `calls_detail` med `status="error"`, `error` satt, tokenfält `null` och `missing=true`.

För kostnads- eller tokeneffektivitetsjämförelser bör rader med `metadata.llm_usage.missing=false` användas som uppmätta rader, och rader med saknade tokenvärden rapporteras separat.
