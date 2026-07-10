# Live API Post-Deploy Check

- Datum: `2026-07-10`
- Deploy-status: efter ny deploy
- Bas-URL: `https://helmfridsson-systeminforande.hf.space`

## 1. Kontroll av vald ny modell

Försök att köra live-appen med:

- `zai-org/GLM-5.2`

gav fel direkt från den deployade Gradio-appen:

`Value: zai-org/GLM-5.2 is not in the list of choices: ['google/gemma-2-2b-it', 'deepseek-ai/DeepSeek-R1', 'openai/gpt-oss-120b', 'zai-org/GLM-4.5', 'Qwen/Qwen3-4B-Thinking-2507']`

Detta visar att den live deployade appen den 10 juli 2026 fortfarande exponerade den gamla modellistan och alltså inte hade tagit in den nya standardmodellen `zai-org/GLM-5.2`.

## 2. Live-svit mot modell som faktiskt fanns i deployen

- Modell: `openai/gpt-oss-120b`
- Kommando:
  `SYSTEMINFORANDE_BASE_URL=https://helmfridsson-systeminforande.hf.space SYSTEMINFORANDE_LLM_MODEL=openai/gpt-oss-120b pytest -m live_api -q`

### Sammanfattning

- `1 passed`
- `8 failed`
- `4 deselected`

### Observation

Detta är en tydlig regression jämfört med baseline före deploy (`6 passed`, `3 failed`).

## 3. Konkreta symptom i den deployade appen

Flera tidigare frågor som åtminstone gav delvis relevanta svar föll nu tillbaka till generiska avvisningar som:

- `Frågan verkar inte ha relevant stöd i det tillgängliga källmaterialet.`

Även den separata frågan:

- `Vilka införandekrav måste vi ha?`

gav svaret:

- `Frågan verkar inte ha relevant stöd i det tillgängliga källmaterialet.`

## 4. Tolkning

Efter denna deploy verkar två problem finnas samtidigt:

1. Den deployade appen använder fortfarande den gamla modellistan och accepterar inte `zai-org/GLM-5.2`.
2. Den nuvarande liveversionen verkar samtidigt ha blivit betydligt mer restriktiv eller tappat retrievalstöd, eftersom många frågor nu faller tillbaka till generiska avvisningar där baseline tidigare gav faktiska svar.
