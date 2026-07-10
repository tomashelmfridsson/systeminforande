# Live API Result After Successful GLM-5.2 Deploy

- Datum: `2026-07-10`
- Deploy-status: efter färdig restart och ny runtime
- Modell: `zai-org/GLM-5.2`
- Bas-URL: `https://helmfridsson-systeminforande.hf.space`
- Runtime SHA: `dc08bb0da2751cc358af8d34a7e81195ffc1dbdc`
- Kommando:
  `SYSTEMINFORANDE_BASE_URL=https://helmfridsson-systeminforande.hf.space SYSTEMINFORANDE_LLM_MODEL=zai-org/GLM-5.2 pytest -m live_api -q`

## Sammanfattning

- `9 passed`
- `4 deselected`

## Viktig observation

Den tidigare frågan:

- `Vilka införandekrav måste vi ha?`

gav nu ett faktiskt svar med källor i stället för fallback.

## Exempel på svarsbeteende för införandekrav

Svaret sammanfattade att införandekrav är indirekta krav som kompletterar funktionella och icke-funktionella krav, delade in dem i:

- system
- verksamhet
- teknik och drift

och hänvisade till:

- `https://www.systeminforande.se/infrandekrav-1`
- `Checklista_Inforandekrav.pdf`

## Jämförelse mot tidigare lägen

### Baseline före ny deploy

Fil:

- `tests/results/live_api_baseline_2026-07-10_gpt-oss-120b.md`

Resultat:

- `6 passed`
- `3 failed`
- `4 deselected`

### Mellanläge med felaktig/halv gammal runtime

Fil:

- `tests/results/live_api_postdeploy_2026-07-10_current-hf-state.md`

Resultat:

- `1 passed`
- `8 failed`
- `4 deselected`

### Slutläge efter korrekt GLM-5.2-runtime

Resultat:

- `9 passed`
- `4 deselected`

## Tolkning

Efter att den nya runtime-versionen verkligen hade gått igång förbättrades RAG-kedjan tydligt i praktiken:

- legitima frågor passerade åter relevanskontrollen
- fallback användes inte längre för centrala frågor som `acceptanstest`, `arbetsområde` och `införandekrav`
- den nya modellen `zai-org/GLM-5.2` fungerade korrekt i den live deployade appen
