# Live API Baseline

- Datum: `2026-07-10`
- Deploy-status: före ny deploy
- Modell: `openai/gpt-oss-120b`
- Bas-URL: `https://helmfridsson-systeminforande.hf.space`
- Kommando:
  `SYSTEMINFORANDE_BASE_URL=https://helmfridsson-systeminforande.hf.space SYSTEMINFORANDE_LLM_MODEL=openai/gpt-oss-120b pytest -m live_api -q`

## Sammanfattning

- `6 passed`
- `3 failed`
- `4 deselected`

## Felande scenarier

### 1. `acceptance_test`

- Fråga:
  `Hur används acceptanstest i införandet?`
- Förväntade källor:
  - `210_Acceptanstest_testplan.pdf`
  - `222_Acceptanstest_testplan.pdf`
  - `231_Acceptanstest_krav_leveransgodkannande.pdf`
- Faktiska källor:
  - `220_Acceptanstest_delprojektplan.pdf`
  - `240_Utbildning_strategi.pdf`
  - `400_Teknisk_plattform_plan.pdf`
  - `Checklista_Arbetsomraden.pdf`
  - `Checklista_Inforandekrav.pdf`

### 2. `work_area_definition`

- Fråga:
  `Vad är ett arbetsområde?`
- Förväntad källa:
  - `Checklista_Arbetsomraden.pdf`
- Faktisk källa:
  - `210_Acceptanstest_testplan.pdf`

### 3. `misspelling_robustness_acceptance_test`

- Kanonisk fråga:
  `Hur används acceptanstest i införandet?`
- Fråga med stavfel:
  `Hur anvnds acceptanstst i införandet?`
- Förväntat:
  svaret ska fortfarande innehålla `acceptanstest`
- Faktiskt:
  `Frågan verkar inte ha relevant stöd i det tillgängliga källmaterialet.`

## Tolkning

Detta baseline-resultat visar den tidigare live-deployens tre tydliga svagheter:

- fel dokumentfamilj för `acceptanstest`
- fel dokument för definitionen av `arbetsområde`
- bristande robusthet för vanliga stavfel i svenska nyckelord
