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
