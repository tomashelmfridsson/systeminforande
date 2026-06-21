---
title: Systeminforande
emoji: 🦀
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 6.2.0
python_version: 3.11
app_file: app.py
pinned: false
license: apache-2.0
short_description: Sida för systeminforande
---

Session guidelines for coding work live in `AI_CODING_GUIDELINES.md`.
En sammanfattning av labben, arkitekturen och viktiga tekniska val finns i
`RAG_LAB_LESSONS.md`.

PDF-källor för publika länkar ligger under `docs/pdfs/`.
Appen bygger käll-länkar mot GitHub Pages på
`https://tomashelmfridsson.github.io/systeminforande/pdfs/<filnamn>`.
GitHub Pages bör deployas via workflow i `.github/workflows/deploy-pages.yml`
så att LFS-spårade PDF-filer publiceras som riktiga filer och inte som LFS-pekarfiler.
I GitHub-repot behöver Pages vara satt till `GitHub Actions` som build source.
Hugging Face Space kan deployas från GitHub via `.github/workflows/deploy-huggingface.yml`
med en repo-secret `HF_TOKEN`.
