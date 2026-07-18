# Reproduction of reported chatbot link inconsistency

Scope: reproduce the reported difference between these two Swedish chatbot queries:

1. `Finns det en arbetsmodell för införande av system`
2. `Finns det en införandeprocess för införande av system`

## Environment and method

I could not hit the deployed Hugging Face Space from this environment because DNS resolution for `helmfridsson-systeminforande.hf.space` failed. To still verify the current application behavior, I reproduced the response locally from the canonical repository at `/Users/tomashelmfridsson/workspace/systeminforande` by running the same RAG pipeline used by `build_rag_response()`:

- `rag.search.search(query, top_k=5)`
- `rag.grounding.filter_allowed_results(...)`
- `rag.synthesis.build_final_grounded_answer(..., enable_synthesis=False)`
- `rag.source_links.build_sources_md(results)`

This yields the current repository's answer text and `Relaterade hemsidor` block for each query.

## Exact reproduction steps

From the repo root:

```bash
python3 - <<'PY'
import os, sys
os.chdir('/Users/tomashelmfridsson/workspace/systeminforande')
sys.path.insert(0, os.getcwd())
from rag.search import search
from rag.grounding import filter_allowed_results
from rag.synthesis import build_final_grounded_answer
from rag.source_links import build_sources_md

queries = [
    'Finns det en arbetsmodell för införande av system',
    'Finns det en införandeprocess för införande av system',
]

for q in queries:
    results = filter_allowed_results(search(q, top_k=5))
    chunks = [chunk for _, chunk in results]
    synth = build_final_grounded_answer(
        q,
        chunks,
        enable_synthesis=False,
        llm_model='openai/gpt-oss-120b',
        llm_rewrite=None,
    )
    answer = str(synth['final_answer']) + build_sources_md(results)
    print('\n=== QUERY ===')
    print(q)
    print(answer)
PY
```

## Observed results

### Query 1
`Finns det en arbetsmodell för införande av system`

Full answer text:

```md
Ja, materialet innehåller underlag om arbetsmodell för införande av system. Det framgår av dokumentet "Verktyget_projektstyrning.pdf". Bedömningen bygger främst på Verktyget_projektstyrning.pdf (s. 5-6).

---

### Källor
- 📄 [Verktyget_projektstyrning.pdf](https://tomashelmfridsson.github.io/systeminforande/pdfs/Verktyget_projektstyrning.pdf)

### Relaterade hemsidor
- 🏠 [Verktyg](https://www.systeminforande.se/verktyg)
```

Observed `Relaterade hemsidor` list:

- Verktyg

Assessment of answer text:

- Acceptable in the narrow sense that it gives a grounded yes-answer and cites the PDF it relied on.
- It does not include the `Arbetsmodell` homepage link in `Relaterade hemsidor`.

### Query 2
`Finns det en införandeprocess för införande av system`

Full answer text:

```md
Materialet visar att detta framgår framför allt av avsnittet "2.1 Allmänt". Beskrivning för varje fas samt exempel på Det betonas också att och arbetsuppgifter kring ett system under. En viktig förutsättning är att ett lyckat införande är att personer med olika kompetenser är. För teknik och drift krävs kunskap om tekniska krav och driftkrav på nytt system. Bedömningen bygger främst på Verktyget_projektstyrning.pdf (s. 2) och Verktyget_och_systeminforandet.pdf (s. 3-5).

---

### Källor
- 📄 [Verktyget_projektstyrning.pdf](https://tomashelmfridsson.github.io/systeminforande/pdfs/Verktyget_projektstyrning.pdf)
- 📄 [Verktyget_och_systeminforandet.pdf](https://tomashelmfridsson.github.io/systeminforande/pdfs/Verktyget_och_systeminforandet.pdf)
- 📄 [Verktyget_aktiviteter.pdf](https://tomashelmfridsson.github.io/systeminforande/pdfs/Verktyget_aktiviteter.pdf)
- 📄 [Arbetsomraden_checklista.pdf](https://tomashelmfridsson.github.io/systeminforande/pdfs/Arbetsomraden_checklista.pdf)
- 📄 [Inforandekrav_kravmallen.pdf](https://tomashelmfridsson.github.io/systeminforande/pdfs/Inforandekrav_kravmallen.pdf)

### Relaterade hemsidor
- 🏠 [Verktyg](https://www.systeminforande.se/verktyg)
- 🏠 [Arbetsmodell](https://www.systeminforande.se/arbetsmodell)
- 🏠 [Implementering](https://www.systeminforande.se/implementering2)
- 🏠 [Införandekrav](https://www.systeminforande.se/infrandekrav-1)
- 🏠 [Checklistor och mallar](https://www.systeminforande.se/checklistor-och-mallar-till-verktyget-1)
```

Observed `Relaterade hemsidor` list:

- Verktyg
- Arbetsmodell
- Implementering
- Införandekrav
- Checklistor och mallar

Assessment of answer text:

- The answer is weaker than query 1 and contains malformed phrasing, but the related-link behavior matches the report.

## Conclusion

The reported link inconsistency is reproducible in the current repository logic:

- For `Finns det en arbetsmodell för införande av system`, the answer is acceptable enough as a grounded yes-answer, but `Arbetsmodell` is missing from `Relaterade hemsidor`.
- For `Finns det en införandeprocess för införande av system`, `Arbetsmodell` is present in `Relaterade hemsidor`.

Observed retrieval difference behind the behavior:

- Query 1 retrieved only `Verktyget_projektstyrning.pdf` results, which map to `Verktyg` but not directly to `Arbetsmodell`.
- Query 2 retrieved `Verktyget_och_systeminforandet.pdf` among its top results, and that source maps to the `Arbetsmodell` homepage link.
