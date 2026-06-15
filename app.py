import json
import os
import time
from urllib.parse import quote
import gradio as gr

from rag.extractive import build_extractive_reasoning
from rag.prompts import rag_prompt
from rag.search import explain_search, search
from llm.reasoning import generate_reasoning, generate_reasoning_from_prompt
DATA_DIR = "rag/data"
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
GITHUB_PAGES_BASE_URL = "https://tomashelmfridsson.github.io/systeminforande"
GITHUB_PAGES_PDF_BASE_URL = f"{GITHUB_PAGES_BASE_URL}/pdfs"
HEADER_IMAGE_URL = (
    "https://raw.githubusercontent.com/"
    "tomashelmfridsson/systeminforande/main/brain.jpg"
)

if not os.path.exists(CHUNKS_FILE):
    raise FileNotFoundError(
        "rag/data/chunks.json saknas. Bygg indexet innan appen startas."
    )


print("HF_TOKEN present:", bool(os.getenv("HF_TOKEN")))
print("HF_TOKEN length:", len(os.getenv("HF_TOKEN", "")))

# =====================================================
# DATA
# =====================================================

with open("content.json", encoding="utf-8") as f:
    DOCUMENTS = json.load(f)["documents"]

DOC_INDEX = {d["id"]: d for d in DOCUMENTS}
LLM_MODEL_OPTIONS = [
    ("Qwen 2.5 1.5B Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("DeepSeek R1 cheapest", "deepseek-ai/DeepSeek-R1:cheapest"),
    ("OpenAI gpt-oss 120B cheapest", "openai/gpt-oss-120b:cheapest"),
]

# =====================================================
# FUNKTIONER
# =====================================================

def load_document(doc_id):
    rows = [[q["question"]] for q in DOC_INDEX[doc_id]["subquestions"]]
    return rows, doc_id


def fill_message(evt: gr.SelectData):
    value = evt.value
    if isinstance(value, list):
        return value[0]
    return value


def select_and_submit(evt: gr.SelectData, doc_id, debug_mode, llm_model):
    message = fill_message(evt)
    for structured_answer, llm_answer in submit(message, doc_id, debug_mode, llm_model):
        yield "", structured_answer, llm_answer

def submit(message, doc_id, debug_mode, llm_model):
    """
    Central router:
    - Om message matchar en underfråga → vanlig Q&A
    - Annars → RAG över PDF-material
    """

    message = message.strip()
    if not message:
        yield "", ""
        return

    # 1️⃣ Försök matcha mot valt dokument (klassisk väg)
    if doc_id and doc_id in DOC_INDEX:
        doc = DOC_INDEX[doc_id]

        for q in doc["subquestions"]:
            if q["question"] == message:
                fact_answer = format_answer(q["answer"])
                source_results = search(f"{doc['title']} {message}", top_k=5)
                structured_start = time.perf_counter()
                model_reasoning = build_structured_reasoning(
                    question=message,
                    answer=q["answer"],
                )
                structured_elapsed = time.perf_counter() - structured_start
                structured_combined = "### Svar\n\n" + fact_answer

                if model_reasoning:
                    structured_combined += "\n\n### Resonemang\n\n" + model_reasoning

                structured_combined += f"\n\n_Svarstid: {structured_elapsed:.2f} s_"

                sources_md = build_sources_md(source_results)
                structured_combined += sources_md

                if debug_mode:
                    structured_combined += build_predefined_debug_md(
                        question=message,
                        reasoning=model_reasoning,
                        source_results=source_results,
                        llm_answer=None,
                    )
                yield structured_combined, "_Bearbetar LLM-svar..._"

                llm_start = time.perf_counter()
                llm_reasoning = safe_generate_reasoning(
                    title=doc["title"],
                    main_question=doc["main_question"],
                    question=message,
                    answer=q["answer"],
                    model=llm_model,
                )
                llm_elapsed = time.perf_counter() - llm_start

                llm_combined = ""
                if llm_reasoning:
                    llm_combined += "### Resonemang\n\n" + llm_reasoning
                llm_combined += f"\n\n_Svarstid: {llm_elapsed:.2f} s_"
                llm_combined += sources_md

                if debug_mode:
                    llm_combined += build_predefined_debug_md(
                        question=message,
                        reasoning=llm_reasoning,
                        source_results=source_results,
                        llm_answer=llm_reasoning,
                    )

                yield structured_combined, llm_combined
                return
    
    # 2️⃣ Ingen match → RAG-fritext
    yield from handle_rag_query(message, debug_mode, llm_model)

def format_answer(answer):
    out = []
    for key, value in answer.items():
        out.append(f"**{key}**")
        if isinstance(value, list):
            for item in value:
                out.append(f"- {item}")
        else:
            out.append(value)
        out.append("")
    return "\n".join(out)


def build_structured_reasoning(question: str, answer: dict) -> str:
    """
    Bygg resonemang för fördefinierade frågor från det kuraterade faktasvaret.
    Detta ska ge ett stabilt stycke även när retrievalen inte räcker för fritext-RAG.
    """
    parts = []
    description = _as_text(answer.get("Beskrivning"))
    purpose = _as_text(answer.get("Syfte"))
    result = _as_text(answer.get("Resultat"))
    examples = _as_list(answer.get("Exempel"))
    items = _as_list(answer.get("Lista"))

    if purpose:
        parts.append(purpose)

    if description:
        parts.append(description)

    if items:
        if len(items) == 1:
            parts.append(f"Detta omfattar särskilt {items[0].lower()}.")
        else:
            listed = ", ".join(items[:-1]) + f" och {items[-1]}" if len(items) > 1 else items[0]
            parts.append(f"Detta omfattar bland annat {listed.lower()}.")

    if examples:
        if len(examples) == 1:
            parts.append(f"Ett exempel är {examples[0].lower()}.")
        else:
            parts.append(
                "Exempel på detta är "
                + _join_list([example.lower() for example in examples[:3]])
                + "."
            )

    if result:
        parts.append(f"Det bidrar till {result.lower()}.")

    if not parts:
        return ""

    reasoning = " ".join(_ensure_sentence(part) for part in parts)
    return _normalize_reasoning_text(reasoning)


def _as_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _ensure_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text


def _join_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" och {items[-1]}"


def _normalize_reasoning_text(text: str) -> str:
    text = " ".join(text.split())
    text = text.replace("..", ".")
    return text


def safe_generate_reasoning(**kwargs) -> str:
    try:
        return generate_reasoning(**kwargs)
    except Exception as exc:
        print(f"LLM-fel i generate_reasoning: {exc}")
        return format_llm_error(exc)


def safe_generate_reasoning_from_prompt(prompt: str) -> str:
    return safe_generate_reasoning_from_prompt_with_model(prompt, None)


def safe_generate_reasoning_from_prompt_with_model(prompt: str, model: str | None) -> str:
    try:
        return generate_reasoning_from_prompt(prompt, model=model)
    except Exception as exc:
        print(f"LLM-fel i generate_reasoning_from_prompt: {exc}")
        return format_llm_error(exc)


def format_llm_error(exc: Exception) -> str:
    message = str(exc).strip()
    if "model_not_supported" in message or "not supported by any provider" in message:
        return (
            "Resonemang kunde inte genereras just nu eftersom den konfigurerade "
            "språkmodellen inte är tillgänglig för nuvarande Hugging Face-provider."
        )

    return f"Resonemang kunde inte genereras just nu. Tekniskt fel: {message}"


def build_sources_md(results) -> str:
    used_sources = {}
    for _, chunk in results:
        used_sources[chunk["source"]] = chunk

    if not used_sources:
        return ""

    sources_lines = ["\n\n---\n\n### Källor"]
    for chunk in used_sources.values():
        sources_lines.append(f"- {format_source_link(chunk)}")

    return "\n".join(sources_lines)


def clear_all():
    return [], "", "", "", None

def format_pages(pages):
    if not pages:
        return ""

    pages = sorted(set(pages))

    if len(pages) == 1:
        return f"s. {pages[0]}"

    # sammanhängande intervall
    if pages[-1] - pages[0] + 1 == len(pages):
        return f"s. {pages[0]}–{pages[-1]}"

    return "s. " + ", ".join(str(p) for p in pages)
    
def format_source_link(chunk: dict) -> str:
    source = chunk.get("source", "Okänd källa")
    source_type = chunk.get("source_type")
    pages = chunk.get("pages")

    if source_type == "pdf":
        page_info = format_pages(pages)
        encoded_source = quote(source)
        return (
            f"📄 "
            f"[{source}]("
            f"{GITHUB_PAGES_PDF_BASE_URL}/{encoded_source}"
            f")"
            f"{' — ' + page_info if page_info else ''}"
        )

    if source_type == "web":
        return f"🌐 [{source}]({source})"

    return source
    
def handle_rag_query(query: str, debug: bool, llm_model: str):
    results = search(query, top_k=5)

    if not results:
        no_data = "Det finns inget tillräckligt underlag i materialet för att besvara frågan."
        yield no_data, no_data
        return

    # -----------------------------
    # Confidence score
    # -----------------------------
    scores = [score for score, _ in results]
    confidence = round(sum(scores) / len(scores), 2)

    chunks = [chunk for _, chunk in results]

    structured_start = time.perf_counter()
    structured_answer = build_extractive_reasoning(query, chunks)
    structured_elapsed = time.perf_counter() - structured_start
    llm_prompt = rag_prompt(query, chunks)

    sources_md = build_sources_md(results)
    search_debug = explain_search(query, top_k=5)

    # -----------------------------
    # Debug (valfritt)
    # -----------------------------
    model_debug_md = ""
    if debug:
        model_debug_lines = [
            "\n\n---\n\n### Debug",
            f"**Confidence:** {confidence}",
            f"**Frågetyp:** {translate_intent(search_debug['intent'])}",
            f"**Query-termer:** `{', '.join(search_debug['query_terms'])}`",
            f"**Diagnos:** {diagnose_retrieval(structured_answer, search_debug)}",
            ""
        ]

        for item in search_debug["top_results"]:
            score = item["score"]
            c = item["chunk"]
            parts = item["parts"]
            block = (
                f"""**📄 Källa:** {c['source']}
- **Typ:** {c.get('source_type')}
- **Rubrik:** {c.get('title')}
- **Sidor:** {c.get('pages')}
- **Score:** `{round(score, 4)}`
- **BM25:** `{parts['bm25']}`
- **Titelboost:** `{parts['title_overlap']}`
- **Definitionsboost:** `{parts['definition_boost']}`
- **Domänboost:** `{parts['domain_boost']}`
- **Intentboost:** `{parts['intent_boost']}`

{c['text'][:500]}{'…' if len(c['text']) > 500 else ''}
---
"""
            )
            model_debug_lines.append(block)

        model_debug_md = "\n".join(model_debug_lines)

    # -----------------------------
    # Slutligt svar
    # -----------------------------
    final_structured_answer = (
        structured_answer
        + f"\n\n_Svarstid: {structured_elapsed:.2f} s_"
        + sources_md
        + model_debug_md
    )
    yield final_structured_answer, "_Bearbetar LLM-svar..._"

    llm_start = time.perf_counter()
    llm_answer = safe_generate_reasoning_from_prompt_with_model(llm_prompt, llm_model)
    llm_elapsed = time.perf_counter() - llm_start
    llm_debug_md = ""
    if debug:
        llm_debug_lines = [
            "\n\n---\n\n### Debug",
            f"**Confidence:** {confidence}",
            f"**Frågetyp:** {translate_intent(search_debug['intent'])}",
            f"**Query-termer:** `{', '.join(search_debug['query_terms'])}`",
            f"**Diagnos:** {diagnose_retrieval(llm_answer, search_debug)}",
            f"**LLM-status:** {diagnose_llm_status(llm_answer)}",
            ""
        ]

        for item in search_debug["top_results"]:
            score = item["score"]
            c = item["chunk"]
            parts = item["parts"]
            block = (
                f"""**📄 Källa:** {c['source']}
- **Typ:** {c.get('source_type')}
- **Rubrik:** {c.get('title')}
- **Sidor:** {c.get('pages')}
- **Score:** `{round(score, 4)}`
- **BM25:** `{parts['bm25']}`
- **Titelboost:** `{parts['title_overlap']}`
- **Definitionsboost:** `{parts['definition_boost']}`
- **Domänboost:** `{parts['domain_boost']}`
- **Intentboost:** `{parts['intent_boost']}`

{c['text'][:500]}{'…' if len(c['text']) > 500 else ''}
---
"""
            )
            llm_debug_lines.append(block)

        llm_debug_md = "\n".join(llm_debug_lines)

    final_llm_answer = (
        llm_answer
        + f"\n\n_Svarstid: {llm_elapsed:.2f} s_"
        + sources_md
        + llm_debug_md
    )
    yield final_structured_answer, final_llm_answer


def translate_intent(intent: str) -> str:
    labels = {
        "definition": "Definition",
        "purpose": "Syfte",
        "overview_list": "Översiktslista",
        "list": "Lista",
        "process": "Process",
        "timing_or_decision": "Tidpunkt eller beslut",
        "general": "Allmän fråga",
    }
    return labels.get(intent, intent)


def diagnose_retrieval(answer: str, search_debug: dict) -> str:
    top_results = search_debug.get("top_results", [])
    if not top_results:
        return "Inga träffar hittades."

    top_score = top_results[0]["score"]
    intent = search_debug.get("intent")
    fallback = "Det finns inte tillräckligt tydligt underlag" in answer

    if fallback and top_score >= 8:
        return (
            "Retrievalen verkar relativt stark, men syntesen lyckades inte bygga ett "
            "pålitligt resonemang från chunkarna."
        )

    if top_score < 3:
        return "Retrievalen verkar svag. De högsta träffarna är sannolikt inte tillräckligt relevanta."

    if intent in {"overview_list", "list"} and fallback:
        return (
            "Frågan ser ut som en lista eller översiktsfråga. Retrievalen hittade material, "
            "men chunkarna verkar inte vara tillräckligt välstrukturerade för nuvarande syntes."
        )

    return "Retrievalen verkar rimlig för frågetypen."


def diagnose_llm_status(answer: str) -> str:
    if "språkmodellen inte är tillgänglig" in answer:
        return "LLM-syntesen kunde inte köras med nuvarande Hugging Face-provider."
    if "Tekniskt fel" in answer:
        return "LLM-syntesen gav tekniskt fel."
    return "LLM-syntesen genererades."


def build_predefined_debug_md(question: str, reasoning: str, source_results, llm_answer: str | None) -> str:
    debug_lines = [
        "\n\n---\n\n### Debug",
        f"**Frågetyp:** Fördefinierad fråga",
        f"**Fråga:** {question}",
        f"**Antal källträffar:** {len(source_results)}",
    ]

    if llm_answer is not None:
        debug_lines.append(f"**LLM-status:** {diagnose_llm_status(llm_answer)}")

    fallback = "Det finns inte tillräckligt tydligt underlag" in (reasoning or "")
    debug_lines.append(
        "**Diagnos:** "
        + (
            "Resonemanget bygger på kuraterat svarsinnehåll."
            if not fallback
            else "Resonemanget föll tillbaka trots att frågan är fördefinierad."
        )
    )
    debug_lines.append("")

    for score, chunk in source_results:
        debug_lines.append(
            f"""**📄 Källa:** {chunk['source']}
- **Rubrik:** {chunk.get('title')}
- **Sidor:** {chunk.get('pages')}
- **Score:** `{round(score, 4)}`
"""
        )

    return "\n".join(debug_lines)
    
# =====================================================
# UI
# =====================================================

# with gr.Blocks(css=".gradio-container {background-color: white}") as demo:
with gr.Blocks() as demo:
    gr.HTML("<h1 class='title'>Citrus-chatbot</h1>")

    gr.Image(
        value=HEADER_IMAGE_URL,
        show_label=False,
        interactive=False,
        elem_classes="brain-header"
    )

    current_doc = gr.State(None)
    
    # -------------------------
    # HUVUDFRÅGOR
    # -------------------------
    with gr.Row():
        main_buttons = []
    
        for doc in DOCUMENTS:
            with gr.Column(elem_classes="card"):
                gr.HTML(
                    f"""
                    <div class="card-content">
                        <div class="card-title">{doc["title"]}</div>
                        <div class="card-question">{doc["main_question"]}</div>
                    </div>
                    """
                )
    
                btn = gr.Button(
                    "",
                    elem_classes="card-overlay"
                )
    
                main_buttons.append((btn, doc["id"]))

    # -------------------------
    # INNEHÅLL
    # -------------------------
    with gr.Row():
        
        # VÄNSTER: Underfrågor
        with gr.Column(scale=2):
            gr.Markdown("<h3>Underfrågor</h3>")
            questions = gr.Dataframe(
                headers=[""],
                interactive=False,
                elem_classes="question-list"
            )
    
        # HÖGER: Fritextfråga
        with gr.Column(scale=3):
            gr.Markdown("<h3>Egen fråga</h3>")
            message = gr.Textbox(
                placeholder="Skriv en fritextfråga här om du vill söka i källmaterialet.",
                lines=1,
                label=None,  
                show_label=False,
                elem_classes="message-box"
            )
    
            with gr.Row():
                send_btn = gr.Button("Skicka", elem_classes="send-btn")
                clear_btn = gr.Button("Rensa", elem_classes="send-btn")
                debug_mode = gr.Checkbox(
                    label="Debug",
                    value=False
                )

            llm_model = gr.Dropdown(
                label="LLM-modell (experimentell)",
                choices=LLM_MODEL_OPTIONS,
                value="Qwen/Qwen2.5-1.5B-Instruct",
                interactive=True,
            )

    # RAD 2 – Svar över hela bredden
    with gr.Row():
        with gr.Column():
            gr.Markdown("<h3>Strukturerad</h3>")
            answer_structured = gr.Markdown(
                "",
                elem_classes="answer-box"
            )

        with gr.Column():
            gr.Markdown("<h3>LLM-baserad</h3>")
            answer_llm = gr.Markdown(
                "",
                elem_classes="answer-box"
            )
            
    # -------------------------
    # EVENTS
    # -------------------------

    for btn, doc_id in main_buttons:
        btn.click(
            fn=lambda d=doc_id: load_document(d),
            outputs=[questions, current_doc]
        )

    questions.select(
        fn=select_and_submit,
        inputs=[current_doc, debug_mode, llm_model],
        outputs=[message, answer_structured, answer_llm]
    )

    send_btn.click(
        fn=submit,
        inputs=[message, current_doc, debug_mode, llm_model],
        outputs=[answer_structured, answer_llm]
    )
    
    message.submit(
        fn=submit,
        inputs=[message, current_doc, debug_mode, llm_model],
        outputs=[answer_structured, answer_llm]
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[questions, message, answer_structured, answer_llm, current_doc]
    )

# =====================================================
# LAUNCH
# =====================================================

with open("style.css", encoding="utf-8") as f:
    css = f.read()

demo.launch(theme=None,css=css, ssr_mode=False)
