import json
import os
from urllib.parse import quote
import gradio as gr

from rag.search import search
from rag.prompts import rag_prompt
from llm.reasoning import generate_reasoning
from llm.reasoning import generate_reasoning_from_prompt
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

def submit(message, doc_id, debug_mode):
    """
    Central router:
    - Om message matchar en underfråga → vanlig Q&A
    - Annars → RAG över PDF-material
    """

    message = message.strip()
    if not message:
        return "", "<h3>Svar</h3>"

    # 1️⃣ Försök matcha mot valt dokument (klassisk väg)
    if doc_id and doc_id in DOC_INDEX:
        doc = DOC_INDEX[doc_id]

        for q in doc["subquestions"]:
            if q["question"] == message:
                fact_answer = format_answer(q["answer"])
                source_results = search(f"{doc['title']} {message}", top_k=5)
                reasoning = safe_generate_reasoning(
                    title=doc["title"],
                    main_question=doc["main_question"],
                    question=message,
                    answer=q["answer"],
                )
                
                combined = (
                    "### Svar\n\n"
                    + fact_answer
                )

                if reasoning:
                    combined += (
                        "\n\n---\n\n"
                        + "### Resonemang\n\n"
                        + reasoning
                    )

                combined += build_sources_md(source_results)
                
                return combined, "<h3>Svar</h3>"
    
    # 2️⃣ Ingen match → RAG-fritext
    return handle_rag_query(message, debug_mode)

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


def format_llm_error(exc: Exception) -> str:
    message = str(exc).strip()
    if "model_not_supported" in message or "not supported by any provider" in message:
        return (
            "Resonemang kunde inte genereras just nu eftersom den konfigurerade "
            "språkmodellen inte är tillgänglig för nuvarande Hugging Face-provider."
        )

    return f"Resonemang kunde inte genereras just nu. Tekniskt fel: {message}"


def safe_generate_reasoning(**kwargs) -> str:
    try:
        return generate_reasoning(**kwargs)
    except Exception as exc:
        print(f"LLM-fel i generate_reasoning: {exc}")
        return format_llm_error(exc)


def safe_generate_reasoning_from_prompt(prompt: str) -> str:
    try:
        return generate_reasoning_from_prompt(prompt)
    except Exception as exc:
        print(f"LLM-fel i generate_reasoning_from_prompt: {exc}")
        return (
            "Det gick inte att generera ett sammanhängande resonemang just nu.\n\n"
            + format_llm_error(exc)
        )


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
    return [], "", "", None

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
    
def handle_rag_query(query: str, debug: bool):
    results = search(query, top_k=5)

    if not results:
        return (
            "Det finns inget tillräckligt underlag i materialet för att besvara frågan.",
            "<h3>Svar</h3>"
        )

    # -----------------------------
    # Confidence score
    # -----------------------------
    scores = [score for score, _ in results]
    confidence = round(sum(scores) / len(scores), 2)

    chunks = [chunk for _, chunk in results]

    # -----------------------------
    # Generera svar
    # -----------------------------
    prompt = rag_prompt(query=query, chunks=chunks)
    answer = safe_generate_reasoning_from_prompt(prompt)

    sources_md = build_sources_md(results)

    # -----------------------------
    # Debug (valfritt)
    # -----------------------------
    debug_md = ""
    if debug:
        debug_lines = [
            "\n\n---\n\n### Debug",
            f"**Confidence:** {confidence}",
            ""
        ]

        for score, c in results:
            debug_lines.append(
                f"""**📄 Källa:** {c['source']}
- **Typ:** {c.get('source_type')}
- **Rubrik:** {c.get('title')}
- **Sidor:** {c.get('pages')}
- **Score:** `{round(score, 4)}`

{c['text'][:500]}{'…' if len(c['text']) > 500 else ''}
---
"""
            )

        debug_md = "\n".join(debug_lines)

    # -----------------------------
    # Slutligt svar
    # -----------------------------
    final_answer = answer + sources_md + debug_md
    return final_answer, "<h3>Svar</h3>"
    
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
    
        # HÖGER: Meddelande
        with gr.Column(scale=3):
            gr.Markdown("<h3>Meddelande</h3>")
            message = gr.Textbox(
                placeholder="Välj ett område, klicka på en underfråga och tryck på Skicka.",
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

    # RAD 2 – Svar över hela bredden
    with gr.Row():
        with gr.Column():
            answer_title = gr.Markdown(
                "<h3>Svar</h3>",
                elem_classes="answer-title"
            )
            
            answer = gr.Markdown(
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
        fn=fill_message,
        outputs=message
    )

    send_btn.click(
        fn=submit,
        inputs=[message, current_doc, debug_mode],
        outputs=[answer, answer_title]
    )
    
    message.submit(
        fn=submit,
        inputs=[message, current_doc, debug_mode],
        outputs=[answer, answer_title]
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[questions, message, answer, current_doc]
    )

# =====================================================
# LAUNCH
# =====================================================

with open("style.css", encoding="utf-8") as f:
    css = f.read()

demo.launch(theme=None,css=css, ssr_mode=False)
