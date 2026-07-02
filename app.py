import json
import os
import re
import time
from urllib.parse import quote
import gradio as gr
import requests

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
DEPLOY_REVISION_FILE = "deploy_revision.txt"


def load_deploy_revision() -> str:
    try:
        with open(DEPLOY_REVISION_FILE, encoding="utf-8") as revision_file:
            return revision_file.read().strip() or "local"
    except OSError:
        return "local"


DEPLOY_REVISION = load_deploy_revision()

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

HF_MODELS_ENDPOINT = "https://router.huggingface.co/v1/models"
FALLBACK_LLM_MODELS = [
    {
        "id": "google/gemma-2-2b-it",
        "label": "Gemma 2 2B IT",
        "description": "Liten instruktionsmodell, sannolikt snabbare.",
    },
    {
        "id": "deepseek-ai/DeepSeek-R1",
        "label": "DeepSeek R1",
        "description": "Resonemangsmodell, ofta långsammare.",
    },
    {
        "id": "openai/gpt-oss-120b",
        "label": "OpenAI gpt-oss 120B",
        "description": "Stor modell med högre latens.",
    },
    {
        "id": "zai-org/GLM-4.5",
        "label": "GLM 4.5",
        "description": "Stark generell textmodell.",
    },
    {
        "id": "Qwen/Qwen3-4B-Thinking-2507",
        "label": "Qwen3 4B Thinking",
        "description": "Liten resonemangsmodell.",
    },
]
FALLBACK_MODEL_DESCRIPTIONS = {
    model["id"]: model["description"] for model in FALLBACK_LLM_MODELS
}


def _format_model_label(model_id: str, description: str = "") -> str:
    if not description:
        return model_id
    return f"{model_id} - {description}"


def _choice_from_model(model_id: str, description: str = "") -> tuple[str, str]:
    return (_format_model_label(model_id, description), model_id)


def load_llm_model_options() -> list[tuple[str, str]]:
    headers = {}
    token = os.getenv("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(HF_MODELS_ENDPOINT, headers=headers, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"Kunde inte hämta modellista från HF API: {exc}")
        return [
            _choice_from_model(model["id"], model["description"])
            for model in FALLBACK_LLM_MODELS
        ]

    if not isinstance(payload, list):
        print("HF API returnerade oväntat format för modellistan.")
        return [
            _choice_from_model(model["id"], model["description"])
            for model in FALLBACK_LLM_MODELS
        ]

    available = {}
    for item in payload:
        if not isinstance(item, dict):
            continue

        model_id = (
            item.get("id")
            or item.get("model")
            or item.get("name")
        )
        if not isinstance(model_id, str) or not model_id.strip():
            continue

        description = (
            item.get("description")
            or item.get("summary")
            or FALLBACK_MODEL_DESCRIPTIONS.get(model_id, "")
        )
        available[model_id] = _choice_from_model(model_id, str(description).strip())

    if not available:
        print("HF API gav ingen användbar modellista.")
        return [
            _choice_from_model(model["id"], model["description"])
            for model in FALLBACK_LLM_MODELS
        ]

    curated = []
    seen = set()

    for model in FALLBACK_LLM_MODELS:
        model_id = model["id"]
        if model_id in available:
            curated.append(available[model_id])
            seen.add(model_id)

    extra_candidates = []
    for model_id, choice in available.items():
        lowered = model_id.lower()
        if model_id in seen:
            continue
        if any(keyword in lowered for keyword in ["instruct", "chat", "it", "thinking", "r1", "gemma", "glm", "gpt-oss"]):
            extra_candidates.append(choice)

    extra_candidates.sort(key=lambda choice: choice[1].lower())
    curated.extend(extra_candidates[:10])

    return curated or [
        _choice_from_model(model["id"], model["description"])
        for model in FALLBACK_LLM_MODELS
    ]


LLM_MODEL_OPTIONS = load_llm_model_options()
PREFERRED_LLM_MODEL = "openai/gpt-oss-120b"
DEFAULT_LLM_MODEL = (
    PREFERRED_LLM_MODEL
    if any(choice[1] == PREFERRED_LLM_MODEL for choice in LLM_MODEL_OPTIONS)
    else (LLM_MODEL_OPTIONS[0][1] if LLM_MODEL_OPTIONS else FALLBACK_LLM_MODELS[0]["id"])
)

# =====================================================
# FUNKTIONER
# =====================================================

def build_main_card_html(doc: dict, selected: bool = False) -> str:
    selected_class = " selected" if selected else ""
    return f"""
    <div class="card-shell{selected_class}">
        <div class="card-content">
            <div class="card-title">{doc["title"]}</div>
            <div class="card-question">{doc["main_question"]}</div>
        </div>
    </div>
    """


def build_main_card_updates(selected_doc_id: str | None):
    updates = []
    for doc in DOCUMENTS:
        updates.append(
            gr.update(value=build_main_card_html(doc, selected=doc["id"] == selected_doc_id))
        )
    return updates


def load_document(doc_id):
    questions = [q["question"] for q in DOC_INDEX[doc_id]["subquestions"]]
    return (
        gr.update(choices=questions, value=None),
        "",
        doc_id,
        *build_main_card_updates(doc_id),
    )


def select_and_submit(message: str, doc_id, debug_mode, llm_model):
    for answer in submit(message, doc_id, debug_mode, llm_model):
        yield answer

def submit(message, doc_id, debug_mode, llm_model):
    """
    Central router:
    - Om message matchar en underfråga → vanlig Q&A
    - Annars → RAG över PDF-material
    """

    if not message:
        yield ""
        return

    message = message.strip()
    if not message:
        yield ""
        return

    # 1️⃣ Försök matcha mot valt dokument (klassisk väg)
    if doc_id and doc_id in DOC_INDEX:
        doc = DOC_INDEX[doc_id]

        for q in doc["subquestions"]:
            if q["question"] == message:
                fact_answer = format_answer(q["answer"])

                if debug_mode:
                    source_results = search(f"{doc['title']} {message}", top_k=5)
                    fact_answer += build_predefined_debug_md(
                        question=message,
                        reasoning="",
                        source_results=source_results,
                        llm_answer=None,
                    )
                yield fact_answer
                return
    
    # 2️⃣ Ingen match → RAG-fritext
    yield from handle_rag_query(message, debug_mode, llm_model)

def format_answer(answer):
    return "\n".join(_format_answer_sections(answer)).strip()


def _format_answer_sections(answer: dict, level: int = 0) -> list[str]:
    out = []
    for key, value in answer.items():
        heading_prefix = "#" * min(level + 4, 6)
        if level == 0:
            out.append(f"**{key}**")
        else:
            out.append(f"{heading_prefix} {key}")

        out.extend(_format_answer_value(value, level))
        out.append("")
    return out


def _format_answer_value(value, level: int) -> list[str]:
    if isinstance(value, dict):
        return _format_answer_sections(value, level + 1)

    if isinstance(value, list):
        if _is_table_rows(value):
            return _format_markdown_table(value)

        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.extend(_format_answer_sections(item, level + 1))
            else:
                lines.append(f"- {item}")
        return lines

    return [str(value)]


def _is_table_rows(value: list) -> bool:
    if not value or not all(isinstance(item, dict) for item in value):
        return False

    columns = list(value[0].keys())
    if not columns:
        return False

    return all(list(item.keys()) == columns for item in value)


def _format_markdown_table(rows: list[dict]) -> list[str]:
    columns = list(rows[0].keys())
    table = [
        "| " + " | ".join(_format_markdown_cell(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in rows:
        table.append(
            "| "
            + " | ".join(_format_markdown_cell(row.get(column, "")) for column in columns)
            + " |"
        )

    return table


def _format_markdown_cell(value) -> str:
    text = str(value).strip()
    text = " ".join(text.split())
    return text.replace("|", "\\|")


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
    lowered = message.lower()

    if "429" in message or "too many requests" in lowered:
        return (
            "LLM-modellen är tillfälligt överbelastad. "
            "Försök igen lite senare."
        )

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
    return gr.update(choices=[], value=None), "", "", None, *build_main_card_updates(None)


def clear_chatbot():
    return "", ""

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


def _simple_tokenize(text: str) -> set[str]:
    tokens = set()
    for raw_token in re.findall(r"\w+", (text or "").lower()):
        token = raw_token
        if token.endswith("erna") and len(token) > 6:
            token = token[:-1]
        elif token.endswith("arna") and len(token) > 6:
            token = token[:-1]
        elif token.endswith("or") and len(token) > 5:
            token = token[:-2]
        elif token.endswith("ar") and len(token) > 5:
            token = token[:-2]
        elif token.endswith("er") and len(token) > 5:
            token = token[:-2]
        elif token.endswith("en") and len(token) > 5:
            token = token[:-2]
        elif token.endswith("n") and len(token) > 5:
            token = token[:-1]

        if len(token) > 2:
            tokens.add(token)
            if token.endswith("område"):
                tokens.add(token[:-1])
            elif token.endswith("områd"):
                tokens.add(f"{token}e")

    return tokens


def _has_relevant_rag_support(search_debug: dict) -> tuple[bool, str]:
    top_results = search_debug.get("top_results", [])
    query_terms = set(search_debug.get("query_terms", []))

    if not top_results or not query_terms:
        return False, "Inga tydliga träffar hittades i materialet."

    top_score = top_results[0]["score"]
    if top_score < 3:
        return False, "De högsta träffarna är för svaga för att betraktas som relevanta."

    combined_tokens = set()
    for item in top_results[:3]:
        chunk = item["chunk"]
        combined_tokens |= _simple_tokenize(chunk.get("title", ""))
        combined_tokens |= _simple_tokenize(chunk.get("text", ""))

    matched_terms = query_terms & combined_tokens
    if not matched_terms:
        return False, "Frågans nyckelord återfinns inte i de högst rankade träffarna."

    coverage = len(matched_terms) / len(query_terms)
    if len(query_terms) >= 3 and coverage < 0.34:
        return False, "För liten del av frågans nyckelord stöds av de högst rankade träffarna."

    return True, ""
    
def handle_rag_query(query: str, debug: bool, llm_model: str):
    results = search(query, top_k=5)

    if not results:
        no_data = "Det finns inget tillräckligt underlag i materialet för att besvara frågan."
        yield no_data
        return

    # -----------------------------
    # Confidence score
    # -----------------------------
    scores = [score for score, _ in results]
    confidence = round(sum(scores) / len(scores), 2)

    search_debug = explain_search(query, top_k=5)
    is_relevant, relevance_reason = _has_relevant_rag_support(search_debug)
    if not is_relevant:
        no_data = "Frågan verkar inte ha relevant stöd i det tillgängliga källmaterialet."
        if debug:
            no_data += (
                "\n\n---\n\n### Debug\n"
                f"**Confidence:** {confidence}\n"
                f"**Frågetyp:** {translate_intent(search_debug['intent'])}\n"
                f"**Query-termer:** `{', '.join(search_debug['query_terms'])}`\n"
                f"**Diagnos:** {relevance_reason}"
            )
        yield no_data
        return

    chunks = [chunk for _, chunk in results]
    structured_answer = build_extractive_reasoning(query, chunks)
    llm_prompt = rag_prompt(query, chunks)
    sources_md = build_sources_md(results)

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
    yield final_llm_answer


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
    gr.HTML(
        f"""
        <div class="app-header-row">
            <h1 class="title">Digitalt Bollplank 24/7</h1>
            <img
                src="{HEADER_IMAGE_URL}"
                alt="Digitalt Bollplank"
                class="app-header-logo"
            />
        </div>
        """
    )

    current_doc = gr.State(None)
    
    with gr.Tabs() as tabs:
        with gr.Tab("FAQ"):
            gr.Markdown("<p class='tab-intro'>Välj ämnesområde och underfråga.</p>")

            with gr.Row():
                main_buttons = []

                for doc in DOCUMENTS:
                    with gr.Column(min_width=260, elem_classes="card"):
                        card_html = gr.HTML(build_main_card_html(doc))

                        btn = gr.Button(
                            "",
                            elem_classes="card-overlay"
                        )

                        main_buttons.append((btn, doc["id"], card_html))

            gr.Markdown("<h3>Underfrågor</h3>")
            questions = gr.Radio(
                choices=[],
                value=None,
                label="",
                show_label=False,
                interactive=True,
                elem_classes="question-list"
            )

            gr.Markdown("<h3>Svar</h3>")
            faq_answer = gr.Markdown(
                "",
                elem_classes="answer-box"
            )

        with gr.Tab("CHATTBOT"):
            message = gr.Textbox(
                placeholder="Skriv en fritextfråga här om du vill söka i källmaterialet.",
                lines=6,
                label=None,
                show_label=False,
                elem_classes="message-box"
            )

            with gr.Row():
                send_btn = gr.Button("Skicka", elem_classes="send-btn")
                clear_btn = gr.Button("Rensa", elem_classes="send-btn")
                debug_mode = gr.Checkbox(
                    label="Debug",
                    value=False,
                    visible=False
                )

            llm_model = gr.Dropdown(
                label="LLM-modell (experimentell)",
                choices=LLM_MODEL_OPTIONS,
                value=DEFAULT_LLM_MODEL,
                interactive=True,
                visible=False,
            )

            gr.Markdown("<h3>Svar</h3>")
            chatbot_answer = gr.Markdown(
                "",
                elem_classes="answer-box"
            )
            gr.Markdown(
                "_Obs: Detta svar är AI-genererat och bör vid behov verifieras mot källmaterialet._",
                elem_classes="answer-note"
            )
            
    # -------------------------
    # EVENTS
    # -------------------------

    card_outputs = [card_html for _, _, card_html in main_buttons]

    for btn, doc_id, _ in main_buttons:
        btn.click(
            fn=lambda d=doc_id: load_document(d),
            outputs=[questions, faq_answer, current_doc, *card_outputs]
        )

    questions.change(
        fn=select_and_submit,
        inputs=[questions, current_doc, debug_mode, llm_model],
        outputs=[faq_answer]
    )

    send_btn.click(
        fn=submit,
        inputs=[message, current_doc, debug_mode, llm_model],
        outputs=[chatbot_answer]
    )
    
    message.submit(
        fn=submit,
        inputs=[message, current_doc, debug_mode, llm_model],
        outputs=[chatbot_answer]
    )

    clear_btn.click(
        fn=clear_chatbot,
        outputs=[message, chatbot_answer]
    )


@demo.app.get("/health")
def health():
    return {"status": "ok", "revision": DEPLOY_REVISION}


@demo.app.get("/ready")
def ready():
    return {"status": "ok", "revision": DEPLOY_REVISION}

# =====================================================
# LAUNCH
# =====================================================

with open("style.css", encoding="utf-8") as f:
    css = f.read()

print("Deploy revision:", DEPLOY_REVISION)
demo.launch(theme=None,css=css, ssr_mode=False)
