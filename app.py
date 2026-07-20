import inspect
import json
import os
import re
import time
import uuid
import threading
from typing import Any
from pathlib import Path
from fastapi import HTTPException, Request as FastAPIRequest
import gradio as gr
from gradio.routes import App as GradioApp
import requests

from rag.grounding import filter_allowed_results, grounded_answer_or_fallback
from rag.model_selection import (
    build_model_choices,
    requested_llm_model_from_request,
    resolve_llm_model,
)
from rag.prompts import rag_prompt
from rag.source_links import (
    build_sources_md,
    format_source_url,
    serialize_homepage_links,
)
from rag.synthesis import (
    DEFAULT_SYNTHESIS_MODEL,
    build_final_grounded_answer,
    resolve_synthesis_settings,
)
from rag.search import explain_search, search
from llm.reasoning import generate_reasoning, generate_reasoning_from_prompt, generate_reasoning_from_prompt_with_usage
DATA_DIR = "rag/data"
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
HEADER_IMAGE_URL = (
    "https://raw.githubusercontent.com/"
    "tomashelmfridsson/systeminforande/main/brain.jpg"
)
DEPLOY_REVISION_FILE = "deploy_revision.txt"
DEPLOY_REVISION_SOURCE = DEPLOY_REVISION_FILE
DEPLOY_REVISION_FALLBACK = "local"
LOG_DIR = Path(os.getenv("SYSTEMINFORANDE_LOG_DIR", "/data/logs"))
ENABLE_USAGE_LOGGING = os.getenv("SYSTEMINFORANDE_ENABLE_LOGGING", "true").strip().lower() not in {"0", "false", "no"}
_LOG_WRITE_LOCK = threading.Lock()
EMBED_RESIZE_JS = """
() => {
    const messageType = "systeminforande:resize";

    const getHeight = () => Math.max(
        document.body.scrollHeight,
        document.documentElement.scrollHeight,
        document.body.offsetHeight,
        document.documentElement.offsetHeight
    );

    const sendHeight = () => {
        window.parent.postMessage(
            {
                type: messageType,
                height: getHeight()
            },
            "*"
        );
    };

    sendHeight();

    if (!window.__systeminforandeResizeObserver) {
        window.__systeminforandeResizeObserver = new ResizeObserver(sendHeight);
        window.__systeminforandeResizeObserver.observe(document.body);
    }

    if (!window.__systeminforandeResizeListener) {
        window.__systeminforandeResizeListener = true;
        window.addEventListener("load", sendHeight);
        window.addEventListener("resize", sendHeight);
        window.setInterval(sendHeight, 1000);
    }
}
"""


def load_deploy_revision() -> str:
    try:
        with open(DEPLOY_REVISION_FILE, encoding="utf-8") as revision_file:
            return revision_file.read().strip() or DEPLOY_REVISION_FALLBACK
    except OSError:
        return DEPLOY_REVISION_FALLBACK


def log_startup_version(version: str) -> None:
    print(
        f"[startup] app_version={version} version_source={DEPLOY_REVISION_SOURCE} fallback={DEPLOY_REVISION_FALLBACK}",
        flush=True,
    )


DEPLOY_REVISION = load_deploy_revision()
API_APP = GradioApp()


def _ensure_log_dir() -> bool:
    if not ENABLE_USAGE_LOGGING:
        return False

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as exc:
        print(f"Kunde inte skapa loggkatalog {LOG_DIR}: {exc}")
        return False


def _request_header(request: Any, header_name: str) -> str | None:
    headers = getattr(request, "headers", None)
    if headers is None:
        return None
    try:
        return headers.get(header_name)
    except Exception:
        return None


def _get_session_id(request: Any) -> str:
    session_hash = getattr(request, "session_hash", None)
    if session_hash:
        return str(session_hash)

    header_value = _request_header(request, "x-session-id")
    if header_value:
        return header_value.strip()

    return f"anon-{uuid.uuid4().hex}"


def _trim_text(value: str | None, max_length: int = 4000) -> str:
    text = (value or "").strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"


def _serialize_sources_for_log(sources: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out = []
    for source in sources or []:
        out.append(
            {
                "source": source.get("source"),
                "source_type": source.get("source_type"),
                "title": source.get("title"),
                "pages": source.get("pages"),
                "url": source.get("url"),
            }
        )
    return out


def _append_log_record(event_type: str, payload: dict[str, Any]) -> None:
    if not _ensure_log_dir():
        return

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    file_name = time.strftime("%Y-%m-%d", time.gmtime()) + ".jsonl"
    record = {
        "timestamp": timestamp,
        "event_type": event_type,
        "revision": DEPLOY_REVISION,
        **payload,
    }

    try:
        with _LOG_WRITE_LOCK:
            with (LOG_DIR / file_name).open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"Kunde inte skriva loggrad till {LOG_DIR / file_name}: {exc}")


def log_usage_event(
    event_type: str,
    *,
    request: Any = None,
    question: str | None = None,
    answer: str | None = None,
    doc_id: str | None = None,
    doc_title: str | None = None,
    route: str | None = None,
    llm_model: str | None = None,
    timing_ms: float | int | None = None,
    sources: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "session_id": _get_session_id(request),
        "question": _trim_text(question),
        "answer": _trim_text(answer),
        "doc_id": doc_id,
        "doc_title": doc_title,
        "route": route,
        "llm_model": llm_model,
        "latency_ms": timing_ms,
        "sources": _serialize_sources_for_log(sources),
        "metadata": metadata or {},
    }
    _append_log_record(event_type, payload)

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
        "id": "openai/gpt-oss-120b",
        "label": "OpenAI gpt-oss 120B",
        "description": "Standardmodell för experimentell källbunden syntes.",
    },
    {
        "id": "zai-org/GLM-5.2",
        "label": "GLM 5.2",
        "description": "Alternativ modell för experimentell källbunden syntes.",
    },
    {
        "id": "zai-org/GLM-4.5",
        "label": "GLM 4.5",
        "description": "Alternativ i samma modellfamilj som 5.2 men mer försiktig att prova.",
    },
    {
        "id": "deepseek-ai/DeepSeek-R1",
        "label": "DeepSeek R1",
        "description": "Resonemangsmodell, ofta långsammare.",
    },
    {
        "id": "Qwen/Qwen3-32B",
        "label": "Qwen3 32B",
        "description": "Stark flerspråkig modell med bra alternativ för svensk syntes.",
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
PREFERRED_LLM_MODEL = DEFAULT_SYNTHESIS_MODEL
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


def load_document(doc_id, request: gr.Request | None = None):
    doc = DOC_INDEX[doc_id]
    log_usage_event(
        "faq_doc_click",
        request=request,
        doc_id=doc_id,
        doc_title=doc.get("title"),
        metadata={"main_question": doc.get("main_question")},
    )
    questions = [q["question"] for q in doc["subquestions"]]
    main_answer = format_answer(doc.get("main_answer", {}))
    return (
        gr.update(choices=questions, value=None),
        main_answer,
        doc_id,
        *build_main_card_updates(doc_id),
    )


def select_and_submit(message: str, doc_id, debug_mode, llm_model, request: gr.Request | None = None):
    for answer in submit(message, doc_id, debug_mode, llm_model, request):
        yield answer


def answer_question(message, doc_id=None, debug_mode=False, llm_model=None, enable_synthesis: bool | None = None, request: Any = None) -> dict[str, Any]:
    started_at = time.perf_counter()
    raw_message = message or ""
    message = raw_message.strip()
    resolved_llm_model = resolve_llm_model(llm_model, request, default_model=DEFAULT_LLM_MODEL)
    resolved_enable_synthesis = bool(
        resolve_synthesis_settings(
            enable_synthesis=enable_synthesis,
            llm_model=resolved_llm_model,
            default_model=DEFAULT_LLM_MODEL,
        )["enabled"]
    )

    base_response = {
        "question": raw_message,
        "normalized_question": message,
        "doc_id": doc_id,
        "debug_mode": bool(debug_mode),
        "enable_synthesis": resolved_enable_synthesis,
        "llm_model": resolved_llm_model,
        "route": None,
        "answer_markdown": "",
        "sources": [],
        "homepage_links": [],
        "retrieval": None,
        "timing_ms": 0,
        "llm_usage": summarize_llm_usage([], resolved_llm_model),
    }

    if not message:
        base_response["timing_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        return base_response

    if doc_id and doc_id in DOC_INDEX:
        doc = DOC_INDEX[doc_id]

        for q in doc["subquestions"]:
            if q["question"] == message:
                fact_answer = format_answer(q["answer"])
                source_results = search(f"{doc['title']} {message}", top_k=5)

                if debug_mode:
                    fact_answer += build_predefined_debug_md(
                        question=message,
                        reasoning="",
                        source_results=source_results,
                        llm_answer=None,
                    )

                base_response.update(
                    {
                        "route": "predefined",
                        "answer_markdown": fact_answer,
                        "sources": serialize_results(source_results),
                        "retrieval": {
                            "query": f"{doc['title']} {message}",
                            "top_results": serialize_results(source_results),
                            "llm_status": "LLM-syntes används inte för fördefinierade FAQ-svar.",
                            "llm_synthesis_enabled": False,
                            "llm_synthesis_model": resolved_llm_model,
                            "llm_usage": summarize_llm_usage([], resolved_llm_model),
                        },
                    }
                )
                base_response["timing_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
                return base_response

    rag_response = build_rag_response(message, debug_mode, resolved_llm_model, resolved_enable_synthesis)
    base_response.update(rag_response)
    retrieval = rag_response.get("retrieval") or {}
    base_response["llm_usage"] = rag_response.get("llm_usage") or retrieval.get("llm_usage") or summarize_llm_usage([], resolved_llm_model)
    base_response["timing_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
    return base_response


def submit(message, doc_id, debug_mode, llm_model, request: gr.Request | None = None):
    """
    Central router:
    - Om message matchar en underfråga → vanlig Q&A
    - Annars → RAG över PDF-material
    """

    response = answer_question(message, doc_id, debug_mode, llm_model, request=request)
    doc = DOC_INDEX.get(doc_id) if doc_id else None
    event_type = "faq_question" if doc_id else "chat_question"
    log_usage_event(
        event_type,
        request=request,
        question=response["normalized_question"],
        answer=response["answer_markdown"],
        doc_id=doc_id,
        doc_title=doc.get("title") if doc else None,
        route=response.get("route"),
        llm_model=response.get("llm_model"),
        timing_ms=response.get("timing_ms"),
        sources=response.get("sources"),
        metadata={
            "debug_mode": bool(debug_mode),
            "llm_usage": response.get("llm_usage"),
        },
    )
    yield response["answer_markdown"]

def format_answer(answer):
    return "\n".join(_format_answer_sections(answer)).strip()


def _format_answer_sections(answer: dict, level: int = 0) -> list[str]:
    out = []
    for key, value in answer.items():
        if key == "Tabell" and isinstance(value, list) and _is_table_rows(value):
            out.extend(_format_markdown_table(value))
            out.append("")
            continue

        heading_prefix = "#" * min(level + 4, 6)
        if key == "Exempel":
            out.append("<p class='faq-example-label'><strong>Exempel</strong></p>")
        elif level == 0:
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


def _llm_usage_call_record(
    *,
    purpose: str,
    model: str | None,
    usage: dict[str, int | None] | None = None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    usage = usage or {}
    return {
        "purpose": purpose,
        "provider": "huggingface_hub.InferenceClient.chat_completion",
        "model": model,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "missing": not any(usage.get(key) is not None for key in ("prompt_tokens", "completion_tokens", "total_tokens")),
        "status": status,
        "error": error,
    }


def _sum_usage_field(records: list[dict[str, Any]], field: str) -> int | None:
    values = [record.get(field) for record in records]
    ints = [value for value in values if isinstance(value, int)]
    if not ints:
        return None
    if len(ints) != len(values):
        return None
    return sum(ints)


def summarize_llm_usage(records: list[dict[str, Any]] | None, model: str | None) -> dict[str, Any]:
    records = records or []
    return {
        "provider": "huggingface_hub.InferenceClient.chat_completion",
        "model": model,
        "prompt_tokens": _sum_usage_field(records, "prompt_tokens"),
        "completion_tokens": _sum_usage_field(records, "completion_tokens"),
        "total_tokens": _sum_usage_field(records, "total_tokens"),
        "calls": len(records),
        "missing": not records or any(record.get("missing") for record in records),
        "calls_detail": records,
    }


def safe_generate_reasoning_from_prompt_with_usage_records(
    prompt: str,
    model: str | None,
    *,
    purpose: str,
    usage_records: list[dict[str, Any]],
) -> str:
    try:
        result = generate_reasoning_from_prompt_with_usage(prompt, model=model)
        usage_records.append(
            _llm_usage_call_record(
                purpose=purpose,
                model=model,
                usage=result.usage,
                status="ok",
            )
        )
        return result.text
    except Exception as exc:
        print(f"LLM-fel i generate_reasoning_from_prompt: {exc}")
        usage_records.append(
            _llm_usage_call_record(
                purpose=purpose,
                model=model,
                status="error",
                error=str(exc),
            )
        )
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


def _strip_answer_metadata(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""

    for marker in ("\n\n---\n\n### Källor", "\n\n---\n\n### Debug"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    return text


def _looks_like_narrative_answer(answer: str) -> bool:
    body = _strip_answer_metadata(answer)
    if not body:
        return False

    # Ignore bare markdown scaffolding or link-heavy fragments without actual prose.
    body = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", body)
    body = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", body)
    body = re.sub(r"[#>*_`-]+", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) < 40:
        return False

    words = re.findall(r"[A-Za-zÅÄÖåäö0-9]+", body)
    return len(words) >= 8


def _select_reasoning_answer(llm_answer: str, structured_answer: str) -> str:
    llm_answer = (llm_answer or "").strip()
    structured_answer = (structured_answer or "").strip()

    if _looks_like_narrative_answer(llm_answer):
        return llm_answer

    if structured_answer:
        return structured_answer

    return llm_answer


def clear_all():
    return gr.update(choices=[], value=None), "", "", None, *build_main_card_updates(None)


def clear_chatbot():
    return "", ""


def reset_tab_state():
    return (
        gr.update(choices=[], value=None),
        "",
        None,
        "",
        "",
        *build_main_card_updates(None),
    )

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
    
def serialize_chunk(chunk: dict, score: float | None = None) -> dict[str, Any]:
    payload = {
        "source": chunk.get("source"),
        "source_type": chunk.get("source_type"),
        "title": chunk.get("title"),
        "pages": chunk.get("pages"),
        "url": format_source_url(chunk),
    }
    if score is not None:
        payload["score"] = round(score, 4)
    return payload


def serialize_results(results) -> list[dict[str, Any]]:
    return [serialize_chunk(chunk, score) for score, chunk in results]


def _simple_tokenize(text: str) -> set[str]:
    def _ascii_fold(value: str) -> str:
        return (
            value.replace("å", "a")
            .replace("ä", "a")
            .replace("ö", "o")
            .replace("Å", "A")
            .replace("Ä", "A")
            .replace("Ö", "O")
        )

    tokens = set()
    for raw_token in re.findall(r"\w+", _ascii_fold((text or "").lower())):
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
    original_query_terms = {
        term
        for term in (search_debug.get("query_terms") or [])
        if len(term) >= 4 and term not in {"maste", "onskat"}
    }
    expanded_query_terms = set(search_debug.get("expanded_query_terms") or original_query_terms)

    if not top_results or not (original_query_terms or expanded_query_terms):
        return False, "Inga tydliga träffar hittades i materialet."

    top_score = top_results[0]["score"]
    if top_score < 3:
        return False, "De högsta träffarna är för svaga för att betraktas som relevanta."

    combined_tokens = set()
    for item in top_results[:3]:
        chunk = item["chunk"]
        combined_tokens |= _simple_tokenize(chunk.get("title", ""))
        combined_tokens |= _simple_tokenize(chunk.get("text", ""))
        combined_tokens |= _simple_tokenize(chunk.get("source", ""))

    matched_original_terms = original_query_terms & combined_tokens
    if matched_original_terms:
        coverage = len(matched_original_terms) / max(len(original_query_terms), 1)
        if coverage >= 0.34 or len(matched_original_terms) >= 2 or top_score >= 10:
            return True, ""

    matched_expanded_terms = expanded_query_terms & combined_tokens
    if not matched_expanded_terms:
        return False, "Frågans nyckelord återfinns inte i de högst rankade träffarna."

    expanded_coverage = len(matched_expanded_terms) / max(len(expanded_query_terms), 1)
    if top_score >= 14 and matched_expanded_terms:
        return True, ""
    if len(expanded_query_terms) >= 3 and expanded_coverage < 0.2:
        return False, "För liten del av frågans nyckelord stöds av de högst rankade träffarna."

    return True, ""


def build_rag_response(query: str, debug: bool, llm_model: str | None, enable_synthesis: bool | None = None) -> dict[str, Any]:
    synthesis_settings = resolve_synthesis_settings(
        enable_synthesis=enable_synthesis,
        llm_model=llm_model,
        default_model=DEFAULT_LLM_MODEL,
    )
    resolved_llm_model = synthesis_settings["model"]
    synthesis_enabled = bool(synthesis_settings["enabled"])
    llm_usage_records: list[dict[str, Any]] = []

    results = filter_allowed_results(search(query, top_k=5))

    if not results:
        no_data = grounded_answer_or_fallback("")
        return {
            "route": "rag",
            "answer_markdown": no_data,
            "sources": [],
            "retrieval": {
                "query": query,
                "query_terms": [],
                "expanded_query_terms": [],
                "intent": "general",
                "top_results": [],
                "relevance_supported": False,
                "relevance_reason": "Inga godkända PDF-/hemsideutdrag hittades i materialet.",
                "confidence": 0,
                "llm_status": "LLM-syntes används inte i det användarvända RAG-svaret.",
                "llm_synthesis_enabled": False,
                "llm_synthesis_model": resolved_llm_model,
            },
        }

    scores = [score for score, _ in results]
    confidence = round(sum(scores) / len(scores), 2)

    search_debug = explain_search(query, top_k=5)
    allowed_result_ids = {
        chunk.get("id") or f"{chunk.get('source')}::{chunk.get('title')}::{chunk.get('section')}"
        for _, chunk in results
    }
    search_debug["top_results"] = [
        item for item in search_debug["top_results"]
        if (item["chunk"].get("id") or f"{item['chunk'].get('source')}::{item['chunk'].get('title')}::{item['chunk'].get('section')}") in allowed_result_ids
    ]

    is_relevant, relevance_reason = _has_relevant_rag_support(search_debug)
    if not is_relevant:
        no_data = grounded_answer_or_fallback("")
        if debug:
            no_data += (
                "\n\n---\n\n### Debug\n"
                f"**Confidence:** {confidence}\n"
                f"**Frågetyp:** {translate_intent(search_debug['intent'])}\n"
                f"**Query-termer:** `{', '.join(search_debug['query_terms'])}`\n"
                f"**Diagnos:** {relevance_reason}"
            )
        return {
            "route": "rag",
            "answer_markdown": no_data,
            "sources": serialize_results(results),
            "retrieval": {
                "query": query,
                "query_terms": search_debug["query_terms"],
                "expanded_query_terms": search_debug["expanded_query_terms"],
                "intent": search_debug["intent"],
                "top_results": serialize_results(results),
                "relevance_supported": False,
                "relevance_reason": relevance_reason,
                "confidence": confidence,
                "llm_status": "LLM-syntes används inte i det användarvända RAG-svaret.",
                "llm_synthesis_enabled": False,
                "llm_synthesis_model": resolved_llm_model,
            },
        }

    chunks = [chunk for _, chunk in results]
    synthesis_result = build_final_grounded_answer(
        query,
        chunks,
        enable_synthesis=synthesis_enabled,
        llm_model=resolved_llm_model,
        llm_rewrite=lambda prompt, model: safe_generate_reasoning_from_prompt_with_usage_records(
            prompt,
            model,
            purpose="synthesis",
            usage_records=llm_usage_records,
        ),
    )
    structured_answer = str(synthesis_result["extractive_answer"])
    final_answer = str(synthesis_result["final_answer"])
    sources_md = build_sources_md(results)
    homepage_links = serialize_homepage_links(results)

    llm_debug_md = ""
    llm_answer = ""
    if debug:
        llm_prompt = rag_prompt(query, chunks)
        llm_answer = safe_generate_reasoning_from_prompt_with_usage_records(
            llm_prompt,
            resolved_llm_model,
            purpose="debug_comparison",
            usage_records=llm_usage_records,
        )
        llm_debug_lines = [
            "\n\n---\n\n### Debug",
            f"**Confidence:** {confidence}",
            f"**Frågetyp:** {translate_intent(search_debug['intent'])}",
            f"**Query-termer:** `{', '.join(search_debug['query_terms'])}`",
            f"**Diagnos:** {diagnose_retrieval(structured_answer, search_debug)}",
            "**LLM-status:** LLM-syntes kördes endast för debug-jämförelse; användarsvaret bygger på extraherade PDF-/hemsideutdrag.",
            "",
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

        if llm_answer:
            llm_debug_lines.extend([
                "**LLM-jämförelse (ej använd i svaret):**",
                llm_answer,
            ])

        llm_debug_md = "\n".join(llm_debug_lines)

    final_llm_answer = (
        final_answer
        + sources_md
        + llm_debug_md
    )
    llm_usage = summarize_llm_usage(llm_usage_records, resolved_llm_model)
    return {
        "route": "rag",
        "llm_model": resolved_llm_model,
        "enable_synthesis": synthesis_enabled,
        "answer_markdown": final_llm_answer,
        "sources": serialize_results(results),
        "homepage_links": homepage_links,
        "llm_usage": llm_usage,
        "retrieval": {
            "query": query,
            "query_terms": search_debug["query_terms"],
            "expanded_query_terms": search_debug["expanded_query_terms"],
            "intent": search_debug["intent"],
            "top_results": serialize_results(results),
            "homepage_links": homepage_links,
            "relevance_supported": True,
            "relevance_reason": "",
            "confidence": confidence,
            "diagnosis": diagnose_retrieval(final_answer, search_debug),
            "llm_status": str(synthesis_result["llm_status"]),
            "llm_synthesis_enabled": synthesis_enabled,
            "llm_synthesis_used": bool(synthesis_result["synthesis_used"]),
            "llm_synthesis_model": resolved_llm_model,
            "llm_synthesis_enabled_source": synthesis_settings["enabled_source"],
            "llm_usage": llm_usage,
        },
    }


def handle_rag_query(query: str, debug: bool, llm_model: str):
    yield build_rag_response(query, debug, llm_model)["answer_markdown"]


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


def apply_request_model_selection(request: gr.Request | None = None):
    requested_model = requested_llm_model_from_request(request)
    if not requested_model:
        return gr.update()

    return gr.update(
        value=requested_model,
        choices=build_model_choices(LLM_MODEL_OPTIONS, requested_model),
    )
    
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
        with gr.Tab("FAQ") as faq_tab:
            gr.Markdown("<p class='tab-intro'>Välj ämnesområde och underfråga.</p>")

            with gr.Row(elem_classes="main-card-grid"):
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

        with gr.Tab("CHATTBOT") as chatbot_tab:
            message = gr.Textbox(
                placeholder="Skriv en fritextfråga här om du vill söka i källmaterialet.",
                lines=6,
                label=None,
                show_label=False,
                elem_classes="message-box"
            )

            with gr.Row():
                send_btn = gr.Button("Sök", elem_classes="send-btn")
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
                allow_custom_value=True,
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

    questions.select(
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

    faq_tab.select(
        fn=reset_tab_state,
        outputs=[questions, faq_answer, current_doc, message, chatbot_answer, *card_outputs],
    )

    chatbot_tab.select(
        fn=reset_tab_state,
        outputs=[questions, faq_answer, current_doc, message, chatbot_answer, *card_outputs],
    )

    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js=EMBED_RESIZE_JS,
    )
    demo.load(
        fn=apply_request_model_selection,
        inputs=None,
        outputs=[llm_model],
    )


@API_APP.get("/health")
def health():
    return {"status": "ok", "revision": DEPLOY_REVISION}


@API_APP.get("/ready")
def ready():
    return {"status": "ok", "revision": DEPLOY_REVISION}


@API_APP.post("/api/ask")
def api_ask(payload: dict[str, Any], request: FastAPIRequest):
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=400, detail="question måste vara en icke-tom sträng")

    doc_id = payload.get("doc_id")
    if doc_id is not None and not isinstance(doc_id, str):
        raise HTTPException(status_code=400, detail="doc_id måste vara en sträng om det anges")

    debug_mode = bool(payload.get("debug_mode", False))
    llm_model = payload.get("llm_model")
    if llm_model is not None and not isinstance(llm_model, str):
        raise HTTPException(status_code=400, detail="llm_model måste vara en sträng om det anges")

    enable_synthesis = payload.get("enable_synthesis")
    if enable_synthesis is not None and not isinstance(enable_synthesis, bool):
        raise HTTPException(status_code=400, detail="enable_synthesis måste vara en bool om det anges")

    response = answer_question(
        message=question,
        doc_id=doc_id,
        debug_mode=debug_mode,
        llm_model=llm_model,
        enable_synthesis=enable_synthesis,
        request=request,
    )
    doc = DOC_INDEX.get(doc_id) if doc_id else None
    log_usage_event(
        "api_question",
        request=request,
        question=response["normalized_question"],
        answer=response["answer_markdown"],
        doc_id=doc_id,
        doc_title=doc.get("title") if doc else None,
        route=response.get("route"),
        llm_model=response.get("llm_model"),
        timing_ms=response.get("timing_ms"),
        sources=response.get("sources"),
        metadata={
            "debug_mode": debug_mode,
            "enable_synthesis": enable_synthesis,
            "llm_usage": response.get("llm_usage"),
        },
    )
    return response

# =====================================================
# LAUNCH
# =====================================================

with open("style.css", encoding="utf-8") as f:
    css = f.read()

log_startup_version(DEPLOY_REVISION)


def launch_demo(**kwargs):
    launch_kwargs = {"theme": None, "css": css, "ssr_mode": False}
    launch_kwargs.update(kwargs)

    # Newer Gradio versions accept a private `_app` hook that lets us attach
    # custom FastAPI routes (/health, /ready, /api/ask) to the same ASGI app.
    # Some Hugging Face runtimes install a Gradio build where Blocks.launch()
    # does not expose that hook, so guard it at runtime instead of passing an
    # unsupported keyword and crashing during startup.
    launch_params = inspect.signature(demo.launch).parameters
    if "_app" in launch_params:
        launch_kwargs["_app"] = API_APP
        return demo.launch(**launch_kwargs)

    if not hasattr(gr, "mount_gradio_app"):
        # Last-resort compatibility path: the UI still launches, but custom
        # FastAPI routes will not be mounted on very old Gradio versions.
        return demo.launch(**launch_kwargs)

    server_name = launch_kwargs.pop(
        "server_name",
        os.getenv("GRADIO_SERVER_NAME") or "0.0.0.0",
    )
    server_port = int(
        launch_kwargs.pop(
            "server_port",
            os.getenv("GRADIO_SERVER_PORT") or os.getenv("PORT") or "7860",
        )
    )
    mount_params = inspect.signature(gr.mount_gradio_app).parameters
    mount_kwargs = {key: value for key, value in launch_kwargs.items() if key in mount_params}
    gr.mount_gradio_app(
        API_APP,
        demo,
        path="/",
        server_name=server_name,
        server_port=server_port,
        **mount_kwargs,
    )

    import uvicorn

    return uvicorn.run(API_APP, host=server_name, port=server_port)


launch_demo()
