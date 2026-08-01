from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "https://helmfridsson-systeminforande.hf.space"
DEFAULT_DATASET_PATH = Path("tests/data/ragas_hf_evaluation_dataset_30_questions.json")
DEFAULT_RESULTS_DIR = Path("tests/results")
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_REQUEST_RETRIES = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
FORBIDDEN_BASE_URL_PREFIXES = ("http://localhost", "http://127.0.0.1")
METRIC_NAMES = [
    "faithfulness",
    "answer relevance",
    "context precision",
    "context recall",
]


def _rstrip_slash(value: str) -> str:
    return value.rstrip("/")


def ensure_remote_base_url(base_url: str) -> None:
    normalized = _rstrip_slash(base_url)
    if any(normalized.startswith(prefix) for prefix in FORBIDDEN_BASE_URL_PREFIXES):
        raise ValueError(
            f"Refusing to run official HF capture against local endpoint: {base_url}"
        )


def build_headers() -> dict[str, str]:
    import os

    headers = {"Content-Type": "application/json"}
    auth_token = None
    for env_name in ("SYSTEMINFORANDE_API_TOKEN", "HF_TOKEN"):
        value = os.getenv(env_name)
        if value:
            auth_token = value.strip()
            break

    if auth_token:
        header_name = os.getenv(
            "SYSTEMINFORANDE_AUTH_HEADER_NAME", "Authorization"
        ).strip()
        scheme = os.getenv("SYSTEMINFORANDE_AUTH_SCHEME", "Bearer").strip()
        headers[header_name] = f"{scheme} {auth_token}".strip()
    return headers


def _is_transient_network_error(exc: BaseException) -> bool:
    if isinstance(exc, socket.gaierror):
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, socket.gaierror | TimeoutError):
            return True
        if isinstance(reason, OSError):
            message = str(reason).lower()
            if "temporary failure" in message or "timed out" in message:
                return True
        if isinstance(reason, str):
            lowered = reason.lower()
            if "temporary failure" in lowered or "timed out" in lowered:
                return True
    if isinstance(exc, OSError):
        message = str(exc).lower()
        if "temporary failure" in message or "timed out" in message:
            return True
    return False


def _with_network_retries(operation: str, func, *, retries: int, backoff_seconds: float):
    attempt = 1
    while True:
        try:
            return func()
        except Exception as exc:
            if attempt >= retries or not _is_transient_network_error(exc):
                raise
            sleep_seconds = backoff_seconds * (2 ** (attempt - 1))
            print(
                f"retrying {operation} after transient network error "
                f"(attempt {attempt}/{retries}, sleep={sleep_seconds:.1f}s): {exc}",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)
            attempt += 1


class RemoteGradioClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        *,
        request_retries: int = DEFAULT_REQUEST_RETRIES,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    ):
        self.base_url = _rstrip_slash(base_url)
        self.timeout_seconds = timeout_seconds
        self.headers = build_headers()
        self.request_retries = request_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def probe(self) -> dict[str, Any]:
        return self._json_request(f"{self.base_url}/gradio_api/info")

    def submit_question(self, question: str, llm_model: str, debug_mode: bool = True) -> str:
        payload = {"data": [question, None, debug_mode, llm_model]}
        submit_response = self._json_request(
            f"{self.base_url}/gradio_api/call/submit",
            method="POST",
            payload=payload,
        )
        event_id = submit_response["event_id"]
        return self._read_sse_answer(
            f"{self.base_url}/gradio_api/call/submit/{event_id}"
        )

    def _json_request(
        self,
        url: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        def make_request() -> dict[str, Any]:
            data = None
            if payload is not None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=data,
                headers=self.headers,
                method=method,
            )
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))

        return _with_network_retries(
            f"{method} {url}",
            make_request,
            retries=self.request_retries,
            backoff_seconds=self.retry_backoff_seconds,
        )

    def _read_sse_answer(self, url: str) -> str:
        def read_stream() -> str:
            request = urllib.request.Request(url, headers=self.headers, method="GET")
            current_event = None
            latest_answer: str | None = None
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", "replace").rstrip("\n")
                    if not line:
                        continue
                    if line.startswith("event: "):
                        current_event = line[len("event: ") :]
                        continue
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if current_event == "error":
                        raise RuntimeError(f"Gradio submit returned event:error with data={data}")
                    if current_event in {"generating", "complete"}:
                        parsed = json.loads(data)
                        if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
                            latest_answer = parsed[0]
                            if current_event == "complete":
                                return latest_answer
            if latest_answer is not None:
                return latest_answer
            raise RuntimeError("No answer payload received from Gradio SSE stream")

        return _with_network_retries(
            f"GET {url}",
            read_stream,
            retries=self.request_retries,
            backoff_seconds=self.retry_backoff_seconds,
        )


def split_answer_sections(answer_markdown: str) -> tuple[str, str, str]:
    text = (answer_markdown or "").strip()
    if not text:
        return "", "", ""

    answer_body = text
    sources_section = ""
    debug_section = ""

    sources_marker = "\n\n---\n\n### Källor"
    debug_marker = "\n\n---\n\n### Debug"

    if sources_marker in answer_body:
        answer_body, remainder = answer_body.split(sources_marker, 1)
        sources_section = "### Källor" + remainder
    elif debug_marker in answer_body:
        answer_body, remainder = answer_body.split(debug_marker, 1)
        debug_section = "### Debug" + remainder
        return answer_body.strip(), sources_section.strip(), debug_section.strip()

    if sources_section and debug_marker in sources_section:
        sources_section, debug_remainder = sources_section.split(debug_marker, 1)
        debug_section = "### Debug" + debug_remainder

    return answer_body.strip(), sources_section.strip(), debug_section.strip()


def extract_source_links(answer_markdown: str) -> list[str]:
    links = re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", answer_markdown or "")
    return _dedupe_preserve_order(links)


def parse_debug_contexts(debug_section: str) -> list[dict[str, Any]]:
    if not debug_section:
        return []

    lines = debug_section.splitlines()
    contexts: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    text_lines: list[str] = []
    in_llm_comparison = False

    def flush_current() -> None:
        nonlocal current, text_lines
        if current is None:
            return
        context_text = "\n".join(line.rstrip() for line in text_lines).strip()
        if context_text:
            current["text"] = context_text
        else:
            current.setdefault("text", "")
        contexts.append(current)
        current = None
        text_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("**LLM-jämförelse"):
            in_llm_comparison = True
            flush_current()
            continue
        if in_llm_comparison:
            continue
        if stripped.startswith("**📄 Källa:** "):
            flush_current()
            current = {"source": stripped[len("**📄 Källa:** ") :].strip()}
            text_lines = []
            continue
        if current is None:
            continue
        if stripped == "---":
            flush_current()
            continue
        if stripped.startswith("- **Typ:** "):
            current["source_type"] = stripped[len("- **Typ:** ") :].strip()
            continue
        if stripped.startswith("- **Rubrik:** "):
            current["title"] = stripped[len("- **Rubrik:** ") :].strip()
            continue
        if stripped.startswith("- **Sidor:** "):
            current["pages"] = _parse_pages_value(stripped[len("- **Sidor:** ") :].strip())
            continue
        if stripped.startswith("- **Score:** "):
            score_text = stripped[len("- **Score:** ") :].strip().strip("`")
            try:
                current["score"] = float(score_text)
            except ValueError:
                current["score"] = score_text
            continue
        if stripped.startswith("- **"):
            continue
        text_lines.append(line)

    flush_current()
    return contexts


def _parse_pages_value(value: str) -> list[int] | None:
    if value in {"None", "null", ""}:
        return None
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, list) and all(isinstance(item, int) for item in parsed):
        return parsed
    return None


def parse_answer_markdown(answer_markdown: str) -> dict[str, Any]:
    response_text, _sources_section, debug_section = split_answer_sections(answer_markdown)
    source_links = extract_source_links(answer_markdown)
    context_blocks = parse_debug_contexts(debug_section)
    retrieved_contexts = [block["text"] for block in context_blocks if block.get("text")]
    retrieved_context_sources = [
        block["source"]
        for block in context_blocks
        if block.get("source")
    ]
    return {
        "response_text": response_text,
        "source_links": source_links,
        "retrieved_contexts": retrieved_contexts,
        "retrieved_context_sources": _dedupe_preserve_order(retrieved_context_sources),
        "debug_context_blocks": context_blocks,
    }


def populate_case_capture(
    case: dict[str, Any],
    *,
    answer_markdown: str,
    base_url: str,
    llm_model: str,
    captured_at: str,
) -> dict[str, Any]:
    updated = copy.deepcopy(case)
    parsed = parse_answer_markdown(answer_markdown)
    contexts = parsed["retrieved_contexts"]
    response_text = parsed["response_text"] or None
    metric_status = "captured" if response_text and contexts else "missing_retrieved_contexts"

    updated["runtime_record"]["hf_base_url"] = DEFAULT_BASE_URL
    updated["runtime_record"]["debug_mode"] = True
    updated["runtime_record"]["response_text"] = response_text
    updated["runtime_record"]["response_markdown"] = answer_markdown
    updated["runtime_record"]["retrieved_contexts"] = contexts
    updated["runtime_record"]["retrieved_context_sources"] = parsed["retrieved_context_sources"]
    updated["runtime_record"]["source_links"] = parsed["source_links"]
    updated["runtime_record"]["captured_at"] = captured_at
    updated["runtime_record"]["llm_model"] = llm_model

    updated["ragas_record"]["response"] = response_text
    updated["ragas_record"]["retrieved_contexts"] = contexts
    updated["ragas_record"]["metric_status"] = {
        metric_name: metric_status for metric_name in METRIC_NAMES
    }
    return updated


def run_capture(
    *,
    dataset: dict[str, Any],
    base_url: str,
    llm_model: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    enforce_remote: bool = True,
) -> dict[str, Any]:
    if enforce_remote:
        ensure_remote_base_url(base_url)

    client = RemoteGradioClient(base_url=base_url, timeout_seconds=timeout_seconds)
    info_payload = client.probe()
    named_endpoints = info_payload.get("named_endpoints", {})
    if "/submit" not in named_endpoints:
        raise RuntimeError("/submit was not present in gradio_api/info")

    captured_dataset = copy.deepcopy(dataset)
    captured_cases = []
    for case in captured_dataset["cases"]:
        answer_markdown = client.submit_question(
            question=case["user_input"],
            llm_model=llm_model,
            debug_mode=True,
        )
        captured_cases.append(
            populate_case_capture(
                case,
                answer_markdown=answer_markdown,
                base_url=base_url,
                llm_model=llm_model,
                captured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
        )

    captured_dataset["cases"] = captured_cases
    return captured_dataset


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slugify(value: str) -> str:
    slug = value.lower().replace("/", "-").replace("_", "-").replace(".", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "unknown"


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def build_summary(captured_dataset: dict[str, Any], *, base_url: str, llm_model: str) -> dict[str, Any]:
    cases = captured_dataset["cases"]
    with_contexts = sum(1 for case in cases if case["runtime_record"]["retrieved_contexts"])
    with_responses = sum(1 for case in cases if case["runtime_record"]["response_text"])
    return {
        "base_url": base_url,
        "llm_model": llm_model,
        "question_count": len(cases),
        "responses_captured": with_responses,
        "retrieved_context_captures": with_contexts,
        "metrics": METRIC_NAMES,
    }


def write_outputs(
    captured_dataset: dict[str, Any],
    *,
    results_dir: Path,
    llm_model: str,
    run_timestamp: str,
    base_url: str,
) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp_slug = run_timestamp.replace(":", "").replace("-", "")
    model_slug = _slugify(llm_model)
    dataset_path = results_dir / f"ragas_hf_capture_{timestamp_slug}_{model_slug}.json"
    summary_path = results_dir / f"ragas_hf_capture_{timestamp_slug}_{model_slug}.md"

    dataset_path.write_text(
        json.dumps(captured_dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = build_summary(captured_dataset, base_url=base_url, llm_model=llm_model)
    lines = [
        "# RAGAS HF capture run",
        "",
        f"- Timestamp: `{run_timestamp}`",
        f"- Base URL: `{base_url}`",
        f"- Model: `{llm_model}`",
        f"- Questions: `{summary['question_count']}`",
        f"- Responses captured: `{summary['responses_captured']}`",
        f"- Retrieved-context captures: `{summary['retrieved_context_captures']}`",
        f"- Output JSON: `{dataset_path}`",
        "",
        "## Metrics prepared",
        "",
    ]
    for metric_name in METRIC_NAMES:
        lines.append(f"- `{metric_name}`")
    lines.extend(["", "## Per-question capture status", ""])
    for case in captured_dataset["cases"]:
        response_status = "yes" if case["runtime_record"]["response_text"] else "no"
        context_status = "yes" if case["runtime_record"]["retrieved_contexts"] else "no"
        lines.append(
            f"- `{case['question_id']}` response={response_status} contexts={context_status} question={case['user_input']}"
        )
    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return dataset_path, summary_path


def validate_output(dataset_path: Path) -> None:
    validator_path = Path(__file__).with_name("validate_ragas_hf_evaluation_dataset.py")
    spec = importlib.util.spec_from_file_location(
        "validate_ragas_hf_evaluation_dataset", validator_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load validator from {validator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate(dataset_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the live HF Gradio /submit capture for the 30-question RAGAS dataset "
            "and emit a populated raw-input artifact."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the 30-question RAGAS HF evaluation dataset JSON.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Live Hugging Face base URL. Must not be localhost for the official run.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model id to submit as the fourth Gradio input.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where the captured dataset and markdown summary are written.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout for each Gradio request.",
    )
    parser.add_argument(
        "--allow-non-remote-base-url",
        action="store_true",
        help="Disable the localhost guard. Intended only for tests and local mocking.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = load_json(args.dataset)
    captured = run_capture(
        dataset=dataset,
        base_url=args.base_url,
        llm_model=args.model,
        timeout_seconds=args.timeout_seconds,
        enforce_remote=not args.allow_non_remote_base_url,
    )
    run_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dataset_path, summary_path = write_outputs(
        captured,
        results_dir=args.results_dir,
        llm_model=args.model,
        run_timestamp=run_timestamp,
        base_url=args.base_url,
    )
    validate_output(dataset_path)

    summary = build_summary(captured, base_url=args.base_url, llm_model=args.model)
    if summary["responses_captured"] != summary["question_count"]:
        raise RuntimeError(
            "Not all questions produced response_text; inspect the output artifact for details."
        )
    if summary["retrieved_context_captures"] != summary["question_count"]:
        raise RuntimeError(
            "Not all questions produced retrieved contexts; inspect the output artifact for details."
        )

    print(f"ok: wrote captured dataset to {dataset_path}")
    print(f"ok: wrote run summary to {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, urllib.error.URLError) as exc:
        print(f"capture failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
