import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Set, Tuple


BASE_URL = os.getenv("SYSTEMINFORANDE_BASE_URL", "https://helmfridsson-systeminforande.hf.space").rstrip("/")
DEFAULT_MODEL = os.getenv("SYSTEMINFORANDE_LLM_MODEL", "openai/gpt-oss-120b")
TIMEOUT_SECONDS = float(os.getenv("SYSTEMINFORANDE_API_TIMEOUT", "90"))
MAX_REGRESSION_SECONDS = float(os.getenv("SYSTEMINFORANDE_MAX_REGRESSION_RESPONSE_SECONDS", "20"))
ROOT_DIR = Path(__file__).resolve().parent.parent
SCENARIO_PATH = ROOT_DIR / "tests" / "data" / "live_api_scenarios.json"
RESULTS_DIR = ROOT_DIR / "tests" / "results"


def _load_scenarios():
    return json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))


def _http_json(url: str) -> dict:
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _submit_question(question: str, debug_mode: bool = False, llm_model: str = DEFAULT_MODEL) -> Tuple[str, float]:
    payload = json.dumps({"data": [question, None, debug_mode, llm_model]}).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}/gradio_api/call/submit",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    started_at = time.perf_counter()
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        event_id = json.loads(response.read().decode("utf-8"))["event_id"]

    with urllib.request.urlopen(
        f"{BASE_URL}/gradio_api/call/submit/{event_id}",
        timeout=TIMEOUT_SECONDS,
    ) as response:
        answer = _read_sse_answer(response)

    elapsed = time.perf_counter() - started_at
    return answer, elapsed


def _read_sse_answer(response) -> str:
    current_event = None

    for raw_line in response:
        line = raw_line.decode("utf-8", "replace").rstrip("\n")
        if not line:
            continue

        if line.startswith("event: "):
            current_event = line[len("event: ") :]
            continue

        if line.startswith("data: "):
            data = line[len("data: ") :]
            if current_event == "error":
                raise RuntimeError(f"Gradio submit returned event:error with data={data}")
            if current_event in {"generating", "complete"}:
                parsed = json.loads(data)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
                    return parsed[0]

    raise RuntimeError("No answer payload received from Gradio SSE stream")


def _extract_sources(answer_markdown: str) -> Set[str]:
    matches = re.findall(r"\[([^\]]+\.(?:pdf|PDF))\]\(", answer_markdown)
    return {match.strip() for match in matches}


def _extract_answer_body(answer_markdown: str) -> str:
    body = answer_markdown
    for marker in ("\n\n---\n\n### Källor", "\n\n---\n\n### Debug"):
        if marker in body:
            body = body.split(marker, 1)[0]
    return body.strip()


def _has_narrative_answer(answer_markdown: str) -> bool:
    body = _extract_answer_body(answer_markdown)
    if not body:
        return False

    body = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", body)
    body = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", body)
    body = re.sub(r"[#>*_`-]+", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) < 40:
        return False

    words = re.findall(r"[A-Za-zÅÄÖåäö0-9]+", body)
    return len(words) >= 8


def _assert(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def _slugify(value: str) -> str:
    slug = value.lower().replace("/", "-").replace("_", "-").replace(".", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "unknown"


def _truncate_one_line(value: str, limit: int = 220) -> str:
    text = " ".join((value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _write_results_report(
    started_epoch: float,
    model: str,
    checks: List[dict],
    summary: dict,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_date = time.strftime("%Y-%m-%d", time.localtime(started_epoch))
    filename = f"live_http_smoke_{run_date}_{_slugify(model)}.md"
    report_path = RESULTS_DIR / filename

    lines = [
        "# Live HTTP Smoke Result",
        "",
        f"- Datum: `{run_date}`",
        f"- Modell: `{model}`",
        f"- Bas-URL: `{BASE_URL}`",
        f"- Kommando: `python3 tools/run_live_http_smoke.py`",
        "",
        "## Sammanfattning",
        "",
        f"- `passed`: `{summary['passed']}`",
        f"- `failed`: `{summary['failed']}`",
        f"- `total`: `{summary['total']}`",
        "",
        "## Kontroller",
        "",
    ]

    for check in checks:
        label = check["label"]
        status = check["status"]
        elapsed = check.get("elapsed_s")
        question = check.get("question")
        answer_excerpt = check.get("answer_excerpt")
        detail = check.get("detail")

        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- Status: `{status}`")
        if elapsed is not None:
            lines.append(f"- Tid: `{elapsed:.2f}s`")
        if question:
            lines.append(f"- Fråga: `{question}`")
        if answer_excerpt:
            lines.append(f"- Svarsutdrag: `{answer_excerpt}`")
        if detail:
            lines.append(f"- Detalj: `{detail}`")
        lines.append("")

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def _run():
    started_epoch = time.time()
    checks = []
    summary = {"passed": 0, "failed": 0, "total": 0}
    report_path = None

    def record_pass(label: str, elapsed: float = None, question: str = None, answer: str = None):
        summary["passed"] += 1
        summary["total"] += 1
        checks.append(
            {
                "label": label,
                "status": "passed",
                "elapsed_s": elapsed,
                "question": question,
                "answer_excerpt": _truncate_one_line(answer) if answer else None,
            }
        )

    def record_fail(label: str, detail: str, elapsed: float = None, question: str = None, answer: str = None):
        summary["failed"] += 1
        summary["total"] += 1
        checks.append(
            {
                "label": label,
                "status": "failed",
                "elapsed_s": elapsed,
                "question": question,
                "answer_excerpt": _truncate_one_line(answer) if answer else None,
                "detail": detail,
            }
        )

    try:
        payload = _http_json(f"{BASE_URL}/gradio_api/info")
        _assert("/submit" in payload["named_endpoints"], "Missing /submit in gradio_api/info")
        record_pass("info endpoint")
        print("PASS info endpoint")

        for scenario in _load_scenarios():
            label = f"scenario {scenario['id']}"
            answer = ""
            elapsed = None
            try:
                answer, elapsed = _submit_question(scenario["question"])
                answer_lower = answer.lower()

                _assert(answer.strip(), f"{scenario['id']}: empty answer")
                _assert(
                    any(keyword.lower() in answer_lower for keyword in scenario["answer_contains_any"]),
                    f"{scenario['id']}: expected one of {scenario['answer_contains_any']!r}",
                )

                expected_sources = set(scenario["expected_source_any"])
                if expected_sources:
                    actual_sources = _extract_sources(answer)
                    _assert(
                        bool(actual_sources & expected_sources),
                        f"{scenario['id']}: expected one of {sorted(expected_sources)} but got {sorted(actual_sources)}",
                    )

                record_pass(label, elapsed=elapsed, question=scenario["question"], answer=answer)
                print(f"PASS scenario {scenario['id']} ({elapsed:.2f}s)")
            except (AssertionError, RuntimeError, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                record_fail(label, str(exc), elapsed=elapsed, question=scenario["question"], answer=answer)
                raise

        for question, expected_keywords in [
            ("Hur testar man ett nytt system", ["test", "acceptans", "verifier", "prestanda", "godkänn"]),
            ("Vilka etapper finns det", ["etapp", "planering", "acceptanstest", "pilotdrift", "driftsättning"]),
        ]:
            label = f"regression {question}"
            answer = ""
            elapsed = None
            try:
                answer, elapsed = _submit_question(question)
                answer_lower = answer.lower()

                _assert(_has_narrative_answer(answer), f"{question!r}: expected narrative answer before sources")
                _assert(
                    any(keyword in answer_lower for keyword in expected_keywords),
                    f"{question!r}: expected one of {expected_keywords!r}",
                )

                if question == "Hur testar man ett nytt system":
                    _assert(
                        elapsed <= MAX_REGRESSION_SECONDS,
                        f"{question!r}: exceeded latency budget {elapsed:.2f}s > {MAX_REGRESSION_SECONDS:.2f}s",
                    )

                record_pass(label, elapsed=elapsed, question=question, answer=answer)
                print(f"PASS regression {question} ({elapsed:.2f}s)")
            except (AssertionError, RuntimeError, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                record_fail(label, str(exc), elapsed=elapsed, question=question, answer=answer)
                raise

        print("PASS live smoke suite")
    finally:
        report_path = _write_results_report(started_epoch, DEFAULT_MODEL, checks, summary)
        print(f"Saved report: {report_path}")


if __name__ == "__main__":
    try:
        _run()
    except (AssertionError, RuntimeError, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        sys.exit(1)
