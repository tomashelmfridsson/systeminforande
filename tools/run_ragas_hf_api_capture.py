from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

from run_ragas_hf_capture import (
    DEFAULT_BASE_URL,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TIMEOUT_SECONDS,
    RemoteGradioClient,
    build_summary,
    ensure_remote_base_url,
    load_json,
    populate_case_capture,
    validate_output,
    write_outputs,
)


def _agent_status(response: dict[str, Any]) -> dict[str, Any]:
    retrieval = response.get("retrieval") or {}
    pipeline = retrieval.get("agentic_pipeline") or {}
    return {
        "review_status": pipeline.get("review_status"),
        "fallback_reason": pipeline.get("fallback_reason"),
        "fallback_retrieval_used": retrieval.get("agentic_fallback_retrieval_used"),
        "accepted_rewrites": pipeline.get("accepted_rewrites"),
        "rejected_rewrites": pipeline.get("rejected_rewrites"),
    }


def run_api_capture(
    *,
    dataset: dict[str, Any],
    base_url: str,
    llm_model: str,
    enable_agentic_rag: bool,
    timeout_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ensure_remote_base_url(base_url)
    client = RemoteGradioClient(base_url=base_url, timeout_seconds=timeout_seconds)
    captured_dataset = copy.deepcopy(dataset)
    captured_cases = []
    telemetry = []

    for index, case in enumerate(captured_dataset["cases"], start=1):
        question_id = case["question_id"]
        print(f"[{index:02d}/{len(captured_dataset['cases']):02d}] {question_id}", flush=True)
        response = client._json_request(
            f"{base_url.rstrip('/')}/api/ask",
            method="POST",
            payload={
                "question": case["user_input"],
                "debug_mode": True,
                "enable_agentic_rag": enable_agentic_rag,
                "llm_model": llm_model,
            },
        )
        captured_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        captured_cases.append(
            populate_case_capture(
                case,
                answer_markdown=response["answer_markdown"],
                base_url=base_url,
                llm_model=llm_model,
                captured_at=captured_at,
            )
        )
        telemetry.append(
            {
                "question_id": question_id,
                "timing_ms": response.get("timing_ms"),
                "llm_usage": response.get("llm_usage"),
                "enable_agentic_rag": response.get("enable_agentic_rag"),
                "agentic": _agent_status(response),
            }
        )

    captured_dataset["cases"] = captured_cases
    return captured_dataset, telemetry


def write_telemetry(
    telemetry: list[dict[str, Any]],
    *,
    results_dir: Path,
    run_timestamp: str,
    enable_agentic_rag: bool = True,
) -> tuple[Path, Path]:
    timestamp_slug = run_timestamp.replace(":", "").replace("-", "")
    mode_slug = str(enable_agentic_rag).lower()
    json_path = results_dir / f"ragas_hf_agentic_{mode_slug}_telemetry_{timestamp_slug}.json"
    md_path = results_dir / f"ragas_hf_agentic_{mode_slug}_telemetry_{timestamp_slug}.md"

    complete = [
        item
        for item in telemetry
        if not (item.get("llm_usage") or {}).get("missing", True)
    ]
    total_tokens = [
        item["llm_usage"]["total_tokens"]
        for item in complete
        if item["llm_usage"].get("total_tokens") is not None
    ]
    timings = [
        item["timing_ms"]
        for item in telemetry
        if item.get("timing_ms") is not None
    ]
    calls = [
        call
        for item in telemetry
        for call in (item.get("llm_usage") or {}).get("calls_detail", [])
    ]
    known_call_tokens = [
        call["total_tokens"]
        for call in calls
        if call.get("total_tokens") is not None
    ]
    fallback_count = sum(
        bool(item["agentic"].get("fallback_reason"))
        or bool(item["agentic"].get("fallback_retrieval_used"))
        for item in telemetry
    )
    approved_without_fallback = sum(
        item["agentic"].get("review_status") == "approved"
        and not item["agentic"].get("fallback_reason")
        and not item["agentic"].get("fallback_retrieval_used")
        for item in telemetry
    )
    summary = {
        "question_count": len(telemetry),
        "complete_token_rows": len(complete),
        "total_tokens": sum(total_tokens),
        "known_call_tokens_including_partial_rows": sum(known_call_tokens),
        "mean_tokens": round(sum(total_tokens) / len(total_tokens), 2) if total_tokens else None,
        "min_tokens": min(total_tokens) if total_tokens else None,
        "max_tokens": max(total_tokens) if total_tokens else None,
        "mean_timing_ms": round(sum(timings) / len(timings), 2) if timings else None,
        "fallback_count": fallback_count,
        "approved_without_fallback": approved_without_fallback,
        "successful_llm_calls": sum(call.get("status") == "ok" for call in calls),
        "failed_llm_calls": sum(call.get("status") == "error" for call in calls),
    }
    payload = {
        "artifact_type": "ragas_hf_agentic_api_telemetry",
        "captured_at": run_timestamp,
        "enable_agentic_rag": enable_agentic_rag,
        "summary": summary,
        "cases": telemetry,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# RAG API telemetry",
        "",
        f"- Captured at: `{run_timestamp}`",
        f"- `enable_agentic_rag={mode_slug}`",
        f"- Questions: `{summary['question_count']}`",
        f"- Complete token rows: `{summary['complete_token_rows']}`",
        f"- Total tokens: `{summary['total_tokens']}`",
        f"- Known tokens including partial rows: `{summary['known_call_tokens_including_partial_rows']}`",
        f"- Mean tokens/question: `{summary['mean_tokens']}`",
        f"- Min/max tokens: `{summary['min_tokens']}` / `{summary['max_tokens']}`",
        f"- Mean response time: `{summary['mean_timing_ms']} ms`",
        f"- Fallbacks: `{summary['fallback_count']}`",
        f"- Approved without fallback: `{summary['approved_without_fallback']}`",
        f"- Successful/failed LLM calls: `{summary['successful_llm_calls']}` / `{summary['failed_llm_calls']}`",
        "",
        "| Question | Tokens | Calls | Time ms | Review | Fallback |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for item in telemetry:
        usage = item.get("llm_usage") or {}
        agentic = item["agentic"]
        fallback = agentic.get("fallback_reason") or (
            "fallback_retrieval" if agentic.get("fallback_retrieval_used") else ""
        )
        lines.append(
            f"| {item['question_id']} | {usage.get('total_tokens')} | "
            f"{usage.get('calls')} | {item.get('timing_ms')} | "
            f"{agentic.get('review_status')} | {fallback or ''} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--agentic-mode", choices=("true", "false"), default="true")
    args = parser.parse_args()
    enable_agentic_rag = args.agentic_mode == "true"

    captured, telemetry = run_api_capture(
        dataset=load_json(args.dataset),
        base_url=args.base_url,
        llm_model=args.model,
        enable_agentic_rag=enable_agentic_rag,
        timeout_seconds=args.timeout_seconds,
    )
    run_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dated_results_dir = args.results_dir / time.strftime("%Y-%m-%d", time.gmtime())
    dataset_path, summary_path = write_outputs(
        captured,
        results_dir=dated_results_dir,
        llm_model=args.model,
        run_timestamp=run_timestamp,
        base_url=args.base_url,
    )
    validate_output(dataset_path)
    summary = build_summary(captured, base_url=args.base_url, llm_model=args.model)
    if summary["responses_captured"] != summary["question_count"]:
        raise RuntimeError("Not all questions produced response_text")
    if summary["retrieved_context_captures"] != summary["question_count"]:
        raise RuntimeError("Not all questions produced retrieved contexts")
    telemetry_json, telemetry_md = write_telemetry(
        telemetry,
        results_dir=dated_results_dir,
        run_timestamp=run_timestamp,
        enable_agentic_rag=enable_agentic_rag,
    )
    print(
        json.dumps(
            {
                "capture": str(dataset_path),
                "summary": str(summary_path),
                "telemetry_json": str(telemetry_json),
                "telemetry_markdown": str(telemetry_md),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
