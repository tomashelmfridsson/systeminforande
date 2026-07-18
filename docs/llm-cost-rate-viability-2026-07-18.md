# LLM cost/rate/harness viability assessment

Date: 2026-07-18
Task: t_a17a952f
Canonical repo: `/Users/tomashelmfridsson/workspace/systeminforande`

## Scope

Assessed the current practical options already identified for this project:

- `openai/gpt-oss-120b` (hosted via Hugging Face routing)
- `zai-org/GLM-5.2` (hosted via Hugging Face routing)
- `Qwen/Qwen3-32B` (hosted candidate via Hugging Face routing)
- optional local `gemma3:4b` via Ollama

Target load for viability check: about 1000 requests/day.
That equals about `0.0116 requests/second` on average.

## Important limitation from this run

This worker had no outbound DNS/network resolution during the run.
Verified failures:

- `socket.gethostbyname('huggingface.co')` -> `gaierror(8, 'nodename nor servname provided, or not known')`
- `socket.gethostbyname('helmfridsson-systeminforande.hf.space')` -> same error
- `python3 tools/run_live_http_smoke.py` -> `FAIL <urlopen error [Errno 8] nodename nor servname provided, or not known>`
- `pip install ...` in a fresh venv also failed on DNS resolution

Because of that, this run could not retrieve current live Hugging Face pricing/rate-limit pages or execute fresh hosted smoke tests. Any hosted latency/quality evidence below is therefore limited to already-captured repo artifacts, and any missing pricing fields are left explicitly unknown rather than guessed.

## Grounded evidence available inside the repo

### Hosted model evidence already captured earlier

1. `tests/results/live_http_smoke_2026-07-11_openai-gpt-oss-120b.md`
   - `10 passed`, `0 failed`
   - observed scenario latencies roughly `0.73s` to `2.21s`
   - strongest project-side evidence for current stable hosted path

2. `tests/results/live_api_baseline_2026-07-10_gpt-oss-120b.md`
   - `6 passed`, `3 failed`
   - older baseline before later retrieval/extractive fixes

3. `tests/results/live_api_postdeploy_2026-07-10_glm-5-2_success.md`
   - `9 passed`, `4 deselected`
   - confirms a successful hosted GLM-5.2 deployment after the correct runtime came up

4. `tests/results/live_api_postdeploy_2026-07-10_current-hf-state.md`
   - shows that one deployed runtime still exposed an older model list
   - on that runtime, `zai-org/GLM-5.2` was rejected because it was not in the app's available choices
   - practical lesson: model availability through the HF-routed stack is operationally fragile and must be rechecked live

5. `docs/RAG_LAB_LESSONS.md`
   - records that several routed models were explored
   - preserves candidate list including `zai-org/GLM-5.2` and `Qwen/Qwen3-32B`

### Local model evidence measured in this run

Commands run locally:

- `ollama list`
  - confirms `gemma3:4b` is present locally
- `ollama run gemma3:4b "Besvara kort på svenska: Vad är acceptanstest i ett systeminförande?"`
  - exit `0`
  - elapsed `11.01s`
  - returned a short Swedish answer
- `ollama run qwen3:8b "Besvara kort på svenska: Vad är acceptanstest i ett systeminförande?"`
  - exit `0`
  - elapsed `73.69s`
  - output began with explicit `Thinking...`, which is undesirable for a grounded end-user RAG path

## Practical throughput proxy for ~1000 requests/day

Using the measured/report latencies only as rough sequential-throughput proxies:

- `openai/gpt-oss-120b` at `1.66s` -> about `52,048` requests/day theoretical sequential headroom
- local `gemma3:4b` at `11.01s` -> about `7,847` requests/day theoretical sequential headroom
- local `qwen3:8b` at `73.69s` -> about `1,172` requests/day theoretical sequential headroom

Interpretation:

- 1000 requests/day is a low sustained load
- even the slower local measurements can theoretically clear that volume if requests are spread through the day
- the real bottlenecks are not average throughput, but quality, hosted availability, cost surprises, and operational stability

## Comparison table

| Option | Grounded latency evidence | Cost evidence from this run | Rate/load viability for ~1000/day | Operational tradeoffs | Recommendation |
|---|---:|---|---|---|---|
| `openai/gpt-oss-120b` hosted | Strongest repo evidence: 2026-07-11 smoke report passed `10/10`, roughly `0.73s-2.21s` | Current live price not retrievable in this run because DNS/network was unavailable | Easily viable from observed latency alone | Best project-side stability evidence so far; already the preferred/default model in code; still depends on HF routing availability | Best safer production path right now |
| `zai-org/GLM-5.2` hosted | Repo evidence: successful 2026-07-10 live run with `9 passed`; no fresh live rerun possible today | Current live price not retrievable in this run because DNS/network was unavailable | Likely viable at 1000/day if availability holds, but fresh latency/rate confirmation is missing | Looks promising for quality, but repo history also shows runtime/model-list drift and availability mismatch risk | Best hosted challenger for later retest, not current default |
| `Qwen/Qwen3-32B` hosted | Candidate is documented in repo, but there is no project-side live result artifact here | Current live price not retrievable in this run because DNS/network was unavailable | Unknown until a real hosted smoke run exists | Interesting multilingual candidate, but currently under-evidenced for this project | Keep as future experiment, not default |
| local `gemma3:4b` via Ollama | Measured here: `11.01s` one-shot local response | Marginal/incremental token cost is effectively local compute only; no hosted billing claim needed | Operationally viable for 1000/day from throughput alone | Cheapest path for experiments; quality and grounding are not yet validated against this project's RAG scenarios; local hardware must stay up | Cheapest-safe experimentation path |

## gradio_client and live hosted regression harness

### What was broken before

Before any change, full test collection failed because `tests/test_live_gradio_api.py` imported `gradio_client` at module import time:

- `python3 -m pytest --collect-only -q`
- result before fix: `ModuleNotFoundError: No module named 'gradio_client'`

### Change made

I changed:

- `tests/test_live_gradio_api.py`

from a hard import:

- `from gradio_client import Client`

to an optional test dependency pattern:

- `Client = pytest.importorskip("gradio_client").Client`

This keeps the live hosted harness available when `gradio-client` is installed, but prevents the whole suite from failing collection on machines that intentionally do not have the live test dependency.

### Verification after the change

1. `python3 -m pytest --collect-only -q`
   - result: `35 tests collected in 0.48s`

2. `python3 -m pytest -q -m 'not live_api'`
   - result: `35 passed, 1 skipped in 0.76s`

3. Fresh-install attempt in a new venv:
   - created venv under the task workspace
   - attempted `pip install -r requirements.txt -r requirements-dev.txt`
   - install failed because DNS/network was unavailable, not because of package metadata

### Worth it?

Yes, but only as an opt-in dev harness.

Why yes:
- the existing live tests check the exact deployed Gradio API contract
- they cover regressions that local unit tests cannot catch: endpoint availability, real model selection behavior, latency budget, misspelling robustness, and grounded fallback behavior
- the repo already has useful historical evidence from these tests

Why not make it mandatory everywhere:
- it requires network access to the deployed Space
- it depends on optional package installation (`gradio-client`)
- it will be noisy/flaky in offline environments or during provider/runtime outages

Recommended posture:
- keep `gradio-client` in `requirements-dev.txt`
- keep live tests marked `@pytest.mark.live_api`
- keep them optional in local/dev collection
- run them explicitly in a connected environment or scheduled post-deploy smoke job

## Decision and recommendations

### Cheapest-safe experimentation path

Use local Ollama `gemma3:4b` for cheap exploratory experiments only.

Why:
- it is already installed locally
- it answered successfully in this run
- measured single-call latency (`11.01s`) is fast enough for 1000/day in aggregate
- it avoids hosted billing entirely during experimentation

Why it is not the production default:
- no grounded project-specific RAG evaluation was run against it here
- local uptime/GPU availability becomes your responsibility
- local-only experiments do not prove the HF-hosted user experience

### Safer production path

Keep `openai/gpt-oss-120b` as the safer production default for now.

Why:
- strongest project-specific live evidence currently on disk
- 2026-07-11 smoke report shows clean `10/10` pass with sub-3-second response times on tested scenarios
- the code currently treats it as the preferred model
- it has fewer availability surprises in the project history than GLM-5.2

### What to do next when network is available again

1. Re-run the hosted smoke suite explicitly for:
   - `openai/gpt-oss-120b`
   - `zai-org/GLM-5.2`
   - `Qwen/Qwen3-32B`
2. Capture fresh:
   - pass/fail counts
   - latency distribution
   - actual HF provider/routing availability
   - current price page / free-credit / rate-limit screenshots or saved markdown notes
3. If GLM-5.2 matches or beats `gpt-oss-120b` on quality without materially worse latency or availability, promote it to a feature-flagged hosted alternative rather than immediate default replacement.
4. Leave local `gemma3:4b` as a dev-only baseline until it has been evaluated on the same grounded RAG scenarios.

## Bottom line

- Cheapest-safe experiment path: local `gemma3:4b`
- Safer production path today: hosted `openai/gpt-oss-120b`
- Best hosted challenger to retest later: `zai-org/GLM-5.2`
- `Qwen/Qwen3-32B` remains interesting but under-evidenced
- Enabling the live hosted regression harness is worth it, but it should remain optional and explicitly invoked, not required for every offline/local test run
