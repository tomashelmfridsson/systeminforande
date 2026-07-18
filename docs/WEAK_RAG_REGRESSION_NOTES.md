# Weak RAG regression notes

These tests were derived from the concrete failure-case inventory in Kanban task `t_aa494e57` and encode deterministic failures that can be reproduced locally from the indexed PDF chunks.

## Automated in `tests/test_weak_rag_regressions.py`

1. Wrong-topic contamination for `Vad ska en utbildningsstrategi innehålla`
2. Compound prompt coverage failure for verksamhet + konvertering
3. Compound prompt coverage failure for säkerhet + driftsättning + IT-miljöer + förvaltning
4. False fallback for `Finns det en utbildningsstrategi`
5. Setup question misrouted into utbildning/acceptanstest content
6. Systemsamband fragment-dump rendering
7. Svarstider/bearbetningstider fragment-dump rendering

## Implemented improvements for the first priority weak cases

1. Retrieval tuning for topic specificity
   - Added stronger topical-term scoring so generic matches like `krav`, `systemet` or `innehåll` do not outrank chunks that match the real subject of the question.
   - Added domain synonyms for system setup, driftsättning, systemsamband, svarstider/körningstider and förvaltningsöverlämnande.

2. Compound-question coverage
   - Multi-part questions are now split into clause-sized retrieval passes and merged so one subquestion cannot drown out the rest.
   - This specifically improved verksamhet + konvertering and säkerhet + driftsättning + IT-miljöer + förvaltning coverage.

3. Grounded extractive synthesis cleanup
   - Added focused answer builders for utbildningsstrategi, systemuppsättning, systemsamband and svarstider/bearbetningstider.
   - Added filtering to drop template placeholders, bracket prompts and raw question bullets from the final answer.
   - Added source focusing so named-document questions stay grounded in the matching document family instead of drifting into neighboring material.

## Not fully automated yet

1. Retry-without-improvement behavior
   - The prior log analysis showed repeated weak answers across multiple retries.
   - The local extractive path is deterministic and does not model the full historical retry loop, so this needs either log-based assertions or a higher-level live API harness that records repeated calls over time.

2. Timestamp-specific historical outputs from the 2026-07-17 production log
   - The regression tests assert the intended behavior against the current local retrieval/synthesis pipeline.
   - They do not assert byte-for-byte equality with the exact historical hosted responses because those depended on a deployed environment, runtime state, and formatting at that time.

3. End-to-end Gradio rendering/latency for these weak cases
   - The existing live API suite covers endpoint availability and a smaller regression set.
   - Expanding every weak case into live hosted checks is possible, but would be slower and more brittle than the current local deterministic regression layer.
