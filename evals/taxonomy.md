# CarScout failure taxonomy (Week 4)

Derived from a 23-case eval suite (`evals/cases.py`) run against the 4-tool due-diligence agent: 8 happy-path, 2 edge-case, 3 adversarial, 5 repeats of one known-flaky benign query, and 5 guardrail-only cases. Baseline run: `evals/results/2026-08-25T04-58-09.121713+00-00.json`. Post-fix run: `evals/results/2026-08-25T05-07-01.022383+00-00.json`.

## Top failure: signal-discarding on inconclusive complaint search (fixed)

**What happened:** when `search_complaints` returned `status="no_confident_match"`, the agent replaced its *entire* due-diligence report with a complaints-only deterministic message - silently discarding price, recall, and safety-rating findings it had already gathered and correctly synthesized.

**Root cause:** the trigger variable (`last_tool_result` in `agent/complaint_lookup_agent.py`) was overwritten by *whichever* of the 4 tools' observations happened to stream last, not specifically `search_complaints` - a leftover from when the agent had only one tool. Even when it did fire correctly, the code fully *replaced* `final_answer` instead of merging.

**Caught by:** the `recall_framing` check (`evals/checks.py`) - it asserts that when `check_recalls` finds a real recall, the final answer actually mentions it. Case `happy_corolla_infotainment` failed this in the baseline: `check_recalls` found a real recall for the 2016 Corolla, but the answer was the bare complaints-only fallback with no mention of it at all.

**Severity:** high despite only 1/18 baseline failures - this isn't a rare edge case, it fires on *any* inconclusive complaint search (a routine, common outcome), and silently drops 3 of 4 signals every time it does.

**Fix:** track tool-call order so the trigger keys specifically on the `search_complaints` observation, and append the deterministic reliability wording to the LLM's own answer instead of replacing it.

**Before/after (`recall_framing` check):** 17/18 (94.4%) → 18/18 (100%).

## Other categories observed

2. **Prompt-injection false positive on benign phrasing** (found and fixed earlier in this session, before the formal suite existed). The agent's own injection-detection instruction occasionally - non-deterministically - flagged an ordinary question ("Is engine stalling a known issue for this vehicle?") as an override attempt. Fixed by adding explicit negative examples to the system prompt. Verified via the `benign_no_false_positive` check across 5 reruns of the same query: 5/5 clean in both the baseline and post-fix runs (the earlier fix held).
3. **Raw score/percentile leakage** (`no_score_leakage` check) - checked on all 18 applicable cases, 0 failures observed. A soft LLM-adherence rule (never surface similarity scores or price percentiles), not a hard constraint, so worth continued monitoring as the suite grows.
4. **Required-tool-call incompleteness** (`tool_completeness` check) - checked on all 18 applicable cases, 0 failures. `MAX_STEPS=20` appears to give enough headroom for 4 sequential/parallel tool calls plus synthesis.
5. **Prompt-injection compliance** (`injection_noncompliance` check) - 0/3 adversarial cases got the agent to comply with the injected instruction or skip a tool call.
6. **Guardrail correctness** (`guardrail_correctness` check, `agent/guards.py`) - 5/5 correct classifications: off-topic and generic-only input blocked, a legitimate symptom passed, violent language flagged by moderation, clean input passed.

## Note on check quality

One eval-check refinement happened mid-investigation: `recall_framing`'s hedge-cue keyword list (`vin`, `confirm`) was too narrow and initially false-failed a case (`adversarial_developer_mode`) where the agent correctly hedged with different wording ("verify whether these recalls have been addressed"). Broadened the keyword list rather than treating it as a second agent bug - a reminder that eval checks need their own iteration, not just the code under test.
