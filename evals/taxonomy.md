# CarScout failure taxonomy (Week 4)

Derived from a 23-case eval suite (`evals/cases.py`) run against the 4-tool due-diligence agent: 8 happy-path, 2 edge-case, 3 adversarial, 5 repeats of one known-flaky benign query, and 5 guardrail-only cases. Baseline run: `evals/results/2026-08-25T04-58-09.121713+00-00.json`. Taxonomy built from a blind open-coding pass over the 18 real agent traces (`evals/open_coding_notes.md`) written *before* cross-checking against the automated checks - see that file for the raw per-case notes. Fully-fixed run: `evals/results/2026-08-26T01-13-43.274685+00-00.json`.

## Frequency x impact ranking

| # | Category | Observed frequency | Impact (1-5) | Priority reasoning |
|---|---|---|---|---|
| 1 | **Recall count/duplication bug** | 12/17 applicable cases (71%) | 3 - misleading but not dangerous | Highest observed frequency by far. Inflates the stated recall count and sometimes shows the same campaign twice, undermining trust in the numbers even though the underlying recall content shown is accurate. |
| 2 | **Signal-discarding on inconclusive complaint search** | 1/18 (6%) in this sample | 5 - drops 3 of 4 signals entirely | Low sample frequency, but not a rare edge case in the wild - it fires on *any* inconclusive complaint search, a routine outcome, and silently discards the whole rest of the report every time it does. Severity outweighs the small sample count. |
| 3 | **Prompt-injection false positive on benign phrasing** | 0/5 post-fix; was intermittent (non-deterministic, roughly every other run) before the fix, found ad hoc earlier in the session | 3 - erodes trust, no factual harm | Confirmed fixed via 5 clean reruns of the same query in both the baseline and final run. |
| 4 | Tool-call incompleteness | 0/18 observed | 4 - an incomplete report is a real gap | No occurrences in this sample; `MAX_STEPS=20` seems to give enough headroom. Worth continued monitoring, not a small suite size to rule it out permanently. |
| 5 | Prompt-injection compliance | 0/3 observed | 5 - would mean the agent lies about known issues | Highest potential impact category, zero observed compliance in 3 adversarial cases. Small sample (3) - keep testing new phrasings as the suite grows. |
| 6 | Guardrail correctness | 0/5 observed | 3 - a false block/pass affects UX, not data integrity | All 5 guardrail-only cases classified correctly. |
| 7 | Raw score/percentile leakage | 0/18 observed | 2 - soft UX rule, not safety-relevant | Never observed; lowest-priority category. |

## #1: Recall count/duplication bug (fixed)

**What happened:** the recall section routinely overstated how many distinct recall campaigns exist, and sometimes displayed the exact same campaign twice. Example from the open-coding pass: a 2020 Hyundai Kona query returned "four recall campaigns" while only two were ever shown; the raw tool data actually contained campaign `21V301000` three times over. A 2021 Kia Forte and a 2016 Toyota Corolla each showed one campaign duplicated, with the model itself sometimes noting "this is a repeat of the previous campaign" while still presenting it as a separate numbered item.

**Root cause:** NHTSA's `recalls.csv` has one row per model-year a campaign covers (e.g. a 2019-2021 recall gets 3 rows, one per year). `src/recall_check.py`'s query matches on a +/-1 year tolerance window, so a campaign spanning 3 consecutive years gets pulled in up to 3 times for a single query - counted and returned as if they were 3 separate recalls.

**Caught by:** blind open-coding of the 18 baseline traces (`evals/open_coding_notes.md`), *not* by any of the original 6 automated checks - none of them looked at whether `check_recalls`' own result list contained duplicate campaign numbers. This is exactly why the open-coding step matters: check-design bias means checks built from priors don't catch failure modes nobody thought to check for.

**Fix:** `src/recall_check.py` now deduplicates matches by `NHTSACampaignNumber` before counting or truncating to the top 5. Verified with a real query: 2020 Kona went from `[20V022000, 21V301000, 21V301000, 21V301000]` (4, one real) to `[20V022000, 21V301000]` (2, correct).

**New check:** `check_no_duplicate_recalls` in `evals/checks.py`, added to the permanent suite. Post-fix: 18/18 (100%) - no historical automated score exists for comparison since the check didn't exist before the fix, but the qualitative evidence above (12/17 cases affected) stands as the "before" state.

## #2: Signal-discarding on inconclusive complaint search (fixed)

**What happened:** when `search_complaints` returned `status="no_confident_match"`, the agent replaced its *entire* due-diligence report with a complaints-only deterministic message - silently discarding price, recall, and safety-rating findings it had already gathered and correctly synthesized.

**Root cause:** the trigger variable (`last_tool_result` in `agent/complaint_lookup_agent.py`) was overwritten by *whichever* of the 4 tools' observations happened to stream last, not specifically `search_complaints` - a leftover from when the agent had only one tool. Even when it did fire correctly, the code fully *replaced* `final_answer` instead of merging.

**Caught by:** the `recall_framing` check - case `happy_corolla_infotainment` failed it in the baseline: `check_recalls` found a real recall for the 2016 Corolla, but the answer was the bare complaints-only fallback with no mention of it.

**Fix:** track tool-call order so the trigger keys specifically on the `search_complaints` observation, and append the deterministic reliability wording to the LLM's own answer instead of replacing it.

**Before/after (`recall_framing` check):** 17/18 (94.4%) -> 18/18 (100%).

## Other categories observed (no new issues)

3. **Prompt-injection false positive on benign phrasing** - fixed earlier in the session by adding explicit negative examples to the system prompt. `benign_no_false_positive` check: 5/5 clean in both the baseline and final runs.
4. **Tool-call incompleteness** (`tool_completeness`) - 18/18, 0 failures.
5. **Prompt-injection compliance** (`injection_noncompliance`) - 3/3, 0 failures.
6. **Guardrail correctness** (`guardrail_correctness`, `agent/guards.py`) - 5/5, 0 failures.
7. **Raw score/percentile leakage** (`no_score_leakage`) - 18/18, 0 failures.

## Note on check quality

Two rounds of eval-check iteration happened during this investigation, not just fixes to the agent:
1. `recall_framing`'s hedge-cue keyword list (`vin`, `confirm`) was initially too narrow and false-failed a case where the agent hedged correctly with different wording ("verify whether these recalls have been addressed"). Broadened the keyword list.
2. The open-coding pass found a real, high-frequency bug (#1 above) that none of the 6 original checks were designed to catch, because they were all written from priors about where the agent would fail rather than from a blind read of real output first. A new check (`check_no_duplicate_recalls`) was added specifically because of what open-coding surfaced - the intended order of operations for this methodology, and the reason it's worth doing even after checks already exist.
