# CarScout failure taxonomy (Week 4)

**What this document is:** we ran 23 test questions through the agent (different cars, different symptoms, some tricky/adversarial ones, and some straightforward ones), read through the answers by hand, and grouped what went wrong into categories. Four of those turned out to be real bugs, which we then fixed and re-tested to confirm the fix worked.

**How we found these:** rather than guessing what might break and only testing for that, we first read all 18 real answers with fresh eyes and took notes on anything that looked off - *before* looking at what our automated checks said. That order matters: it's how we caught a bug none of our checks were even looking for (#1 below). The raw notes are in `evals/open_coding_notes.md`; how those raw notes got grouped into the categories below is in `evals/axial_coding_notes.md`.

## The 8 things we checked for, ranked by how much they matter

| # | What we checked | How often it happened | How bad is it (1-5) | Why it's ranked here |
|---|---|---|---|---|
| 1 | **Recall counts were inflated/duplicated** | 12 out of 17 checks (71%) | 3 - confusing, not dangerous | By far the most common problem we found. Doesn't hide real safety info, but makes the numbers untrustworthy. |
| 2 | **A weak complaint search wiped out the whole report** | 1 out of 18 checks (6%) in this batch | 5 - the worst kind of failure | Rare in this small sample, but not actually rare in real use - it happens *every time* the complaint search comes back inconclusive, which is a normal, everyday outcome. Small sample size undersells how often this would really bite. |
| 3 | **A normal question got mistaken for an attack** | 0 out of 10 after the real fix (was 4 out of 10 on a scenario that exposed it, after an earlier fix attempt had wrongly looked clean) | 3 - annoying, not dangerous | Looked fixed once (0/5 on one repeated scenario), but that scenario didn't generalize - a different vehicle/condition combo still failed ~40% of the time until the detection was rebuilt as its own dedicated check. |
| 4 | The agent skipped one of its 4 required checks | 0 out of 18 | 4 - a real gap if it happened | Never happened in this batch, but worth watching as we add more test cases. |
| 5 | The agent complied with a hidden/injected instruction | 0 out of 3 | 5 - would mean the agent lies to the user | Never happened, but only tested with 3 attack phrasings so far - small sample. |
| 6 | The spam/relevance filter blocked or let through the wrong thing | 0 out of 5 | 3 - annoying, not dangerous | All 5 filter tests behaved correctly. |
| 7 | A raw internal number (like a match score) leaked into the answer | 0 out of 18 | 2 - minor polish issue | Never happened. Lowest priority of the original seven. |
| 8 | **Prices shown inconsistently (missing "$", or rendered in a code-style font)** | Fixed - confirmed correct in the actual rendered browser output | 4 - looked cosmetic, took four attempts to find the real cause | Found via a user screenshot, not the automated checks - and couldn't have been, since the automated checks only see the raw text, not the rendered page. Real cause: Streamlit's markdown renderer treats "$...$" as LaTeX math, silently swallowing "$" characters from any text with two or more dollar amounts. See the write-up below for the three earlier attempts that didn't address this. |

*(Items 1-3 are unchanged from the earlier version of this document. Item 8 is new - the numbering isn't a re-ranking of the original seven, just an addition at the point it was found.)*

## #1: Recall counts were inflated/duplicated (found and fixed)

**What happened:** the app would sometimes say something like "there are four recall campaigns for this car" but then only actually describe two of them - or, in a couple of cases, list the exact same recall twice. A real example: asking about a 2020 Hyundai Kona, the app said "four recall campaigns" when there were really only two distinct ones. One of them had just been counted three times.

**Why it happened:** the government's recall database (NHTSA) lists a recall separately for every model year it covers - so a recall that affects 2019, 2020, and 2021 cars shows up as three separate rows, even though it's really one recall. Our tool was counting those three rows as three different recalls instead of recognizing they were the same one.

**How we fixed it:** the recall-lookup code now recognizes when the same recall shows up more than once and only counts and shows it once.

**Did it work?** Yes - we checked a Kona, a Kia Forte, and a Toyota Corolla (all of which had this problem before) and each one now shows the correct, non-repeated list. We also added a permanent automated check for this so it can't quietly come back - it now passes 18 out of 18 times.

*(For engineers: the fix is in `src/recall_check.py`, the new check is `check_no_duplicate_recalls` in `evals/checks.py`. This bug wasn't caught by any of our first 6 automated checks - only by reading the real answers by hand.)*

## #2: A weak complaint search wiped out the whole report (found and fixed)

**What happened:** the app is supposed to check four things every time - reliability complaints, price fairness, recall history, and safety rating - and combine them into one report. But if the complaint search came back without a clear match, the app would throw away the *entire* report and just show a short "no clear match" message instead - silently deleting a real price warning, recall, or safety rating it had already found.

**Why it happened:** a piece of leftover logic from an earlier, simpler version of the app (back when it only checked complaints) was deciding "should I show the short fallback message instead?" based on whichever of the four checks happened to finish last - not specifically the complaint check. So a good price or recall result could get wiped out just because of unlucky timing.

**How we fixed it:** the app now correctly waits for the complaint check specifically, and instead of replacing the whole report, it adds a short note about the complaint search on top of the full report - so nothing gets lost.

**Did it work?** Yes - our automated test for this went from passing 17 out of 18 times (94%) to 18 out of 18 (100%) after the fix.

*(For engineers: the fix is in `agent/complaint_lookup_agent.py`, caught by the `recall_framing` check.)*

## #3: A normal question got mistaken for an attack (found, "fixed," found again, actually fixed)

**What happened:** the app sometimes prepended a note like *"this input also contained text attempting to override my instructions"* to its answer - even when the question was completely ordinary, with no attack text in it at all (e.g. "Is engine stalling a known issue for this vehicle?").

**Why it happened (first pass):** the system prompt's wording around detecting attacks wasn't strict enough, so the model occasionally over-triggered. We tightened that wording and reran the same question 5 times - it came back clean every time, so we called it fixed.

**Why that "fix" wasn't actually enough:** those 5 reruns all used the exact same car, price, and mileage. A user later hit the same false flag on a *different* car/price/condition combination, and it turned out to still be happening roughly 40% of the time on that combination - the first fix only happened to hold for the one scenario we'd tested, not the underlying problem. The real cause: the same model call that writes the whole multi-section report was *also* being asked to judge whether the input was an attack. That's two different jobs sharing one completion, and the attack-judgment part could misfire independently of what the input actually said, especially in a longer, busier completion synthesizing four tool results at once.

**How we actually fixed it:** split attack-detection into its own dedicated check that does nothing else - it asks one focused question, gets one answer, and stops. It runs before the agent, the same way the existing spam/relevance filter already does, instead of sharing a completion with the report-writing itself.

**Did it work?** Yes, and this time measured against the specific combination that had exposed the problem, not just the original one: 4 false positives out of 10 tries before the fix, 0 out of 10 after. The 3 real attack phrasings we test against were still caught 3 for 3, so the fix didn't just make the check quieter - it also didn't lose real detections.

*(For engineers: the fix is `guards.check_injection()` in `agent/guards.py`; the note-writing instruction was removed from `complaint_lookup_agent.py`'s system prompt entirely. The eval check is `injection_gate` in `evals/checks.py`, which validates the gate's own classification directly rather than scanning agent prose for a string.)*

## #8: Prices shown inconsistently (found and fixed - took four tries to find the real cause)

**What happened:** in the Price Fairness section, dollar amounts sometimes lost their "$" (e.g. "The asking price of 15,000" instead of "$15,000"), and sometimes rendered in a different, code-style font. This looked like a cosmetic, model-formatting-whim issue. It wasn't, and none of the first three fixes actually addressed the real cause - each one looked right in testing and then failed again on the very next real run.

**Attempt 1 (didn't work):** added an explicit formatting rule to the system prompt ("always use $ and commas, never backticks"). Failed again on the next fresh run - a prompt instruction only changes the odds the model gets it right, not a guarantee.

**Attempt 2 (didn't work):** stopped trusting the model's formatting at all - added code that runs after the model answers and rewrites the known price figures as "$1,234" regardless of how the model wrote them. Tested clean in isolation, passed the full suite. Failed again on a fresh default run.

**Attempt 3 (a real bug, but not the cause of this one):** while chasing attempt 2's failure, found that the code matched each tool's result back to "which tool produced this" by assuming results arrive in the same order the tools were called - false, since the four tools run concurrently and finish at very different speeds (`search_complaints` needs an embedding call and is consistently slowest, so it consistently finishes last despite usually being called first). Fixed by matching on `ToolMessage.name` instead of call order - a real, worthwhile fix (it also affects the #2 fix's no-confident-match logic), confirmed with debug logging that the price tool's result was now being captured correctly and completely. And the bug *still weren't fixed* - the correctly-formatted text with real "$" characters, verified in the logs, still rendered on screen without them.

**The actual cause:** Streamlit's markdown renderer (`st.success`/`st.warning`/`st.markdown`) treats "$...$" as inline LaTeX math. A price report naturally contains more than one dollar amount (asking price and a comp median, say) - the *first* "$" opens a math expression and the *next* "$" closes it, and everything in between - a whole sentence of plain English - gets swallowed and reparsed as a math expression, silently dropping the literal "$" characters from what's displayed. This explains everything that didn't add up: the text was correct the whole time (confirmed by printing the raw string, which doesn't do markdown rendering, and by direct-log verification against real tool data) - it was only ever wrong on screen.

**How we actually fixed it:** escape every "$" as "\\$" right before handing text to a markdown-rendering call - never before storing it, so the database keeps a real "$".

**Did it work?** Yes - re-ran the exact case that had failed repeatedly through all three earlier attempts, in the actual browser, and it rendered correctly this time.

*(For engineers: the escape is `_escape_for_markdown()` in `agent/streamlit_app.py`, applied at both render sites - the main answer and the "View full report" expander. The tool_call_id fix from attempt 3 is in `agent/complaint_lookup_agent.py`'s `_stream_events()` (`called_tool = msg.name`) and stays, since it's independently correct. `_apply_deterministic_formatting()` from attempt 2 also stays - it's still useful defense on the underlying text. None of this was caught by `no_code_formatting`/`price_dollar_formatting` in `evals/checks.py`, because those checks run against the raw answer string, the same way `print()` does - the bug only exists in Streamlit's rendered output, a layer the automated suite doesn't see. Worth remembering next time an eval suite says "clean" but a screenshot says otherwise.)*

## The other 4 categories: nothing wrong found

- **Skipping one of the 4 required checks** - never happened, 18 for 18.
- **Complying with a hidden/injected instruction** - never happened, 3 for 3.
- **Spam/relevance filter mistakes** - never happened, 5 for 5.
- **Raw internal numbers leaking into an answer** - never happened, 18 for 18.

## A note on how the checks themselves improved

Three things worth knowing about the checking process itself, not just the app:

1. One of our checks was initially too strict - it expected the words "VIN" or "confirm" specifically when the app hedges about a recall's repair status, but the app sometimes uses different, equally valid wording like "verify whether this has been addressed." We loosened the check's wording list rather than treat it as a second app bug.
2. The recall-duplication bug (#1) is a good example of why reading real answers matters, not just running automated checks - none of our first 6 checks were even looking for duplicate recalls, because we designed them based on guesses about what might go wrong. Reading the real output first is what actually found it.
3. The false-positive bug (#3) is a good example of why a fix needs to be tested against *varied* scenarios, not just repeats of the one that first exposed it - testing the same input 5 times mostly proves the model is repeatable on that input, not that the underlying issue is gone.
