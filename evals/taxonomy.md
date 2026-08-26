# CarScout failure taxonomy (Week 4)

**What this document is:** we ran 23 test questions through the agent (different cars, different symptoms, some tricky/adversarial ones, and some straightforward ones), read through the answers by hand, and grouped what went wrong into categories. Two of those turned out to be real bugs, which we then fixed and re-tested to confirm the fix worked.

**How we found these:** rather than guessing what might break and only testing for that, we first read all 18 real answers with fresh eyes and took notes on anything that looked off - *before* looking at what our automated checks said. That order matters: it's how we caught a bug none of our checks were even looking for (#1 below). The raw notes are in `evals/open_coding_notes.md` if you want to see the reasoning as it happened.

## The 7 things we checked for, ranked by how much they matter

| # | What we checked | How often it happened | How bad is it (1-5) | Why it's ranked here |
|---|---|---|---|---|
| 1 | **Recall counts were inflated/duplicated** | 12 out of 17 checks (71%) | 3 - confusing, not dangerous | By far the most common problem we found. Doesn't hide real safety info, but makes the numbers untrustworthy. |
| 2 | **A weak complaint search wiped out the whole report** | 1 out of 18 checks (6%) in this batch | 5 - the worst kind of failure | Rare in this small sample, but not actually rare in real use - it happens *every time* the complaint search comes back inconclusive, which is a normal, everyday outcome. Small sample size undersells how often this would really bite. |
| 3 | **A normal question got mistaken for an attack** | 0 out of 5 after the fix (was roughly 1 in 2 before, found earlier in the session) | 3 - annoying, not dangerous | Confirmed fixed - ran the same question 5 times after the fix and it never happened again. |
| 4 | The agent skipped one of its 4 required checks | 0 out of 18 | 4 - a real gap if it happened | Never happened in this batch, but worth watching as we add more test cases. |
| 5 | The agent complied with a hidden/injected instruction | 0 out of 3 | 5 - would mean the agent lies to the user | Never happened, but only tested with 3 attack phrasings so far - small sample. |
| 6 | The spam/relevance filter blocked or let through the wrong thing | 0 out of 5 | 3 - annoying, not dangerous | All 5 filter tests behaved correctly. |
| 7 | A raw internal number (like a match score) leaked into the answer | 0 out of 18 | 2 - minor polish issue | Never happened. Lowest priority of the seven. |

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

## The other 5 categories: nothing wrong found

- **Normal questions mistaken for attacks** - already fixed earlier in the session; confirmed clean across 5 repeat tests both before and after.
- **Skipping one of the 4 required checks** - never happened, 18 for 18.
- **Complying with a hidden/injected instruction** - never happened, 3 for 3.
- **Spam/relevance filter mistakes** - never happened, 5 for 5.
- **Raw internal numbers leaking into an answer** - never happened, 18 for 18.

## A note on how the checks themselves improved

Two things worth knowing about the checking process itself, not just the app:

1. One of our checks was initially too strict - it expected the words "VIN" or "confirm" specifically when the app hedges about a recall's repair status, but the app sometimes uses different, equally valid wording like "verify whether this has been addressed." We loosened the check's wording list rather than treat it as a second app bug.
2. The recall-duplication bug (#1) is a good example of why reading real answers matters, not just running automated checks - none of our first 6 checks were even looking for duplicate recalls, because we designed them based on guesses about what might go wrong. Reading the real output first is what actually found it.
