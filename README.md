# CarScout

Used-car due-diligence agent. Given a vehicle (make/model/year, from a 6-vehicle
demo shortlist) and a listing's asking price, mileage, and a described symptom,
a LangGraph deep agent calls four MCP tools and produces one report covering:

- **Reliability** — known-issue search over real NHTSA complaint narratives (RAG/Pinecone)
- **Price fairness** — asking price vs. comparable real Craigslist listings for that mileage/year
- **Recall history** — official NHTSA recall campaigns for that make/model/year (never claims a specific listing's repair is outstanding - that needs the actual VIN)
- **Safety rating** — live NHTSA crash-test star rating

## Setup

```
uv sync
cp .env.example .env   # fill in OPENAI_API_KEY, PINECONE_API_KEY
```

Before running the agent, build the local data caches (one-time, or whenever the source datasets change):

```
uv run python src/ingest.py          # NHTSA complaints -> Pinecone index (created automatically if missing)
uv run python src/ingest_prices.py   # Craigslist listings -> data_cache/vehicles_shortlist.parquet
uv run python src/ingest_recalls.py  # NHTSA recalls -> data_cache/recalls_shortlist.parquet
```

Safety ratings are looked up live against NHTSA's public API at query time - no ingestion step needed.

## Layout

- `data/` — junction to the local dataset folder (NHTSA complaints/recalls + Craigslist vehicles). Not tracked in git. (`investigations.csv` and `ratings.csv` also present but not yet used - `ratings.csv` only covers 2024+ model years, so live safety ratings come from NHTSA's API instead.)
- `data_cache/` — filtered/cached subsets of the above, scoped to the demo vehicle shortlist. Not tracked in git as a whole, but the two small shortlist parquet files are committed (see [Week 5](#week-5-memory--deployment)) since the app needs them at runtime and can't regenerate them without the local-only raw datasets.
- `src/config.py` — paths, model names, the shared vehicle shortlist, and env loading.
- `src/ingest.py` — builds the Pinecone index from the NHTSA complaints dataset.
- `src/retrieve.py` — complaint query interface against the Pinecone index.
- `src/ingest_prices.py` / `src/price_check.py` — Craigslist price data caching and comp-based price fairness check.
- `src/ingest_recalls.py` / `src/recall_check.py` — NHTSA recall data caching and recall history lookup.
- `src/safety_rating.py` — live NHTSA crash-test rating lookup.
- `src/memory_store.py` — durable "recent searches" memory store (SQLite/Postgres via SQLAlchemy).
- `mcp_server/server.py` — exposes all four checks as MCP tools.
- `agent/complaint_lookup_agent.py` — the deep agent and its due-diligence system prompt.
- `agent/streamlit_app.py` — demo UI (Streamlit).
- `agent/pages/1_Evaluations.py` — eval suite UI (Streamlit multipage nav).
- `evals/` — Week 4 eval suite: `cases.py` (test cases), `checks.py` (code-based assertions), `run_evals.py` (runner), `results/*.json` (saved runs), `taxonomy.md` (failure taxonomy).

## Week 4: evals

```
uv run python evals/run_evals.py
```

Runs 23 cases (happy-path, edge-case, adversarial/prompt-injection, and guardrail-only) through the agent and its guardrails, checks 7 code-based assertions per applicable case (tool-call completeness, no raw score leakage, recall framing safety, no duplicate recalls, injection non-compliance, benign-input false-positive rate, guardrail correctness), and saves a timestamped result to `evals/results/`. Compare any two runs in the Streamlit "Evaluations" page.

**Top failures found and fixed**: (1) NHTSA's recall data lists one row per model-year a campaign covers, so the recall lookup was counting the same campaign multiple times - affected 71% of applicable cases; fixed by de-duplicating on campaign number. (2) The agent used to silently discard its price/recall/safety findings whenever the complaint search came back inconclusive, because the code that generates deterministic reliability wording keyed off *whichever* tool's result streamed last rather than specifically `search_complaints` - fixed by tracking tool-call order; the `recall_framing` check went from 17/18 (94.4%) to 18/18 (100%). (3) Ordinary, non-adversarial questions were occasionally mis-flagged as prompt-injection attempts (~40% of the time on some vehicle/price combinations) because the same model call that wrote the due-diligence report was also judging whether the input was an attack - fixed by splitting attack-detection into its own dedicated check (`guards.check_injection`) that runs before the agent, mirroring the existing moderation/relevance gates; false positives went from 4/10 to 0/10 on the scenario that exposed it, with real attack phrasings still caught 3/3. (4) Dollar amounts sometimes lost their "$" sign or rendered in a code-style font - took four attempts to find the actual cause: Streamlit's markdown renderer treats "$...$" as LaTeX math, so any report mentioning two or more dollar amounts had its "$" characters silently swallowed on screen, even though the underlying text was always correct (confirmed by direct logging against real tool data). Fixed by escaping "$" right before rendering. Along the way, also found and fixed a real (separate) bug where tool results were matched back to their tool by assuming arrival order matched call order - false, since the four tools run concurrently and finish at different speeds; now matched via LangGraph's own `ToolMessage.name` instead. Neither bug was catchable by `no_code_formatting`/`price_dollar_formatting` in the eval suite, since those check the raw text, not the rendered page. See `evals/taxonomy.md` for the full four-attempt write-up.

**Golden dataset (Path B stretch)**: each of the three fixes above is pinned as a permanent regression case in `evals/cases.py` (tagged `"golden"`, each with an `origin` field naming which finding it guards). `uv run python evals/run_evals.py --golden` runs just those 3 cases - a fast check that a known-fixed bug hasn't come back, meant to run right after touching the agent/guards/tools, before spending the time on the full 24-case suite.

## Week 5: memory & deployment

**Memory** — CarScout has no login, so "memory" here is a single shared **recent searches** list rather than per-user memory; that's a deliberate scope choice for a single-user demo tool, not an oversight.

- **What gets stored**: each completed due-diligence run's inputs (make, model, year, asking price, odometer, condition, symptom) plus a short preview (~200 chars) of the agent's final answer. Full tool traces and raw retrieval output are *not* stored - only the durable, human-meaningful facts.
- **When**: after a run finishes with a real answer. Runs that hit the step cap without a confident answer are not saved - the write gate only keeps stable, high-confidence facts.
- **Where**: `src/memory_store.py`, one `recent_searches` table via SQLAlchemy - SQLite locally (`data_cache/memory.db`) when `DATABASE_URL` is unset, Postgres in production when it is (Render sets this automatically for an attached Postgres instance). Same code path either way.
- **Retrieval**: the Streamlit sidebar reads the most recent rows on every page load and renders them - no session/auth scoping, since there's no login.
- **Forgetting policy**: only the 20 most recent searches are kept; older rows are deleted on write.

Cross-session recall is proven the way the syllabus's own bar describes it: write a search, kill the process, restart it, and the entry is still there without re-entering anything - verified locally (SQLite) and against the deployed Render instance (Postgres, see below).

The agent's own hard rules (never answer from training knowledge, injection handling, tool-calling requirements) live in `agent/complaint_lookup_agent.py`'s `SYSTEM_PROMPT` - a persistent, code-level location, not something stated only in chat. Memory is deliberately kept as an app-level (Streamlit) feature and isn't fed into the agent's own reasoning.

**Deployment** — Render, free tier:

- Build command: `pip install uv && uv sync`
- Start command: `uv run streamlit run agent/streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- Env vars: `OPENAI_API_KEY`, `PINECONE_API_KEY`, `DATABASE_URL` (from an attached Render Postgres instance)
- Live URL: _TODO once deployed_
