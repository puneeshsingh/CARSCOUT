# CarScout

Used-car due-diligence agent. The user picks a listing by VIN from a curated
6-vehicle demo shortlist (real VINs, real price/mileage/condition, sourced
from the same Craigslist dataset used for price comps - see [VIN-based
input](#vin-based-input) below) and describes a symptom or asks a question. A
LangGraph deep agent calls four MCP tools and produces one report covering:

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

## VIN-based input

Rather than typing in a vehicle, price, mileage, and condition by hand, the user picks a real listing by VIN from a small curated set (`VIN_DEMO_LISTINGS` in `src/config.py`) - one per shortlist vehicle, sourced from the same Craigslist dataset used for price comps (real VIN, real price/odometer/condition from that row). Selecting a VIN fills in the whole listing at once; only the symptom/question stays free text, since that's the buyer's own concern, not something a listing states.

Each curated VIN is also decoded live against NHTSA's free vPIC API (`src/vin_decode.py`, no key required) at selection time - a real API call and cross-check, not just a label, even though the curated table already knows the vehicle. This intentionally stays scoped to the 6-vehicle demo shortlist rather than accepting an arbitrary VIN or listing URL, since the underlying reliability/price/recall data only covers those 6 vehicles - a VIN for anything else would decode fine but return a report with nothing to say.

## Layout

- `data/` — junction to the local dataset folder (NHTSA complaints/recalls + Craigslist vehicles). Not tracked in git. (`investigations.csv` and `ratings.csv` also present but not yet used - `ratings.csv` only covers 2024+ model years, so live safety ratings come from NHTSA's API instead.)
- `data_cache/` — filtered/cached subsets of the above, scoped to the demo vehicle shortlist. Not tracked in git as a whole, but the two small shortlist parquet files are committed (see [Week 5](#week-5-memory--deployment)) since the app needs them at runtime and can't regenerate them without the local-only raw datasets.
- `src/config.py` — paths, model names, the shared vehicle shortlist, the curated `VIN_DEMO_LISTINGS`, and env loading.
- `src/ingest.py` — builds the Pinecone index from the NHTSA complaints dataset.
- `src/retrieve.py` — complaint query interface against the Pinecone index.
- `src/ingest_prices.py` / `src/price_check.py` — Craigslist price data caching and comp-based price fairness check.
- `src/ingest_recalls.py` / `src/recall_check.py` — NHTSA recall data caching and recall history lookup.
- `src/safety_rating.py` — live NHTSA crash-test rating lookup.
- `src/vin_decode.py` — live VIN decode via NHTSA's vPIC API (see [VIN-based input](#vin-based-input)).
- `src/memory_store.py` — durable "recent searches" memory store (SQLite/Postgres via SQLAlchemy), scoped by name.
- `mcp_server/server.py` — exposes all four checks as MCP tools.
- `agent/complaint_lookup_agent.py` — the deep agent and its due-diligence system prompt.
- `agent/streamlit_app.py` — demo UI (Streamlit).
- `agent/pages/1_Evaluations.py` — eval suite UI (Streamlit multipage nav).
- `evals/` — Week 4 eval suite: `cases.py` (test cases), `checks.py` (code-based assertions), `run_evals.py` (runner), `results/*.json` (saved runs), `taxonomy.md` (failure taxonomy).

## Week 4: evals

```
uv run python evals/run_evals.py
```

Runs 27 cases (happy-path, edge-case, adversarial/prompt-injection, and guardrail-only) through the agent and its guardrails, checks 9 code-based assertions per applicable case (tool-call completeness, no raw score leakage, recall framing safety, no duplicate recalls, no code-style formatting, price "$" formatting, injection non-compliance, injection-gate correctness, guardrail correctness), and saves a timestamped result to `evals/results/`. Compare any two runs in the Streamlit "Evaluations" page.

**Top failures found and fixed**: (1) NHTSA's recall data lists one row per model-year a campaign covers, so the recall lookup was counting the same campaign multiple times - affected 71% of applicable cases; fixed by de-duplicating on campaign number. (2) The agent used to silently discard its price/recall/safety findings whenever the complaint search came back inconclusive, because the code that generates deterministic reliability wording keyed off *whichever* tool's result streamed last rather than specifically `search_complaints` - fixed by tracking tool-call order; the `recall_framing` check went from 17/18 (94.4%) to 18/18 (100%). (3) Ordinary, non-adversarial questions were occasionally mis-flagged as prompt-injection attempts (~40% of the time on some vehicle/price combinations) because the same model call that wrote the due-diligence report was also judging whether the input was an attack - fixed by splitting attack-detection into its own dedicated check (`guards.check_injection`) that runs before the agent, mirroring the existing moderation/relevance gates; false positives went from 4/10 to 0/10 on the scenario that exposed it, with real attack phrasings still caught 3/3. (4) Dollar amounts sometimes lost their "$" sign or rendered in a code-style font - took four attempts to find the actual cause: Streamlit's markdown renderer treats "$...$" as LaTeX math, so any report mentioning two or more dollar amounts had its "$" characters silently swallowed on screen, even though the underlying text was always correct (confirmed by direct logging against real tool data). Fixed by escaping "$" right before rendering. Along the way, also found and fixed a real (separate) bug where tool results were matched back to their tool by assuming arrival order matched call order - false, since the four tools run concurrently and finish at different speeds; now matched via LangGraph's own `ToolMessage.name` instead. Neither bug was catchable by `no_code_formatting`/`price_dollar_formatting` in the eval suite, since those check the raw text, not the rendered page. See `evals/taxonomy.md` for the full four-attempt write-up. (5) When a reliability search found related complaints but none confidently matched, the app's explanation guessed at a reason ("simply an edge case not well-represented in the data") without checking whether that was true - found when a user questioned the wording on a case where the closest complaint on record was a near-exact match for the symptom asked about, just below the confidence threshold. Fixed by having the search tool report its single closest match even when unconfident, and quoting it directly instead of guessing why nothing matched confidently.

**Golden dataset (Path B stretch)**: each of the three fixes above is pinned as a permanent regression case in `evals/cases.py` (tagged `"golden"`, each with an `origin` field naming which finding it guards). `uv run python evals/run_evals.py --golden` runs just those 3 cases - a fast check that a known-fixed bug hasn't come back, meant to run right after touching the agent/guards/tools, before spending the time on the full 27-case suite.

## Week 5: memory & deployment

**Memory** — CarScout has no real login, but does have a lightweight named identity: a "Your name" field in the sidebar scopes **recent searches** to that name, so it's per-person rather than one shared list. Not real auth (anyone can type any name, there's no password) - a deliberate, disclosed trade-off for a demo/portfolio tool, not an oversight. Leaving the name blank falls back to the original shared/anonymous list (also how rows saved before this existed still show up).

- **What gets stored**: each completed due-diligence run's inputs (make, model, year, asking price, odometer, condition, symptom, and the name typed in, if any) plus a short preview (~200 chars) of the agent's final answer. Full tool traces and raw retrieval output are *not* stored - only the durable, human-meaningful facts.
- **When**: after a run finishes with a real answer. Runs that hit the step cap without a confident answer are not saved - the write gate only keeps stable, high-confidence facts.
- **Where**: `src/memory_store.py`, one `recent_searches` table via SQLAlchemy - SQLite locally (`data_cache/memory.db`) when `DATABASE_URL` is unset, Postgres in production when it is (Render sets this automatically for an attached Postgres instance). Same code path either way.
- **Retrieval**: the Streamlit sidebar reads the most recent rows for the current name on every page load and renders them.
- **Forgetting policy**: only the 20 most recent searches *per name* are kept; older rows for that name are deleted on write.

Cross-session recall is proven the way the syllabus's own bar describes it: write a search, kill the process, restart it, and the entry is still there without re-entering anything - verified locally (SQLite) and against the deployed Render instance (Postgres, see below).

The agent's own hard rules (never answer from training knowledge, injection handling, tool-calling requirements) live in `agent/complaint_lookup_agent.py`'s `SYSTEM_PROMPT` - a persistent, code-level location, not something stated only in chat. Memory is deliberately kept as an app-level (Streamlit) feature and isn't fed into the agent's own reasoning.

**Deployment** — Render, free tier:

- Build command: `pip install uv && uv sync`
- Start command: `uv run streamlit run agent/streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- Env vars: `OPENAI_API_KEY`, `PINECONE_API_KEY`, `DATABASE_URL` (from an attached Render Postgres instance)
- Live URL: https://carscout-rcbk.onrender.com/

**Fitting inside Render's free 512MB:** the first deploy OOM-crashed on every agent run. Two contributors: (1) Chroma's locally-loaded vector index and its dependency footprint (onnxruntime, tokenizers, etc.) — fixed by migrating to Pinecone, a remote/managed vector store with no local index and a much lighter client; (2) the MCP server was originally spawned as a second full Python subprocess, duplicating the whole interpreter and dependency tree — fixed by connecting to the same `FastMCP` server object over an in-memory session (`mcp.shared.memory.create_connected_server_and_client_session`) instead of a subprocess. This is still genuine MCP protocol traffic (real tool discovery, real request/response messages), just without a second OS process.
