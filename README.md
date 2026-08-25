# CarScout

Used-car due-diligence agent. Given a vehicle (make/model/year, from a 6-vehicle
demo shortlist) and a listing's asking price, mileage, and a described symptom,
a LangGraph deep agent calls four MCP tools and produces one report covering:

- **Reliability** — known-issue search over real NHTSA complaint narratives (RAG/Chroma)
- **Price fairness** — asking price vs. comparable real Craigslist listings for that mileage/year
- **Recall history** — official NHTSA recall campaigns for that make/model/year (never claims a specific listing's repair is outstanding - that needs the actual VIN)
- **Safety rating** — live NHTSA crash-test star rating

## Setup

```
uv sync
cp .env.example .env   # fill in OPENAI_API_KEY
```

Before running the agent, build the local data caches (one-time, or whenever the source datasets change):

```
uv run python src/ingest.py          # NHTSA complaints -> Chroma vector store
uv run python src/ingest_prices.py   # Craigslist listings -> data_cache/vehicles_shortlist.parquet
uv run python src/ingest_recalls.py  # NHTSA recalls -> data_cache/recalls_shortlist.parquet
```

Safety ratings are looked up live against NHTSA's public API at query time - no ingestion step needed.

## Layout

- `data/` — junction to the local dataset folder (NHTSA complaints/recalls + Craigslist vehicles). Not tracked in git. (`investigations.csv` and `ratings.csv` also present but not yet used - `ratings.csv` only covers 2024+ model years, so live safety ratings come from NHTSA's API instead.)
- `data_cache/` — filtered/cached subsets of the above, scoped to the demo vehicle shortlist. Not tracked in git.
- `src/config.py` — paths, model names, the shared vehicle shortlist, and env loading.
- `src/ingest.py` — builds the Chroma vector store from the NHTSA complaints dataset.
- `src/retrieve.py` — complaint query interface against the Chroma store.
- `src/ingest_prices.py` / `src/price_check.py` — Craigslist price data caching and comp-based price fairness check.
- `src/ingest_recalls.py` / `src/recall_check.py` — NHTSA recall data caching and recall history lookup.
- `src/safety_rating.py` — live NHTSA crash-test rating lookup.
- `mcp_server/server.py` — exposes all four checks as MCP tools.
- `agent/complaint_lookup_agent.py` — the deep agent and its due-diligence system prompt.
- `agent/streamlit_app.py` — demo UI (Streamlit).
- `chroma_db/` — local persistent Chroma store. Not tracked in git.
