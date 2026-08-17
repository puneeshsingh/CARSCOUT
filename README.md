# CarScout

Used-car due-diligence agent. This repo currently covers the RAG ingestion pipeline; the agent itself is a separate phase.

## Setup

```
uv sync
cp .env.example .env   # fill in OPENAI_API_KEY
```

## Layout

- `data/` — junction to the local dataset folder (NHTSA complaints + Craigslist vehicles). Not tracked in git.
- `src/config.py` — paths, model names, and env loading.
- `src/ingest.py` — builds the Chroma vector store from the datasets.
- `src/retrieve.py` — query interface against the Chroma store.
- `chroma_db/` — local persistent Chroma store. Not tracked in git.
