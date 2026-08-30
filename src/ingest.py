import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    NHTSA_COMPLAINTS_DIR,
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    VEHICLE_SHORTLIST,
)

COMPLAINTS_CSV = NHTSA_COMPLAINTS_DIR / "complaints.csv"
UPSERT_BATCH_SIZE = 100
# Metadata carries the narrative text itself (Pinecone doesn't store separate
# "documents" the way Chroma does) - each chunk is well under this per-vector
# metadata cap.
NARRATIVE_METADATA_KEY = "narrative"

NHTSA_COLUMNS = [
    "odiNumber",
    "make",
    "model",
    "modelYear",
    "summary",
    "components",
    "crash",
    "fire",
    "numberOfInjuries",
    "numberOfDeaths",
]

SHORTLIST = VEHICLE_SHORTLIST

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 125

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def load_and_filter_complaints(csv_path=COMPLAINTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=NHTSA_COLUMNS, low_memory=False)

    df["make"] = df["make"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    df["modelYear"] = pd.to_numeric(df["modelYear"], errors="coerce")

    make_lower = df["make"].str.lower()
    model_lower = df["model"].str.lower()

    mask = pd.Series(False, index=df.index)
    for make, model, year, tolerance in SHORTLIST:
        entry_mask = (make_lower == make) & (model_lower == model)
        if year is not None:
            entry_mask &= df["modelYear"].between(year - tolerance, year + tolerance)
        mask |= entry_mask

    filtered = df[mask].copy()
    filtered["summary"] = filtered["summary"].astype(str).str.strip()
    filtered = filtered[filtered["summary"].notna() & (filtered["summary"] != "") & (filtered["summary"] != "nan")]

    counts = filtered.groupby(["make", "model"]).size().sort_values(ascending=False)
    print("Complaint counts per make/model:")
    print(counts.to_string())

    return filtered


def _to_bool(value) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _to_int(value) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def chunk_complaints(df: pd.DataFrame) -> list[dict]:
    chunks = []
    for row in df.itertuples(index=False):
        complaint_id = str(row.odiNumber)
        narrative = row.summary

        texts = [narrative] if len(narrative) <= CHUNK_SIZE else _splitter.split_text(narrative)

        metadata_base = {
            "complaint_id": complaint_id,
            "make": row.make,
            "model": row.model,
            "model_year": _to_int(row.modelYear),
            "components": row.components,
            "crash": _to_bool(row.crash),
            "fire": _to_bool(row.fire),
            "number_of_injuries": _to_int(row.numberOfInjuries),
            "number_of_deaths": _to_int(row.numberOfDeaths),
        }

        for i, text in enumerate(texts):
            doc_id = complaint_id if len(texts) == 1 else f"{complaint_id}_{i}"
            chunks.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": {
                        **metadata_base,
                        "chunk_index": i,
                        "chunk_count": len(texts),
                        NARRATIVE_METADATA_KEY: text,
                    },
                }
            )

    return chunks


def get_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(PINECONE_INDEX_NAME)


def _clean_metadata(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if v is not None}


def embed_and_upsert(chunks: list[dict], index) -> None:
    # Pinecone upserts are idempotent by id, so re-running ingestion simply
    # overwrites existing vectors rather than needing an existence check -
    # unlike Chroma, there's no cheap way to list all stored ids up front.
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[start : start + UPSERT_BATCH_SIZE]
        embeddings = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[c["text"] for c in batch],
        )
        vectors = [
            {
                "id": c["id"],
                "values": embedding.embedding,
                "metadata": _clean_metadata(c["metadata"]),
            }
            for c, embedding in zip(batch, embeddings.data)
        ]
        index.upsert(vectors=vectors)
        print(f"  Embedded and upserted batch {start // UPSERT_BATCH_SIZE + 1}: {len(batch)} chunks")

    print(f"\nUpserted: {len(chunks)} chunks")


def main():
    filtered = load_and_filter_complaints()
    chunks = chunk_complaints(filtered)
    print(f"\nTotal chunks to consider: {len(chunks)}")
    print(f"Example metadata: {chunks[0]['metadata']}")

    index = get_index()
    embed_and_upsert(chunks, index)

    stats = index.describe_index_stats()
    print(f"\nTotal vectors in '{PINECONE_INDEX_NAME}' index: {stats['total_vector_count']}")


if __name__ == "__main__":
    main()
