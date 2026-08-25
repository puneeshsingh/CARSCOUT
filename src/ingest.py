import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DB_DIR, EMBEDDING_MODEL, NHTSA_COMPLAINTS_DIR, OPENAI_API_KEY, VEHICLE_SHORTLIST

COMPLAINTS_CSV = NHTSA_COMPLAINTS_DIR / "complaints.csv"
COLLECTION_NAME = "nhtsa_complaints"
UPSERT_BATCH_SIZE = 100

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
                    "metadata": {**metadata_base, "chunk_index": i, "chunk_count": len(texts)},
                }
            )

    return chunks


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def _clean_metadata(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if v is not None}


def embed_and_upsert(chunks: list[dict], collection) -> None:
    existing_complaint_ids = {m["complaint_id"] for m in collection.get(include=["metadatas"])["metadatas"]}

    pending = [c for c in chunks if c["metadata"]["complaint_id"] not in existing_complaint_ids]
    skipped = len(chunks) - len(pending)

    for start in range(0, len(pending), UPSERT_BATCH_SIZE):
        batch = pending[start : start + UPSERT_BATCH_SIZE]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[_clean_metadata(c["metadata"]) for c in batch],
        )
        print(f"  Embedded and inserted batch {start // UPSERT_BATCH_SIZE + 1}: {len(batch)} chunks")

    print(f"\nSkipped (complaint_id already in collection): {skipped}")
    print(f"Newly embedded and inserted: {len(pending)}")


def main():
    filtered = load_and_filter_complaints()
    chunks = chunk_complaints(filtered)
    print(f"\nTotal chunks to consider: {len(chunks)}")
    print(f"Example metadata: {chunks[0]['metadata']}")

    collection = get_collection()
    embed_and_upsert(chunks, collection)

    print(f"\nTotal documents in '{COLLECTION_NAME}' collection: {collection.count()}")


if __name__ == "__main__":
    main()
