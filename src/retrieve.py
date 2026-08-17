import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

from config import CHROMA_DB_DIR, EMBEDDING_MODEL, OPENAI_API_KEY
from ingest import COLLECTION_NAME
from schemas import ComplaintResult, RetrievalResponse

YEAR_TOLERANCE = 1
DEFAULT_MIN_SCORE = 0.70


def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def _embed_query(query: str) -> list[float]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


def search_complaints(
    make: str,
    model: str,
    year: int,
    query: str,
    top_k: int = 5,
    min_score: float = DEFAULT_MIN_SCORE,
) -> RetrievalResponse:
    collection = _get_collection()

    where = {
        "$and": [
            {"make": make.upper()},
            {"model": model.upper()},
            {"model_year": {"$gte": year - YEAR_TOLERANCE}},
            {"model_year": {"$lte": year + YEAR_TOLERANCE}},
        ]
    }

    query = query.strip().lower()
    query_embedding = _embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for doc, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # Collection uses Chroma's default L2 space (squared L2 distance).
        # OpenAI embeddings are unit-normalized, so squared L2 distance
        # converts to cosine similarity as: sim = 1 - distance / 2.
        similarity_score = 1 - distance / 2
        candidates.append(
            ComplaintResult(
                complaint_id=metadata["complaint_id"],
                score=similarity_score,
                make=metadata["make"],
                model=metadata["model"],
                year=metadata["model_year"],
                narrative=doc,
            )
        )

    best_score_found = max((c.score for c in candidates), default=None)
    filtered = [c for c in candidates if c.score >= min_score]

    if filtered:
        top_score = filtered[0].score
        return RetrievalResponse(
            status="ok",
            query=query,
            make=make,
            model=model,
            year=year,
            min_score=min_score,
            best_score_found=best_score_found,
            results=filtered,
            message=f"Found {len(filtered)} confident match(es) for {make} {model} {year}; top score {top_score:.3f}.",
        )

    if candidates:
        message = (
            f"No result met the {min_score:.2f} similarity threshold "
            f"(best score found: {best_score_found:.3f}). "
            "Likely a typo, an unrelated query, or no complaints on this topic for this vehicle."
        )
    else:
        message = f"No complaints found for {make} {model} {year} (±{YEAR_TOLERANCE} year) in the dataset."

    return RetrievalResponse(
        status="no_confident_match",
        query=query,
        make=make,
        model=model,
        year=year,
        min_score=min_score,
        best_score_found=best_score_found,
        results=[],
        message=message,
    )


def main():
    response = search_complaints("Toyota", "Corolla", 2016, "engine stalling while driving")
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
