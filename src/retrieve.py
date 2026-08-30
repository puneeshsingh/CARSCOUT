from openai import OpenAI
from pinecone import Pinecone

from config import EMBEDDING_MODEL, OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
from ingest import NARRATIVE_METADATA_KEY
from schemas import ComplaintResult, RetrievalResponse

YEAR_TOLERANCE = 1
DEFAULT_MIN_SCORE = 0.70


def _get_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


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
    index = _get_index()

    where = {
        "make": {"$eq": make.upper()},
        "model": {"$eq": model.upper()},
        "model_year": {"$gte": year - YEAR_TOLERANCE, "$lte": year + YEAR_TOLERANCE},
    }

    query = query.strip().lower()
    query_embedding = _embed_query(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        filter=where,
        include_metadata=True,
    )

    candidates = []
    for match in results["matches"]:
        metadata = match["metadata"]
        # Index uses cosine metric, so Pinecone's match score is already a
        # cosine similarity (higher = more similar) - no conversion needed.
        candidates.append(
            ComplaintResult(
                complaint_id=metadata["complaint_id"],
                score=match["score"],
                make=metadata["make"],
                model=metadata["model"],
                year=metadata["model_year"],
                narrative=metadata[NARRATIVE_METADATA_KEY],
            )
        )

    best_score_found = max((c.score for c in candidates), default=None)
    filtered = [c for c in candidates if c.score >= min_score]
    # candidates is already sorted best-first (Pinecone returns matches in
    # descending score order), so candidates[0] is the best_score_found owner.
    closest_candidate = candidates[0] if candidates else None

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
            closest_candidate=closest_candidate,
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
        closest_candidate=closest_candidate,
        message=message,
    )


def main():
    response = search_complaints("Toyota", "Corolla", 2016, "engine stalling while driving")
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
