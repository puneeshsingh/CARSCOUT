import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from mcp.server.fastmcp import FastMCP

from retrieve import DEFAULT_MIN_SCORE, RetrievalResponse
from retrieve import search_complaints as _search_complaints

mcp = FastMCP("carscout-retrieval")


@mcp.tool()
def search_complaints(
    make: str,
    model: str,
    year: int,
    query: str,
    top_k: int = 5,
    min_score: float = DEFAULT_MIN_SCORE,
) -> RetrievalResponse:
    """Search NHTSA complaint narratives for a specific vehicle make, model, and
    year to find known reliability issues (e.g. engine stalling, transmission
    failure, oil consumption, rebuilt title patterns). Call this when the user
    asks about problems, complaints, or reliability history for a specific
    vehicle. `year` matches within +/-1 model year. Returns a structured result
    with a confidence status - if status is "no_confident_match", treat it as
    "no reliable data found," not as a real answer.
    """
    return _search_complaints(make, model, year, query, top_k=top_k, min_score=min_score)


if __name__ == "__main__":
    mcp.run()
