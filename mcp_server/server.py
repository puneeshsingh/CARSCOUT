import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from mcp.server.fastmcp import FastMCP

from price_check import check_price as _check_price
from recall_check import check_recalls as _check_recalls
from retrieve import DEFAULT_MIN_SCORE, RetrievalResponse
from retrieve import search_complaints as _search_complaints
from safety_rating import check_safety_rating as _check_safety_rating
from schemas import PriceCheckResponse, RecallCheckResponse, SafetyRatingResponse

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


@mcp.tool()
def check_price_estimate(
    make: str,
    model: str,
    year: int,
    odometer: int,
    asking_price: float,
    condition: str | None = None,
) -> PriceCheckResponse:
    """Check whether a listing's asking price is fair by comparing it against
    real comparable listings (same make/model, +/-1 model year, similar
    mileage). Call this whenever you're given a specific listing's price and
    mileage to evaluate. Returns a `verdict` of "below_market", "at_market",
    "above_market", or "unknown". If status is "insufficient_data", there
    weren't enough comparable listings to judge - treat it as "no reliable
    price signal," not as a real answer. Never surface the raw
    median/percentile numbers to the end user - only the plain-language
    verdict.
    """
    return _check_price(make, model, year, odometer, asking_price, condition=condition)


@mcp.tool()
def check_recalls(make: str, model: str, year: int) -> RecallCheckResponse:
    """Check NHTSA's official recall database for a specific vehicle make,
    model, and year. Call this whenever evaluating a specific vehicle, not
    just when the user asks about recalls directly. `year` matches within
    +/-1 model year. IMPORTANT: a result here means the manufacturer issued a
    recall campaign for this make/model/year - it does NOT mean this specific
    listing's repair is still outstanding, since that depends on the actual
    vehicle's VIN and service history, which this tool does not have. Always
    phrase findings as recall *history*, and recommend confirming repair
    status by VIN at a dealer or NHTSA's own lookup - never say a specific
    listing "has an open recall."
    """
    return _check_recalls(make, model, year)


@mcp.tool()
def check_safety_rating(make: str, model: str, year: int) -> SafetyRatingResponse:
    """Look up NHTSA's official crash-test safety rating (1-5 stars) for a
    specific vehicle make, model, and year, via a live call to NHTSA's public
    API. Call this whenever evaluating a specific vehicle. If status is
    "not_rated", NHTSA has no rating on file for this vehicle - say so plainly,
    don't guess. If status is "unavailable", the live lookup failed (e.g.
    network issue) - skip this signal in your answer rather than blocking on
    it or retrying.
    """
    return _check_safety_rating(make, model, year)


if __name__ == "__main__":
    mcp.run()
