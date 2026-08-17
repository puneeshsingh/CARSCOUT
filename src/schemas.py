from typing import Literal

from pydantic import BaseModel


class ComplaintResult(BaseModel):
    complaint_id: str
    score: float
    make: str
    model: str
    year: int
    narrative: str


class RetrievalResponse(BaseModel):
    status: Literal["ok", "no_confident_match"]
    query: str
    make: str
    model: str
    year: int
    min_score: float
    best_score_found: float | None = None
    results: list[ComplaintResult] = []
    message: str
