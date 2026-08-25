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


class PriceCheckResponse(BaseModel):
    status: Literal["ok", "insufficient_data"]
    make: str
    model: str
    year: int
    asking_price: float
    odometer: int
    condition: str | None = None
    sample_size: int
    median_comp_price: float | None = None
    p25_comp_price: float | None = None
    p75_comp_price: float | None = None
    verdict: Literal["below_market", "at_market", "above_market", "unknown"] = "unknown"
    message: str


class RecallEntry(BaseModel):
    campaign_number: str
    component: str
    summary: str
    consequence: str
    remedy: str
    report_received_date: str


class RecallCheckResponse(BaseModel):
    status: Literal["ok", "none_found"]
    make: str
    model: str
    year: int
    recalls: list[RecallEntry] = []
    message: str


class SafetyRatingResponse(BaseModel):
    status: Literal["ok", "not_rated", "unavailable"]
    make: str
    model: str
    year: int
    overall_rating: str | None = None
    front_crash_rating: str | None = None
    side_crash_rating: str | None = None
    rollover_rating: str | None = None
    vehicle_description: str | None = None
    message: str
