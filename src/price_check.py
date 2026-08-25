import pandas as pd

from ingest_prices import PRICE_CACHE_PARQUET
from schemas import PriceCheckResponse

YEAR_TOLERANCE = 1
MIN_SAMPLE_SIZE = 5
ODOMETER_BAND_PCT = 0.25
ODOMETER_BAND_FLOOR = 15_000

_df: pd.DataFrame | None = None


def _get_data() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_parquet(PRICE_CACHE_PARQUET)
    return _df


def check_price(
    make: str,
    model: str,
    year: int,
    odometer: int,
    asking_price: float,
    condition: str | None = None,
) -> PriceCheckResponse:
    df = _get_data()

    band = max(odometer * ODOMETER_BAND_PCT, ODOMETER_BAND_FLOOR)
    base_mask = (
        (df["manufacturer"] == make.strip().lower())
        & (df["model"] == model.strip().lower())
        & df["year"].between(year - YEAR_TOLERANCE, year + YEAR_TOLERANCE)
        & df["odometer"].between(odometer - band, odometer + band)
    )
    comps = df[base_mask]

    used_condition_filter = False
    if condition:
        condition_mask = comps["condition"] == condition.strip().lower()
        if condition_mask.sum() >= MIN_SAMPLE_SIZE:
            comps = comps[condition_mask]
            used_condition_filter = True

    sample_size = len(comps)

    if sample_size < MIN_SAMPLE_SIZE:
        return PriceCheckResponse(
            status="insufficient_data",
            make=make,
            model=model,
            year=year,
            asking_price=asking_price,
            odometer=odometer,
            condition=condition,
            sample_size=sample_size,
            verdict="unknown",
            message=(
                f"Only {sample_size} comparable listing(s) found for a {year} {make} {model} "
                f"near {odometer:,} miles - not enough to judge whether ${asking_price:,.0f} is fair."
            ),
        )

    prices = comps["price"]
    median_price = float(prices.median())
    p25 = float(prices.quantile(0.25))
    p75 = float(prices.quantile(0.75))

    if asking_price < p25:
        verdict = "below_market"
    elif asking_price > p75:
        verdict = "above_market"
    else:
        verdict = "at_market"

    condition_note = " (matched on condition)" if used_condition_filter else ""
    message = (
        f"Found {sample_size} comparable listing(s){condition_note} for a {year} {make} {model} "
        f"near {odometer:,} miles: median ${median_price:,.0f} (range ${p25:,.0f}-${p75:,.0f}). "
        f"Asking price ${asking_price:,.0f} is {verdict.replace('_', ' ')}."
    )

    return PriceCheckResponse(
        status="ok",
        make=make,
        model=model,
        year=year,
        asking_price=asking_price,
        odometer=odometer,
        condition=condition,
        sample_size=sample_size,
        median_comp_price=median_price,
        p25_comp_price=p25,
        p75_comp_price=p75,
        verdict=verdict,
        message=message,
    )


def main():
    response = check_price("Toyota", "Corolla", 2016, 80_000, 12_000)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
