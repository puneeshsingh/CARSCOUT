import pandas as pd

from ingest_recalls import RECALLS_CACHE_PARQUET
from schemas import RecallCheckResponse, RecallEntry

YEAR_TOLERANCE = 1
MAX_RECALLS_RETURNED = 5

_df: pd.DataFrame | None = None


def _get_data() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_parquet(RECALLS_CACHE_PARQUET)
    return _df


def check_recalls(make: str, model: str, year: int) -> RecallCheckResponse:
    df = _get_data()

    mask = (
        (df["Make"] == make.strip().lower())
        & (df["Model"] == model.strip().lower())
        & df["ModelYear"].between(year - YEAR_TOLERANCE, year + YEAR_TOLERANCE)
    )
    matches = df[mask].head(MAX_RECALLS_RETURNED)

    if matches.empty:
        return RecallCheckResponse(
            status="none_found",
            make=make,
            model=model,
            year=year,
            recalls=[],
            message=f"No NHTSA recall campaigns on record for a {year} {make} {model}.",
        )

    recalls = [
        RecallEntry(
            campaign_number=str(row.NHTSACampaignNumber),
            component=str(row.Component),
            summary=str(row.Summary),
            consequence=str(row.Consequence),
            remedy=str(row.Remedy),
            report_received_date=str(row.ReportReceivedDate),
        )
        for row in matches.itertuples(index=False)
    ]

    return RecallCheckResponse(
        status="ok",
        make=make,
        model=model,
        year=year,
        recalls=recalls,
        message=(
            f"Found {len(recalls)} recall campaign(s) on record for a {year} {make} {model}. "
            "This means the manufacturer issued a recall for this model/year - it does NOT mean "
            "this specific vehicle's repair is still outstanding, since that depends on its VIN "
            "and service history."
        ),
    )


def main():
    response = check_recalls("Hyundai", "Elantra", 2021)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
