import pandas as pd

from config import DATA_CACHE_DIR, NHTSA_RECALLS_CSV, VEHICLE_SHORTLIST

RECALLS_CACHE_PARQUET = DATA_CACHE_DIR / "recalls_shortlist.parquet"

RECALLS_COLUMNS = [
    "NHTSACampaignNumber",
    "ReportReceivedDate",
    "Component",
    "Summary",
    "Consequence",
    "Remedy",
    "ModelYear",
    "Make",
    "Model",
]


def load_and_filter_recalls(csv_path=NHTSA_RECALLS_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=RECALLS_COLUMNS, low_memory=False)

    df["Make"] = df["Make"].astype(str).str.strip().str.lower()
    df["Model"] = df["Model"].astype(str).str.strip().str.lower()
    df["ModelYear"] = pd.to_numeric(df["ModelYear"], errors="coerce")

    mask = pd.Series(False, index=df.index)
    for make, model, year, tolerance in VEHICLE_SHORTLIST:
        entry_mask = (df["Make"] == make) & (df["Model"] == model)
        if year is not None:
            entry_mask &= df["ModelYear"].between(year - tolerance, year + tolerance)
        mask |= entry_mask

    filtered = df[mask].copy()

    counts = filtered.groupby(["Make", "Model"]).size().sort_values(ascending=False)
    print("Recall campaign counts per make/model:")
    print(counts.to_string())

    return filtered


def main():
    DATA_CACHE_DIR.mkdir(exist_ok=True)
    filtered = load_and_filter_recalls()
    filtered.to_parquet(RECALLS_CACHE_PARQUET, index=False)
    print(f"\nSaved {len(filtered)} filtered recall campaigns to {RECALLS_CACHE_PARQUET}")


if __name__ == "__main__":
    main()
