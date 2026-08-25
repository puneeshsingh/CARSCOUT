import pandas as pd

from config import CRAIGSLIST_VEHICLES_CSV, DATA_CACHE_DIR, VEHICLE_SHORTLIST

PRICE_CACHE_PARQUET = DATA_CACHE_DIR / "vehicles_shortlist.parquet"

USECOLS = ["price", "year", "manufacturer", "model", "condition", "odometer", "title_status"]

CHUNK_SIZE = 50_000
MIN_PRICE = 500
MAX_PRICE = 200_000


def _filter_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.dropna(subset=["manufacturer", "model", "year", "price", "odometer"]).copy()
    chunk["manufacturer"] = chunk["manufacturer"].astype(str).str.strip().str.lower()
    chunk["model"] = chunk["model"].astype(str).str.strip().str.lower()
    chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce")
    chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
    chunk["odometer"] = pd.to_numeric(chunk["odometer"], errors="coerce")

    mask = pd.Series(False, index=chunk.index)
    for make, model, year, tolerance in VEHICLE_SHORTLIST:
        entry_mask = (chunk["manufacturer"] == make) & (chunk["model"] == model)
        if year is not None:
            entry_mask &= chunk["year"].between(year - tolerance, year + tolerance)
        mask |= entry_mask

    filtered = chunk[mask].copy()
    filtered = filtered[
        filtered["price"].between(MIN_PRICE, MAX_PRICE)
        & (filtered["odometer"] > 0)
        & filtered["year"].notna()
    ]
    return filtered


def load_and_filter_vehicles(csv_path=CRAIGSLIST_VEHICLES_CSV) -> pd.DataFrame:
    filtered_chunks = []
    reader = pd.read_csv(csv_path, usecols=USECOLS, chunksize=CHUNK_SIZE, low_memory=False)
    for i, chunk in enumerate(reader):
        filtered_chunks.append(_filter_chunk(chunk))
        if (i + 1) % 10 == 0:
            print(f"  Processed {(i + 1) * CHUNK_SIZE:,} rows...")

    result = pd.concat(filtered_chunks, ignore_index=True) if filtered_chunks else pd.DataFrame(columns=USECOLS)

    counts = result.groupby(["manufacturer", "model"]).size().sort_values(ascending=False)
    print("\nListing counts per make/model:")
    print(counts.to_string())

    return result


def main():
    DATA_CACHE_DIR.mkdir(exist_ok=True)
    filtered = load_and_filter_vehicles()
    filtered.to_parquet(PRICE_CACHE_PARQUET, index=False)
    print(f"\nSaved {len(filtered)} filtered listings to {PRICE_CACHE_PARQUET}")


if __name__ == "__main__":
    main()
