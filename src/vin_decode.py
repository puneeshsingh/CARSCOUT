import httpx

VPIC_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{vin}?format=json"


def decode_vin(vin: str) -> dict:
    """Decode a VIN via NHTSA's free vPIC API - no key required.

    Returns {"status": "ok", "make": str, "model": str, "year": int} on a
    clean decode, or {"status": "error", "error": str} otherwise. Used as a
    real, live cross-check on top of the curated demo listing data (which
    already knows the make/model/year) - not the source of truth itself.
    """
    try:
        response = httpx.get(VPIC_URL.format(vin=vin), timeout=10)
        response.raise_for_status()
        result = response.json()["Results"][0]
        make, model, year = result.get("Make"), result.get("Model"), result.get("ModelYear")
        if not make or not model or not year:
            return {"status": "error", "error": "VIN did not decode to a make/model/year"}
        return {"status": "ok", "make": make, "model": model, "year": int(year)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    for entry in [
        "KM8K2CAA8LU462861", "3MZBN1L3XHM151293", "3KPF24AD1LE141393",
        "KMHD74LF1LU900547", "2HGFG3B06FH517149", "5YFBURHE8HP655686",
    ]:
        print(entry, "->", decode_vin(entry))


if __name__ == "__main__":
    main()
