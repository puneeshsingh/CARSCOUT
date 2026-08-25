import httpx

from schemas import SafetyRatingResponse

BASE_URL = "https://api.nhtsa.gov/SafetyRatings"
TIMEOUT_SECONDS = 5.0


def check_safety_rating(make: str, model: str, year: int) -> SafetyRatingResponse:
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            variants_resp = client.get(
                f"{BASE_URL}/modelyear/{year}/make/{make.strip()}/model/{model.strip()}"
            )
            variants_resp.raise_for_status()
            variants = variants_resp.json().get("Results", [])

            if not variants:
                return SafetyRatingResponse(
                    status="not_rated",
                    make=make,
                    model=model,
                    year=year,
                    message=f"NHTSA has no crash-test rating on file for a {year} {make} {model}.",
                )

            vehicle_id = variants[0]["VehicleId"]
            trim_note = " (rating reflects a representative trim, may vary from the exact one)" if len(variants) > 1 else ""

            rating_resp = client.get(f"{BASE_URL}/VehicleId/{vehicle_id}")
            rating_resp.raise_for_status()
            results = rating_resp.json().get("Results", [])

            if not results:
                return SafetyRatingResponse(
                    status="not_rated",
                    make=make,
                    model=model,
                    year=year,
                    message=f"NHTSA has no crash-test rating on file for a {year} {make} {model}.",
                )

            rating = results[0]
            overall = rating.get("OverallRating")

            if not overall or overall == "Not Rated":
                return SafetyRatingResponse(
                    status="not_rated",
                    make=make,
                    model=model,
                    year=year,
                    vehicle_description=rating.get("VehicleDescription"),
                    message=f"NHTSA has not assigned an overall crash-test rating for a {year} {make} {model}.",
                )

            return SafetyRatingResponse(
                status="ok",
                make=make,
                model=model,
                year=year,
                overall_rating=overall,
                front_crash_rating=rating.get("OverallFrontCrashRating"),
                side_crash_rating=rating.get("OverallSideCrashRating"),
                rollover_rating=rating.get("RolloverRating"),
                vehicle_description=rating.get("VehicleDescription"),
                message=(
                    f"NHTSA overall crash-test rating for a {year} {make} {model}: {overall}/5 stars"
                    f"{trim_note}."
                ),
            )
    except Exception as e:
        return SafetyRatingResponse(
            status="unavailable",
            make=make,
            model=model,
            year=year,
            message=f"Could not reach NHTSA's safety rating service right now ({e.__class__.__name__}).",
        )


def main():
    response = check_safety_rating("Toyota", "Corolla", 2016)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
