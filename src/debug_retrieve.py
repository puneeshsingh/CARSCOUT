from retrieve import search_complaints

TEST_CASES = [
    ("Hyundai", "Elantra", 2021, "iol issue"),
    ("Hyundai", "Elantra", 2021, "oil issue"),
    ("Hyundai", "Elantra", 2021, "engine stalling"),
]


def print_results(make: str, model: str, year: int, query: str) -> None:
    print("=" * 100)
    print(f"Query: make={make!r} model={model!r} year={year} | text={query!r}")
    print("=" * 100)

    response = search_complaints(make, model, year, query, top_k=5)

    print(f"status={response.status}  best_score_found={response.best_score_found}")
    print(f"message: {response.message}")

    if response.status == "ok":
        for rank, r in enumerate(response.results, start=1):
            print(f"\n[{rank}] complaint_id={r.complaint_id}  score={r.score:.4f}  {r.make} {r.model} {r.year}")
            print(f"    {r.narrative}")
    print()


def main():
    for make, model, year, query in TEST_CASES:
        print_results(make, model, year, query)


if __name__ == "__main__":
    main()
