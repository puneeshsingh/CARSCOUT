"""
Eval suite runner for the CarScout due-diligence agent.

Usage: uv run python evals/run_evals.py
Writes one timestamped result file to evals/results/, and prints a summary.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"

sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(EVALS_DIR))

import complaint_lookup_agent as cla  # noqa: E402
import guards  # noqa: E402
from cases import all_cases  # noqa: E402
from checks import (  # noqa: E402
    AGENT_CHECKS,
    check_benign_no_false_positive,
    check_guardrail,
    check_injection_noncompliance,
)

GUARD_FNS = {"moderation": guards.check_moderation, "relevance": guards.check_relevance}


def _run_case(case: dict) -> dict:
    tags = case["tags"]

    if "guardrail" in tags:
        guard_result = GUARD_FNS[case["guard_fn"]](case["symptom"])
        check_result = check_guardrail(case, guard_result)
        return {"id": case["id"], "tags": tags, "checks": [check_result]}

    trace = cla.run_with_trace(
        case["make"], case["model"], case["year"], case["symptom"],
        asking_price=float(case["asking_price"]), odometer=int(case["odometer"]),
        condition=case.get("condition"),
    )

    check_results = [check(case, trace) for check in AGENT_CHECKS]
    if "adversarial" in tags:
        check_results.append(check_injection_noncompliance(case, trace))
    if "benign_repeat" in tags:
        check_results.append(check_benign_no_false_positive(case, trace))

    return {
        "id": case["id"],
        "tags": tags,
        "checks": check_results,
        "final_answer": trace["final_answer"],
        "steps": trace["steps"],  # kept so future checks can be scored retroactively without re-running the agent
    }


def run_suite() -> dict:
    cases = all_cases()
    case_results = [_run_case(case) for case in cases]

    check_summary: dict[str, dict] = {}
    for case_result in case_results:
        for check in case_result["checks"]:
            if not check["applicable"]:
                continue
            summary = check_summary.setdefault(check["name"], {"passed": 0, "total": 0})
            summary["total"] += 1
            if check["passed"]:
                summary["passed"] += 1

    for name, summary in check_summary.items():
        summary["rate"] = summary["passed"] / summary["total"] if summary["total"] else None

    total_applicable = sum(s["total"] for s in check_summary.values())
    total_passed = sum(s["passed"] for s in check_summary.values())

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_cases": len(cases),
        "check_summary": check_summary,
        "overall_pass_rate": total_passed / total_applicable if total_applicable else None,
        "cases": case_results,
    }


def main():
    result = run_suite()

    RESULTS_DIR.mkdir(exist_ok=True)
    filename = result["timestamp"].replace(":", "-").replace("+00:00", "Z") + ".json"
    out_path = RESULTS_DIR / filename
    out_path.write_text(json.dumps(result, indent=2))

    print(f"Ran {result['num_cases']} cases -> {out_path}\n")
    print(f"{'check':30s} {'passed':>8s} {'total':>8s} {'rate':>8s}")
    for name, summary in result["check_summary"].items():
        print(f"{name:30s} {summary['passed']:8d} {summary['total']:8d} {summary['rate']:8.2%}")
    print(f"\nOverall pass rate: {result['overall_pass_rate']:.2%}")


if __name__ == "__main__":
    main()
