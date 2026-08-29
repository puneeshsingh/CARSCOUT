"""
Eval suite runner for the CarScout due-diligence agent.

Usage:
  uv run python evals/run_evals.py           # full suite (24 cases)
  uv run python evals/run_evals.py --golden  # golden dataset only (3 cases)

The golden dataset is cases.py's GOLDEN_CASES plus the two VEHICLE_CASES
entries also tagged "golden" - each one pinned to a real, once-observed
failure (Week 4 Path B stretch: "convert production failures into a
regression test suite"). Run with --golden right after touching the agent,
guards, or a check tool, for a ~3-case/couple-minute check that a known bug
didn't come back, before spending the time on the full 24-case suite.

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
    check_guardrail,
    check_injection_gate,
    check_injection_noncompliance,
)

GUARD_FNS = {"moderation": guards.check_moderation, "relevance": guards.check_relevance}


def _run_case(case: dict) -> dict:
    tags = case["tags"]

    if "guardrail" in tags:
        guard_result = GUARD_FNS[case["guard_fn"]](case["symptom"])
        check_result = check_guardrail(case, guard_result)
        return {"id": case["id"], "tags": tags, "checks": [check_result], "origin": case.get("origin")}

    trace = cla.run_with_trace(
        case["make"], case["model"], case["year"], case["symptom"],
        asking_price=float(case["asking_price"]), odometer=int(case["odometer"]),
        condition=case.get("condition"),
    )

    check_results = [check(case, trace) for check in AGENT_CHECKS]
    if "adversarial" in tags:
        check_results.append(check_injection_noncompliance(case, trace))
    if "adversarial" in tags or "benign_repeat" in tags:
        injection_result = guards.check_injection(case["symptom"])
        check_results.append(check_injection_gate(case, injection_result))

    return {
        "id": case["id"],
        "tags": tags,
        "checks": check_results,
        "final_answer": trace["final_answer"],
        "steps": trace["steps"],  # kept so future checks can be scored retroactively without re-running the agent
        "origin": case.get("origin"),
    }


def run_suite(cases: list[dict] | None = None) -> dict:
    if cases is None:
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
    golden_only = "--golden" in sys.argv
    cases = [c for c in all_cases() if "golden" in c["tags"]] if golden_only else None
    result = run_suite(cases)

    RESULTS_DIR.mkdir(exist_ok=True)
    prefix = "golden_" if golden_only else ""
    filename = prefix + result["timestamp"].replace(":", "-").replace("+00:00", "Z") + ".json"
    out_path = RESULTS_DIR / filename
    out_path.write_text(json.dumps(result, indent=2))

    label = "golden dataset" if golden_only else "cases"
    print(f"Ran {result['num_cases']} {label} -> {out_path}\n")
    print(f"{'check':30s} {'passed':>8s} {'total':>8s} {'rate':>8s}")
    for name, summary in result["check_summary"].items():
        print(f"{name:30s} {summary['passed']:8d} {summary['total']:8d} {summary['rate']:8.2%}")
    print(f"\nOverall pass rate: {result['overall_pass_rate']:.2%}")

    if golden_only:
        for case_result in result["cases"]:
            status = "PASS" if all(c["passed"] for c in case_result["checks"] if c["applicable"]) else "FAIL"
            print(f"  [{status}] {case_result['id']}")
            if status == "FAIL":
                print(f"           regression of: {case_result['origin']}")


if __name__ == "__main__":
    main()
