"""
Week 4 eval suite dashboard - run the eval suite and compare two saved runs.

Moved out of agent/pages/ (was "Evaluations" in the main app's nav) - this
is a dev/grading tool for inspecting eval runs, not something a used-car
buyer using the app needs to see. Run directly with:
    uv run streamlit run evals/eval_dashboard.py
The canonical way to run the suite itself is still the CLI
(`uv run python evals/run_evals.py`, see README) - this page is a
convenience viewer on top of the same saved results/*.json files.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

EVALS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALS_DIR.parent
AGENT_DIR = PROJECT_ROOT / "agent"
RESULTS_DIR = EVALS_DIR / "results"
TAXONOMY_PATH = EVALS_DIR / "taxonomy.md"

for path in (AGENT_DIR, EVALS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_evals  # noqa: E402

# Plain-English label + one-line description for each check - the raw
# snake_case names and code-level phrasing aren't meant for a demo audience.
CHECK_INFO = {
    "tool_completeness": ("Used all 4 data sources", "Checked reliability, price, recalls, and safety rating before answering."),
    "no_score_leakage": ("No confusing internal numbers", "Never shows raw technical scores the user can't interpret."),
    "recall_framing": ("Recall wording is safe", "Says to verify with a VIN, never claims a specific car has an open recall."),
    "no_duplicate_recalls": ("No duplicate recalls", "The same recall campaign isn't counted or shown more than once."),
    "injection_noncompliance": ("Resists hidden instructions", "Ignores attempts to override its instructions hidden in a question."),
    "guardrail_correctness": ("Filters input correctly", "Off-topic or inappropriate questions get blocked; real questions get through."),
    "injection_gate": ("Flags attacks, not normal questions", "Correctly tells hidden-instruction attempts apart from ordinary questions."),
}

st.set_page_config(page_title="CarScout Evals", layout="wide")
st.title("CarScout — Eval suite")
st.caption("Runs a batch of test questions through the agent and checks the answers automatically. Compare any two runs below.")

if st.button("Run eval suite", type="primary"):
    with st.spinner("Running 23 test questions through the agent and guardrails..."):
        result = run_evals.run_suite()
        RESULTS_DIR.mkdir(exist_ok=True)
        filename = result["timestamp"].replace(":", "-").replace("+00:00", "Z") + ".json"
        (RESULTS_DIR / filename).write_text(json.dumps(result, indent=2))
    st.success(f"New run saved.")
    st.rerun()


def _load_runs() -> dict:
    runs = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            runs[f.name] = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return runs


TIMESTAMP_PATTERN = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<hh>\d{2})-(?P<mm>\d{2})-(?P<ss>\d{2})"
    r"\.(?P<micro>\d+)(?P<sign>[+-])(?P<tzh>\d{2})-(?P<tzm>\d{2})"
)


def _run_label(name: str, run: dict) -> str:
    # Result filenames encode an ISO timestamp with ":" swapped for "-" so
    # they're valid on all filesystems - reverse that here for display.
    stem = name.rsplit(".json", 1)[0]
    match = TIMESTAMP_PATTERN.match(stem)
    ts = None
    if match:
        g = match.groupdict()
        iso = f"{g['date']}T{g['hh']}:{g['mm']}:{g['ss']}.{g['micro']}{g['sign']}{g['tzh']}:{g['tzm']}"
        try:
            ts = datetime.fromisoformat(iso)
        except ValueError:
            ts = None
    when = ts.strftime("%b %d, %I:%M %p") if ts else name
    rate = run.get("overall_pass_rate")
    rate_text = f"{rate:.0%} passing" if rate is not None else "no data"
    return f"{when} — {rate_text} ({run.get('num_cases', '?')} cases)"


def _top_failure_from_taxonomy(text: str) -> dict | None:
    """Pull row #1 out of the markdown ranking table in taxonomy.md."""
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0] == "1":
            return {
                "category": re.sub(r"\*\*", "", cells[1]),
                "frequency": cells[2],
                "impact": cells[3],
                "reasoning": cells[4],
            }
    return None


runs = _load_runs()

if not runs:
    st.info('No eval runs yet - click "Run eval suite" above.')
    st.stop()

run_names = list(runs.keys())
labels = {name: _run_label(name, runs[name]) for name in run_names}
default_after = run_names[-1]
default_before = run_names[-2] if len(run_names) >= 2 else run_names[-1]

col1, col2 = st.columns(2)
with col1:
    before_name = st.selectbox(
        "Before", run_names, index=run_names.index(default_before), format_func=lambda n: labels[n]
    )
with col2:
    after_name = st.selectbox(
        "After", run_names, index=run_names.index(default_after), format_func=lambda n: labels[n]
    )

before, after = runs[before_name], runs[after_name]
before_rate = before["overall_pass_rate"] or 0
after_rate = after["overall_pass_rate"] or 0
delta = after_rate - before_rate

with st.container(border=True):
    st.markdown("##### Overall result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Before", f"{before_rate:.0%}")
    m2.metric("After", f"{after_rate:.0%}", delta=f"{delta:+.0%}")
    if after_rate >= 0.999:
        m3.success("All checks passing")
    elif delta > 0:
        m3.warning("Improved, not perfect yet")
    elif delta < 0:
        m3.error("Got worse")
    else:
        m3.info("No change")

st.markdown("##### What each check looks for")
check_names = sorted(set(before["check_summary"]) | set(after["check_summary"]))
rows = []
for name in check_names:
    label, desc = CHECK_INFO.get(name, (name, ""))
    b = before["check_summary"].get(name, {"passed": 0, "total": 0, "rate": None})
    a = after["check_summary"].get(name, {"passed": 0, "total": 0, "rate": None})
    b_rate = b["rate"] * 100 if b["rate"] is not None else None
    a_rate = a["rate"] * 100 if a["rate"] is not None else None
    if b_rate is not None and a_rate is not None:
        change = f"{a_rate - b_rate:+.0f} pts"
    else:
        change = "—"
    rows.append(
        {
            "Check": label,
            "What it checks": desc,
            "Before": b_rate,
            "After": a_rate,
            "Change": change,
        }
    )

st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    use_container_width=True,
    column_config={
        "Before": st.column_config.ProgressColumn("Before", min_value=0, max_value=100, format="%.0f%%"),
        "After": st.column_config.ProgressColumn("After", min_value=0, max_value=100, format="%.0f%%"),
    },
)

st.markdown("##### Top failure found")
taxonomy_text = TAXONOMY_PATH.read_text(encoding="utf-8") if TAXONOMY_PATH.exists() else ""
top = _top_failure_from_taxonomy(taxonomy_text)
if top:
    with st.container(border=True):
        st.markdown(f"**{top['category']}**")
        st.write(top["reasoning"])
        c1, c2 = st.columns(2)
        c1.caption(f"How often it happened: {top['frequency']}")
        c2.caption(f"How bad it was: {top['impact']}")
else:
    st.write("No taxonomy written yet - see evals/taxonomy.md.")

with st.expander("Full technical write-up (evals/taxonomy.md)"):
    if taxonomy_text:
        st.markdown(taxonomy_text)
    else:
        st.write("No taxonomy written yet.")
