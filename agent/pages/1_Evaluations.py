"""
Week 4 eval suite page - run the eval suite and compare two saved runs.
"""

import json
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

AGENT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AGENT_DIR.parent
EVALS_DIR = PROJECT_ROOT / "evals"
RESULTS_DIR = EVALS_DIR / "results"
TAXONOMY_PATH = EVALS_DIR / "taxonomy.md"

for path in (AGENT_DIR, EVALS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_evals  # noqa: E402

st.set_page_config(page_title="CarScout Evals", layout="wide")
st.title("CarScout — Eval suite")
st.caption("Runs the Week 4 eval suite (evals/) against the live agent and guardrails, and compares two saved runs.")

if st.button("Run eval suite", type="primary"):
    with st.spinner("Running 23 cases through the agent and guardrails..."):
        result = run_evals.run_suite()
        RESULTS_DIR.mkdir(exist_ok=True)
        filename = result["timestamp"].replace(":", "-").replace("+00:00", "Z") + ".json"
        (RESULTS_DIR / filename).write_text(json.dumps(result, indent=2))
    st.success(f"Saved new run: {filename}")
    st.rerun()


def _load_runs() -> dict:
    runs = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            runs[f.name] = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return runs


runs = _load_runs()

if not runs:
    st.info('No eval runs yet - click "Run eval suite" above.')
    st.stop()

run_names = list(runs.keys())
default_after = run_names[-1]
default_before = run_names[-2] if len(run_names) >= 2 else run_names[-1]

col1, col2 = st.columns(2)
with col1:
    before_name = st.selectbox("Before run", run_names, index=run_names.index(default_before))
with col2:
    after_name = st.selectbox("After run", run_names, index=run_names.index(default_after))

before, after = runs[before_name], runs[after_name]

st.subheader("Overall pass rate")
m1, m2, m3 = st.columns(3)
before_rate = before["overall_pass_rate"] or 0
after_rate = after["overall_pass_rate"] or 0
m1.metric("Before", f"{before_rate:.1%}")
m2.metric("After", f"{after_rate:.1%}", delta=f"{(after_rate - before_rate):+.1%}")
m3.metric("Cases per run", after["num_cases"])

st.subheader("Per-check pass rate")
check_names = sorted(set(before["check_summary"]) | set(after["check_summary"]))
for name in check_names:
    b = before["check_summary"].get(name, {"passed": 0, "total": 0, "rate": None})
    a = after["check_summary"].get(name, {"passed": 0, "total": 0, "rate": None})
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.write(f"**{name}**")
    b_text = f"before: {b['passed']}/{b['total']}" + (f" ({b['rate']:.0%})" if b["rate"] is not None else "")
    c2.write(b_text)
    a_text = f"after: {a['passed']}/{a['total']}" + (f" ({a['rate']:.0%})" if a["rate"] is not None else "")
    if b["rate"] is not None and a["rate"] is not None:
        delta = a["rate"] - b["rate"]
        a_text += f"  Δ {delta:+.0%}"
    c3.write(a_text)

st.subheader("Failure taxonomy")
if TAXONOMY_PATH.exists():
    st.markdown(TAXONOMY_PATH.read_text(encoding="utf-8"))
else:
    st.write("No taxonomy written yet - see evals/taxonomy.md.")
