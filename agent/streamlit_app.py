"""
Minimal demo UI for complaint_lookup_agent.py - runs the deep agent
(GPT-4o-mini + carscout_retrieval MCP tools: reliability complaints, price
comps, recalls, safety rating) against real NHTSA and Craigslist data and
renders the Think -> Act -> Observe trace plus token cost.

For the bootcamp cohort demo. Not polished, not for production use.
"""

import sys
from datetime import timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import complaint_lookup_agent as cla  # noqa: E402  (must load .env first, sets up logging)
import guards  # noqa: E402
import memory_store  # noqa: E402  (needs SRC_DIR on sys.path for its `config` import)

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


def _format_pacific(dt):
    """Recent-search timestamps are stored in UTC; render in Pacific time.

    SQLite drops tzinfo on round-trip (Postgres keeps it) - dt.tzinfo is
    None either way isn't safe to assume, so normalize to UTC first before
    converting. %Z renders "PST"/"PDT" correctly for the date, since
    zoneinfo (unlike a fixed UTC-8 offset) accounts for daylight saving.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(PACIFIC_TZ).strftime("%b %d, %I:%M %p %Z")


def _escape_for_markdown(text: str) -> str:
    """Streamlit's markdown renderer (st.success/st.warning/st.markdown)
    treats "$...$" as inline LaTeX - two dollar amounts in the same block of
    text (e.g. "$15,000" and "$17,991") get read as opening/closing math
    delimiters, and the literal "$" characters vanish from what's displayed.
    This is display-only: escape right before rendering, never before
    storing to the DB, so the saved text keeps a real "$"."""
    return text.replace("$", "\\$")

st.set_page_config(page_title="CarScout Due-Diligence Agent", layout="wide")


@st.cache_resource
def _get_memory_engine():
    engine = memory_store.get_engine()
    memory_store.init_db(engine)
    return engine


memory_engine = _get_memory_engine()

# (make, model, default_year) - the same shortlist the retrieval dataset was
# filtered to when the Chroma collection was built (see CarScout/src/ingest.py).
# Kept as a plain literal here rather than imported from src/ingest.py to
# avoid pulling in that module's heavier deps (pandas, chromadb) just for a
# 6-entry list.
VEHICLE_SHORTLIST = [
    ("Hyundai", "Kona", 2020),
    ("Mazda", "Mazda3", 2017),
    ("Kia", "Forte", 2021),
    ("Hyundai", "Elantra", 2021),
    ("Honda", "Civic", 2015),
    ("Toyota", "Corolla", 2016),
]
VEHICLE_LABELS = [f"{make} {model} ({year})" for make, model, year in VEHICLE_SHORTLIST]
VEHICLE_LABEL_BY_MAKE_MODEL = {(make, model): label for (make, model, _), label in zip(VEHICLE_SHORTLIST, VEHICLE_LABELS)}

CONDITION_OPTIONS = ["Not sure / skip", "new", "like new", "excellent", "good", "fair", "salvage"]

DEFAULT_SYMPTOM = "Is engine stalling a known issue for this vehicle?"

# Form fields are session_state-backed (rather than plain `value=` literals)
# so a "Use this search" click in the Recent Searches sidebar can populate
# them before the form widgets are created below.
st.session_state.setdefault("form_vehicle_label", VEHICLE_LABELS[0])
st.session_state.setdefault("form_year", VEHICLE_SHORTLIST[0][2])
st.session_state.setdefault("_last_vehicle_label", st.session_state["form_vehicle_label"])
st.session_state.setdefault("form_price", 15000)
st.session_state.setdefault("form_odometer", 45000)
st.session_state.setdefault("form_condition", "Not sure / skip")
st.session_state.setdefault("form_symptom", DEFAULT_SYMPTOM)

with st.sidebar:
    st.subheader("Recent searches")
    st.caption(
        "Persists across restarts (SQLite locally, Postgres when deployed) - "
        "no login, so this list is shared across everyone using this app."
    )
    recent = memory_store.get_recent_searches(memory_engine)
    if not recent:
        st.write("No searches yet.")
    for entry in recent:
        with st.container(border=True):
            st.markdown(f"**{entry.year} {entry.make} {entry.model}**")
            st.caption(
                f"{entry.symptom} · ${entry.asking_price:,.0f} / {entry.odometer:,} mi · "
                f"{_format_pacific(entry.created_at)}"
            )
            # st.text (not st.write/st.markdown): never interprets markdown,
            # so a preview built from any saved row (old or new format)
            # always renders the same way, card to card.
            st.text(memory_store.build_preview(entry.full_answer))
            with st.expander("View full report"):
                st.markdown(_escape_for_markdown(entry.full_answer))
            if st.button("Use this search", key=f"reuse_{entry.id}"):
                label = VEHICLE_LABEL_BY_MAKE_MODEL.get((entry.make, entry.model), VEHICLE_LABELS[0])
                st.session_state["form_vehicle_label"] = label
                st.session_state["_last_vehicle_label"] = label
                st.session_state["form_year"] = entry.year
                st.session_state["form_price"] = entry.asking_price
                st.session_state["form_odometer"] = entry.odometer
                st.session_state["form_condition"] = entry.condition or "Not sure / skip"
                st.session_state["form_symptom"] = entry.symptom
                st.rerun()

st.title("CarScout Due-Diligence Agent")
st.caption(
    "Runs a deepagents agent (GPT-4o-mini + 4 carscout_retrieval MCP tools) against real NHTSA "
    "and Craigslist data to check reliability complaints, price fairness, recall history, and "
    "crash-safety rating. The agent is instructed to answer only from tool results, never from "
    "its own training knowledge."
)

col1, col2 = st.columns([2, 1])
with col1:
    selected_label = st.selectbox("Vehicle", VEHICLE_LABELS, key="form_vehicle_label")
selected_make, selected_model, selected_default_year = VEHICLE_SHORTLIST[VEHICLE_LABELS.index(selected_label)]

if selected_label != st.session_state["_last_vehicle_label"]:
    # Vehicle just changed (by the user, or by a "Use this search" click that
    # didn't also set the year) - follow that vehicle's default year.
    st.session_state["form_year"] = selected_default_year
    st.session_state["_last_vehicle_label"] = selected_label

with col2:
    year = st.number_input("Year", min_value=1990, max_value=2030, step=1, key="form_year")

make, vehicle_model = selected_make, selected_model

col3, col4, col5 = st.columns(3)
with col3:
    asking_price = st.number_input("Asking price ($)", min_value=0, step=500, key="form_price")
with col4:
    odometer = st.number_input("Odometer (miles)", min_value=0, step=1000, key="form_odometer")
with col5:
    condition_choice = st.selectbox("Condition", CONDITION_OPTIONS, key="form_condition")
condition = None if condition_choice == "Not sure / skip" else condition_choice

symptom = st.text_input("Symptom / question", key="form_symptom")

PHASE_LABELS = {"think": "THINK", "act": "ACT", "observe": "OBSERVE"}

if st.button("Run agent", type="primary"):
    make_clean = make.strip()
    model_clean = vehicle_model.strip()
    symptom_clean = symptom.strip()

    errors = []
    if not make_clean or not model_clean:
        errors.append("Please enter both make and model before running the agent.")
    if year is None:
        errors.append("Please enter a valid year before running the agent.")
    if not asking_price or asking_price <= 0:
        errors.append("Please enter the listing's asking price before running the agent.")
    if not odometer or odometer <= 0:
        errors.append("Please enter the listing's odometer reading before running the agent.")
    if not symptom_clean:
        errors.append("Please describe a symptom or ask a question before running the agent.")

    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    # Gate 1: moderation - must run before the relevance check and before the
    # agent ever sees this input. Blocks on flagged=True or on API error
    # (fail closed rather than letting unmoderated input through).
    with st.spinner("Checking input..."):
        moderation = guards.check_moderation(symptom_clean)

    if moderation["status"] == "flagged":
        st.error("This input can't be processed - please describe a vehicle issue.")
        st.stop()
    elif moderation["status"] == "error":
        st.error("Something went wrong checking your input - please try again.")
        st.stop()

    # Gate 2: symptom relevance - only reached if moderation passed. Blocks
    # off-topic input ("hello", "test") before spending an agent/tool call.
    with st.spinner("Checking input..."):
        relevance = guards.check_relevance(symptom_clean)

    if relevance["status"] == "irrelevant":
        st.error("Please describe the actual issue - e.g. 'engine stalling at low speeds' rather than just 'vehicle' or 'problem'.")
        st.stop()
    elif relevance["status"] == "error":
        st.error("Something went wrong checking your input - please try again.")
        st.stop()

    # Gate 3: injection detection - non-blocking (informational only), kept
    # as its own narrowly-scoped classifier call rather than folded into the
    # agent's report-writing completion, which mixed the two tasks and
    # false-flagged ~2 in 5 wholly benign inputs (see evals/taxonomy.md).
    # "error" is treated the same as "clean" - never blocks the run.
    with st.spinner("Checking input..."):
        injection_check = guards.check_injection(symptom_clean)

    # Gate 4: the agent itself - only reached if both blocking gates passed.
    with st.spinner("Agent is running (Think -> Act -> Observe)..."):
        trace = cla.run_with_trace(
            make_clean, model_clean, int(year), symptom_clean,
            asking_price=float(asking_price), odometer=int(odometer), condition=condition,
        )

    final_answer = trace["final_answer"]
    if injection_check["status"] == "flagged":
        final_answer = guards.INJECTION_NOTE + final_answer

    if trace["hit_step_cap"]:
        st.warning(f"Agent hit the {cla.MAX_STEPS}-step cap without a confident final answer (failed closed).")
    else:
        # Write gate: only save runs that produced a real answer, not a
        # step-cap failure - matches the "stable, high-confidence facts only"
        # memory write policy.
        memory_store.save_search(
            memory_engine, make_clean, model_clean, int(year),
            float(asking_price), int(odometer), condition, symptom_clean,
            final_answer,
        )

    st.subheader("Final answer")
    if trace["is_no_confident_match"]:
        # Amber, not green - a no-confident-match result is not a "clean
        # bill of health" and shouldn't be styled like a reassuring success.
        st.warning(_escape_for_markdown(final_answer))
    else:
        st.success(_escape_for_markdown(final_answer))

    st.subheader("Think → Act → Observe trace")
    if not trace["steps"]:
        st.write("No steps recorded.")
    for i, step in enumerate(trace["steps"], start=1):
        label = PHASE_LABELS.get(step["phase"], step["phase"].upper())
        with st.expander(f"{i}. {label}", expanded=False):
            if step["phase"] == "act":
                st.code(f"{step['tool']}({step['args']})", language="python")
            else:
                st.write(step["text"])

    st.subheader("Token usage & estimated cost")
    total_tokens = trace["total_input_tokens"] + trace["total_output_tokens"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Input tokens", trace["total_input_tokens"])
    m2.metric("Output tokens", trace["total_output_tokens"])
    m3.metric("Total tokens", total_tokens)
    m4.metric("Est. cost (USD)", f"${trace['estimated_cost_usd']:.5f}")
    st.caption(
        f"Estimated using gpt-4o-mini list pricing "
        f"(${cla.INPUT_PRICE_PER_1M_TOKENS}/1M input, ${cla.OUTPUT_PRICE_PER_1M_TOKENS}/1M output tokens) - "
        "a rough guide, not an exact bill."
    )
