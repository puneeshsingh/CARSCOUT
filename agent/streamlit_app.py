"""
Minimal demo UI for complaint_lookup_agent.py - runs the deep agent
(GPT-4o-mini + carscout_retrieval MCP tools: reliability complaints, price
comps, recalls, safety rating) against real NHTSA and Craigslist data and
renders the Think -> Act -> Observe trace plus token cost.

For the bootcamp cohort demo. Not polished, not for production use.
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import complaint_lookup_agent as cla  # noqa: E402  (must load .env first, sets up logging)
import guards  # noqa: E402

st.set_page_config(page_title="CarScout Due-Diligence Agent", layout="wide")
st.title("CarScout Due-Diligence Agent")
st.caption(
    "Runs a deepagents agent (GPT-4o-mini + 4 carscout_retrieval MCP tools) against real NHTSA "
    "and Craigslist data to check reliability complaints, price fairness, recall history, and "
    "crash-safety rating. The agent is instructed to answer only from tool results, never from "
    "its own training knowledge."
)

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

CONDITION_OPTIONS = ["Not sure / skip", "new", "like new", "excellent", "good", "fair", "salvage"]

col1, col2 = st.columns([2, 1])
with col1:
    selected_label = st.selectbox("Vehicle", VEHICLE_LABELS)
selected_make, selected_model, selected_year = VEHICLE_SHORTLIST[VEHICLE_LABELS.index(selected_label)]
with col2:
    year = st.number_input("Year", min_value=1990, max_value=2030, value=selected_year, step=1)

make, vehicle_model = selected_make, selected_model

col3, col4, col5 = st.columns(3)
with col3:
    asking_price = st.number_input("Asking price ($)", min_value=0, value=15000, step=500)
with col4:
    odometer = st.number_input("Odometer (miles)", min_value=0, value=45000, step=1000)
with col5:
    condition_choice = st.selectbox("Condition", CONDITION_OPTIONS)
condition = None if condition_choice == "Not sure / skip" else condition_choice

symptom = st.text_input(
    "Symptom / question",
    value="Is engine stalling a known issue for this vehicle?",
)

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

    # Gate 3: the agent itself - only reached if both gates above passed.
    with st.spinner("Agent is running (Think -> Act -> Observe)..."):
        trace = cla.run_with_trace(
            make_clean, model_clean, int(year), symptom_clean,
            asking_price=float(asking_price), odometer=int(odometer), condition=condition,
        )

    if trace["hit_step_cap"]:
        st.warning(f"Agent hit the {cla.MAX_STEPS}-step cap without a confident final answer (failed closed).")

    st.subheader("Final answer")
    if trace["is_no_confident_match"]:
        # Amber, not green - a no-confident-match result is not a "clean
        # bill of health" and shouldn't be styled like a reassuring success.
        st.warning(trace["final_answer"])
    else:
        st.success(trace["final_answer"])

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
