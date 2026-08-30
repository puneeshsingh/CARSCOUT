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
import report_pdf  # noqa: E402
import vin_decode  # noqa: E402
from config import VIN_DEMO_LISTINGS  # noqa: E402  (config.py is lightweight - no pandas/pinecone import cost)

TILE_TITLES = {
    "reliability": "Reliability", "price": "Price fairness",
    "recalls": "Recall history", "safety": "Safety rating",
}
# Deterministic red/amber/green from classify_tiles() in
# complaint_lookup_agent.py, not chosen by the model - st.error/warning/
# success are Streamlit's own semantic red/amber/green containers, so this
# reuses native theming instead of hand-rolled colored HTML.
TILE_RENDERERS = {"red": st.error, "amber": st.warning, "green": st.success}


def _overall_color(tiles: dict) -> str:
    """Worst-signal-wins severity, so a report can't lead with a red
    "known reliability issue" tile and then wrap the same finding in a
    green success box elsewhere on the same card."""
    colors = {t.get("color", "amber") for t in tiles.values()}
    if "red" in colors:
        return "red"
    if "amber" in colors:
        return "amber"
    return "green"


def _report_header(year, make, vehicle_model, asking_price, odometer, symptom) -> str:
    return (
        f"# Due-diligence report: {year} {make} {vehicle_model}\n\n"
        f"Asking price: ${asking_price:,.0f} | Odometer: {odometer:,} mi | Symptom: {symptom}\n\n---\n\n"
    )

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

# Curated real listings (real VIN, real price/odometer/condition from the
# same Craigslist dataset used for price comps) - one per shortlist vehicle.
# Selecting a VIN fills in the whole listing; only the symptom stays
# free-text, since that's the buyer's own concern, not something a listing
# states. See src/config.py for how these were sourced and verified.
VIN_LABELS = [
    f"{v['vin']} - {v['year']} {v['make']} {v['model']}"
    for v in VIN_DEMO_LISTINGS
]
LISTING_BY_VIN_LABEL = dict(zip(VIN_LABELS, VIN_DEMO_LISTINGS))

# A distinct flat color per vehicle for the placeholder car icon - not real
# photos (avoids any licensing question over manufacturer images), just a
# simple visual anchor so each listing reads as a distinct "card."
VEHICLE_ICON_COLORS = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E", "#BA7517", "#7F77DD"]


def _car_icon_svg(color: str) -> str:
    return f"""<svg viewBox="0 0 64 40" width="72" height="45" role="img" aria-label="Vehicle icon">
<path d="M8 26 L13 14 Q15 10 20 10 L44 10 Q49 10 51 14 L56 26 L56 32 L50 32 L50 26 L14 26 L14 32 L8 32 Z"
fill="{color}" stroke="{color}" stroke-width="1"/>
<circle cx="18" cy="32" r="5" fill="#4a4a4a"/>
<circle cx="46" cy="32" r="5" fill="#4a4a4a"/>
</svg>"""


DEFAULT_SYMPTOM = "Is engine stalling a known issue for this vehicle?"

# Form fields are session_state-backed (rather than plain `value=` literals)
# so a "Use this search" click in the Recent Searches sidebar can populate
# them before the form widgets are created below.
st.session_state.setdefault("form_vin_label", VIN_LABELS[0])
st.session_state.setdefault("form_symptom", DEFAULT_SYMPTOM)
st.session_state.setdefault("user_name", "")

# Best-effort (make, model) -> VIN label lookup, used by "Use this search" -
# reused entries snap to the closest curated listing for that vehicle rather
# than needing an exact historical match (a pre-VIN saved search, for
# instance, won't have one).
VIN_LABEL_BY_MAKE_MODEL = {
    (v["make"], v["model"]): label for label, v in LISTING_BY_VIN_LABEL.items()
}

with st.sidebar:
    st.text_input(
        "Your name",
        key="user_name",
        placeholder="e.g. Alex",
        help="Not a real login - just labels your searches so \"Your evaluated listings\" shows your own history, not everyone's.",
    )
    user_name = st.session_state["user_name"].strip() or None
    if user_name:
        st.caption("Persists across restarts (SQLite locally, Postgres when deployed) - scoped to your name above.")
    else:
        st.caption("Enter your name to save and see your own search history under \"Your evaluated listings\".")

st.title("CarScout Due-Diligence Agent")
st.caption(
    "Runs a deepagents agent (GPT-4o-mini + 4 carscout_retrieval MCP tools) against real NHTSA "
    "and Craigslist data to check reliability complaints, price fairness, recall history, and "
    "crash-safety rating. The agent is instructed to answer only from tool results, never from "
    "its own training knowledge."
)

tab_run, tab_compare = st.tabs(["Run a new check", "Your evaluated listings"])

with tab_run:
    col_icon, col_form = st.columns([1, 5])
    vehicle_index = VIN_LABELS.index(st.session_state["form_vin_label"])

    with col_icon:
        st.markdown(_car_icon_svg(VEHICLE_ICON_COLORS[vehicle_index]), unsafe_allow_html=True)
    with col_form:
        selected_label = st.selectbox("Choose a listing (VIN)", VIN_LABELS, key="form_vin_label")
        listing = LISTING_BY_VIN_LABEL[selected_label]

    make, vehicle_model, year = listing["make"], listing["model"], listing["year"]
    asking_price, odometer, condition = listing["price"], listing["odometer"], listing["condition"]
    condition_text = condition or "condition not stated"

    with st.spinner("Decoding VIN..."):
        decoded = vin_decode.decode_vin(listing["vin"])
    if decoded["status"] == "ok":
        st.caption(
            f"Decoded live via NHTSA: {decoded['year']} {decoded['make']} {decoded['model']} - "
            f"${asking_price:,} - {odometer:,} mi - {condition_text}."
        )
    else:
        st.caption(
            f"NHTSA decode unavailable right now - using the listing's own data: "
            f"{year} {make} {vehicle_model} - ${asking_price:,} - {odometer:,} mi - {condition_text}."
        )

    symptom = st.text_input("Symptom / question", key="form_symptom")

    PHASE_LABELS = {"think": "THINK", "act": "ACT", "observe": "OBSERVE"}

    if st.button("Run agent", type="primary"):
        make_clean = make.strip()
        model_clean = vehicle_model.strip()
        symptom_clean = symptom.strip()

        # Vehicle/price/odometer/condition all come from the curated listing,
        # not free text, so only the symptom needs validating here.
        if not symptom_clean:
            st.error("Please describe a symptom or ask a question before running the agent.")
            st.stop()

        # Gate 1: moderation - must run before the relevance check and before
        # the agent ever sees this input. Blocks on flagged=True or on API
        # error (fail closed rather than letting unmoderated input through).
        with st.spinner("Checking input..."):
            moderation = guards.check_moderation(symptom_clean)

        if moderation["status"] == "flagged":
            st.error("This input can't be processed - please describe a vehicle issue.")
            st.stop()
        elif moderation["status"] == "error":
            st.error("Something went wrong checking your input - please try again.")
            st.stop()

        # Gate 2: symptom relevance - only reached if moderation passed.
        # Blocks off-topic input ("hello", "test") before an agent/tool call.
        with st.spinner("Checking input..."):
            relevance = guards.check_relevance(symptom_clean)

        if relevance["status"] == "irrelevant":
            st.error("Please describe the actual issue - e.g. 'engine stalling at low speeds' rather than just 'vehicle' or 'problem'.")
            st.stop()
        elif relevance["status"] == "error":
            st.error("Something went wrong checking your input - please try again.")
            st.stop()

        # Gate 3: injection detection - non-blocking (informational only),
        # kept as its own narrowly-scoped classifier call rather than folded
        # into the agent's report-writing completion, which mixed the two
        # tasks and false-flagged ~2 in 5 wholly benign inputs (see
        # evals/taxonomy.md). "error" is treated the same as "clean" - never
        # blocks the run.
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
            # step-cap failure - matches the "stable, high-confidence facts
            # only" memory write policy.
            memory_store.save_search(
                memory_engine, make_clean, model_clean, int(year),
                float(asking_price), int(odometer), condition, symptom_clean,
                final_answer, user_name=user_name, tiles=trace["tiles"],
            )

        st.subheader("At a glance")
        tile_cols = st.columns(4)
        for col, (signal, title) in zip(tile_cols, TILE_TITLES.items()):
            tile = trace["tiles"].get(signal, {"color": "amber", "headline": "No data"})
            with col:
                TILE_RENDERERS[tile["color"]](f"**{title}**\n\n{tile['headline']}")

        st.subheader("Final answer")
        TILE_RENDERERS[_overall_color(trace["tiles"])](_escape_for_markdown(final_answer))

        report_header = _report_header(year, make, vehicle_model, asking_price, odometer, symptom_clean)
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "Download full report (Markdown)",
                data=report_header + final_answer,
                file_name=f"carscout_{make}_{vehicle_model}_{year}.md".replace(" ", "_"),
                mime="text/markdown",
            )
        with dl_col2:
            pdf_bytes = report_pdf.build_report_pdf(
                f"{year} {make} {vehicle_model}",
                f"${asking_price:,.0f} - {odometer:,} mi - {symptom_clean}",
                trace["tiles"], final_answer,
            )
            st.download_button(
                "Download full report (PDF)",
                data=pdf_bytes,
                file_name=f"carscout_{make}_{vehicle_model}_{year}.pdf".replace(" ", "_"),
                mime="application/pdf",
            )

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

with tab_compare:
    recent = memory_store.get_recent_searches(memory_engine, user_name=user_name)
    if not user_name:
        st.info("Enter your name in the sidebar to save and compare your evaluated listings here.")
    elif not recent:
        st.write("No searches yet - run a check in the other tab first.")
    else:
        st.caption(f"{len(recent)} evaluated listing(s) for {user_name}, most recent first.")
        compare_cols = st.columns(3)
        for i, entry in enumerate(recent):
            tiles = entry.tiles()
            color = _overall_color(tiles) if tiles else "amber"
            with compare_cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{entry.year} {entry.make} {entry.model}**")
                    st.caption(
                        f"{entry.symptom} · ${entry.asking_price:,.0f} / {entry.odometer:,} mi · "
                        f"{_format_pacific(entry.created_at)}"
                    )
                    if tiles:
                        for signal, title in TILE_TITLES.items():
                            tile = tiles.get(signal, {"color": "amber", "headline": "No data"})
                            TILE_RENDERERS[tile["color"]](f"{title}: {tile['headline']}")
                    else:
                        # Saved before tile classification existed - no
                        # per-signal colors to show, just the plain text.
                        st.caption("No at-a-glance summary saved for this older search.")
                    with st.expander("View full report"):
                        # st.text (not st.write/st.markdown): never
                        # interprets markdown, so a preview built from any
                        # saved row (old or new format) always renders the
                        # same way, card to card.
                        st.text(memory_store.build_preview(entry.full_answer))
                        st.markdown(_escape_for_markdown(entry.full_answer))
                    # build_report_pdf() falls back to "No data" per signal
                    # when tiles is {} (older, pre-tile-classification rows),
                    # so PDF download works for every saved search either way.
                    pdf_bytes = report_pdf.build_report_pdf(
                        f"{entry.year} {entry.make} {entry.model}",
                        f"${entry.asking_price:,.0f} - {entry.odometer:,} mi - {entry.symptom}",
                        tiles, entry.full_answer,
                    )
                    st.download_button(
                        "Download PDF", data=pdf_bytes,
                        file_name=f"carscout_{entry.make}_{entry.model}_{entry.year}.pdf".replace(" ", "_"),
                        mime="application/pdf", key=f"pdf_{entry.id}",
                    )
                    if st.button("Use this search", key=f"reuse_{entry.id}"):
                        label = VIN_LABEL_BY_MAKE_MODEL.get((entry.make, entry.model), VIN_LABELS[0])
                        st.session_state["form_vin_label"] = label
                        st.session_state["form_symptom"] = entry.symptom
                        st.rerun()
