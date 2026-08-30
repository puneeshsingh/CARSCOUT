"""
Minimal demo UI for complaint_lookup_agent.py - runs the deep agent
(GPT-4o-mini + carscout_retrieval MCP tools: reliability complaints, price
comps, recalls, safety rating) against real NHTSA and Craigslist data and
renders the Think -> Act -> Observe trace plus token cost.

For the bootcamp cohort demo. Not polished, not for production use.
"""

import base64
import re
import sys
from datetime import timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps

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
# One Material icon per signal (Streamlit's built-in `:material/name:`
# syntax) so each tile reads as its own category at a glance, not just a
# colored box of text.
TILE_ICONS = {
    "reliability": ":material/build:",
    "price": ":material/sell:",
    "recalls": ":material/campaign:",
    "safety": ":material/shield:",
}
# Points per tile color, used to rank evaluated listings best-first in the
# comparison grid.
_RANK_POINTS = {"green": 2, "amber": 1, "red": 0}
MAX_RANK_SCORE = len(TILE_TITLES) * max(_RANK_POINTS.values())

_STAR_HEADLINE_RE = re.compile(r"^(\d)-star overall rating$")


def _starred_headline(headline: str) -> str:
    """Turns "5-star overall rating" into "★★★★★ (5-star overall rating)"
    - real stars, not just a number, for the one tile where a star rating
    is the natural unit. Falls back to the plain headline untouched for
    anything that doesn't match (no rating on file, unparsed, etc.)."""
    match = _STAR_HEADLINE_RE.match(headline)
    if not match:
        return headline
    stars = int(match.group(1))
    return f"{'★' * stars}{'☆' * (5 - stars)}  ({headline})"


# Native st.error/warning/success (still used for the recommend/avoid rank
# badge) are too padded for a per-signal tile - four of them stacked (or
# even four flat full-width chips) read as stretched-out bars, not a quick
# scannable summary. These render the same severity color as a 2x2 grid of
# small, equal-size, square-ish tiles instead - plain Unicode icons rather
# than Streamlit's `:material/` ligature syntax, since that only resolves
# inside native st.* calls, not raw HTML. Every value interpolated in here
# comes from classify_tiles()'s fixed headline templates (TILE_TITLES keys,
# signal counts, star ratings) - never free-text user input - so this is
# safe to render as raw HTML.
_CHIP_COLORS = {
    "red": {"bg": "rgba(224, 49, 49, 0.12)", "accent": "#e03131"},
    "amber": {"bg": "rgba(232, 131, 12, 0.14)", "accent": "#e8830c"},
    "green": {"bg": "rgba(43, 138, 62, 0.12)", "accent": "#2b8a3e"},
}
# A whole comparison-grid card gets tinted with this (see the recommend/
# avoid logic below) - a noticeably stronger tint than a small chip needs,
# since it's meant to read as "this card is green/red" at a glance, not
# just a faint hint.
_CARD_TINTS = {
    "red": {"bg": "rgba(224, 49, 49, 0.28)", "accent": "#e03131"},
    "green": {"bg": "rgba(43, 138, 62, 0.28)", "accent": "#2b8a3e"},
}
SIGNAL_EMOJI = {"reliability": "🔧", "price": "🏷️", "recalls": "📢", "safety": "🛡️"}


def _signal_squares_html(tiles: dict) -> str:
    """4 signal tiles for one card, as a 2x2 grid of fixed-size squares in a
    single HTML block - a CSS grid is what actually guarantees identical
    cell sizes regardless of headline length, which four separate Streamlit
    widgets never would. Headline text that's too long to fit is clamped to
    3 lines with an ellipsis; the full text is still available as a native
    hover tooltip and, always, in "View full report" below.

    Deliberately opaque white, not a tint of the tile's own severity color:
    the whole card can also be tinted green/red (see the recommend/avoid
    logic below), and a light-green tile on a light-green card background
    was reading as no tile at all. The severity color still shows, just as
    a solid border instead of a fill - readable against any card color."""
    cells = []
    for signal, title in TILE_TITLES.items():
        tile = tiles.get(signal, {"color": "amber", "headline": "No data"})
        headline = _starred_headline(tile["headline"]) if signal == "safety" else tile["headline"]
        style = _CHIP_COLORS[tile["color"]]
        cells.append(
            f'<div title="{title}: {headline}" style="display:flex;flex-direction:column;'
            f'gap:3px;height:104px;padding:8px 10px;border-radius:8px;'
            f'background:rgba(255,255,255,0.92);border:2px solid {style["accent"]};overflow:hidden;">'
            f'<div style="display:flex;align-items:center;gap:5px;">'
            f'<span style="font-size:16px;line-height:1;">{SIGNAL_EMOJI[signal]}</span>'
            f'<span style="font-size:12.5px;font-weight:700;line-height:1.2;color:#1a1a2e;">{title}</span>'
            f"</div>"
            f'<span style="font-size:12px;line-height:1.35;color:#333333;display:-webkit-box;'
            f'-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;">{headline}</span>'
            f"</div>"
        )
    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">'
        + "".join(cells)
        + "</div>"
    )


def _rank_score(tiles: dict) -> int:
    """Higher = a better-looking listing overall - green tiles score more
    than amber, amber more than red. Used only to sort the comparison grid
    best-first; the raw score itself is never shown to the user."""
    return sum(_RANK_POINTS.get(t.get("color"), 0) for t in tiles.values())

# Live status-box labels, keyed by tool name, shown as each tool is called
# (see run_with_progress's on_event callback below) - real-time progress
# instead of one blank spinner for the whole ~15-20s run.
STATUS_LABELS = {
    "search_complaints": "Searching reliability complaints...",
    "check_price_estimate": "Checking price fairness...",
    "check_recalls": "Checking recall history...",
    "check_safety_rating": "Checking safety rating...",
}


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

LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "branding" / "carscout_logo.jpg"

st.set_page_config(page_title="CarScout", page_icon=str(LOGO_PATH), layout="wide")


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

# A distinct flat color per vehicle for the placeholder car icon - fallback
# only, for a vehicle that doesn't have a real photo in assets/vehicles/
# (avoids ever showing a blank space if one goes missing).
VEHICLE_ICON_COLORS = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E", "#BA7517", "#7F77DD"]

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "vehicles"


def _vehicle_image_path(make: str, model: str) -> Path | None:
    """Real listing photos, supplied by the user (assets/vehicles/, e.g.
    hyundai_kona.jpg) - not sourced by this app, so there's no licensing
    question to resolve on our end. Falls back to the SVG icon below for
    any vehicle without one."""
    path = ASSETS_DIR / f"{make.lower()}_{model.lower()}.jpg"
    return path if path.exists() else None


@st.cache_data(show_spinner=False)
def _load_vehicle_image(path_str: str, width: int, height: int) -> Image.Image:
    """Source photos come in whatever size/aspect ratio was supplied - a
    mix of dealer-photo and phone-photo dimensions makes cards in the same
    grid look inconsistent if shown at native size. ImageOps.fit crops to
    the target aspect ratio (centered) then resizes, so every vehicle
    renders at the exact same pixel size regardless of its source image."""
    img = Image.open(path_str).convert("RGB")
    return ImageOps.fit(img, (width, height), method=Image.LANCZOS)


# Target sizes for the two places a vehicle photo appears - the single
# "Run a new check" preview gets more room than a grid thumbnail does.
FEATURED_IMAGE_SIZE = (480, 320)
GRID_IMAGE_SIZE = (320, 220)


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
st.session_state.setdefault("form_vin_label", None)
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

st.logo(str(LOGO_PATH), size="large", icon_image=str(LOGO_PATH))


@st.cache_data
def _logo_b64() -> str:
    return base64.b64encode(LOGO_PATH.read_bytes()).decode()


# st.columns() sizes columns proportionally to the full page width, so any
# ratio still leaves a huge gap around a small fixed-width image on a wide
# screen - a plain flex row with a fixed gap keeps the logo snug against the
# title regardless of page width.
st.markdown(
    f"""<div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
<img src="data:image/jpeg;base64,{_logo_b64()}" width="72" style="border-radius:6px; flex-shrink:0;">
<div>
<h1 style="margin:0; padding:0; line-height:1.2;">CarScout — A Deep Agent for Used-Car Due Diligence</h1>
</div>
</div>""",
    unsafe_allow_html=True,
)
st.caption(
    "Checks reliability, price fairness, recalls, and safety for a used-car listing - grounded "
    "in real NHTSA and Craigslist data, never guessed from training knowledge."
)

tab_run, tab_compare = st.tabs(["Run a new check", "Your evaluated listings"])

# tab_compare renders BEFORE tab_run in the script (their with-block order
# doesn't affect which tab appears first in the UI - that's set by the
# order passed to st.tabs() above). tab_run's body calls st.stop() when no
# listing is selected yet, which halts the *entire* script from that point
# on - so tab_compare has to already be fully rendered before that can
# happen, or it would never show anything.
with tab_compare:
    recent = memory_store.get_recent_searches(memory_engine, user_name=user_name)
    if not user_name:
        st.info("Enter your name in the sidebar to save and compare your evaluated listings here.")
    elif not recent:
        st.write("No searches yet - run a check in the other tab first.")
    else:
        # Best-first, not just most-recent-first, so the comparison actually
        # ranks the listings - a green-heavy report leads, red-heavy trails.
        # Entries saved before tile classification existed (score -1) always
        # sort last, since there's nothing to rank them on.
        ranked = sorted(
            recent,
            key=lambda e: _rank_score(e.tiles()) if e.tiles() else -1,
            reverse=True,
        )
        rank_eligible = sum(1 for e in ranked if e.tiles())
        st.caption(
            f"{len(recent)} evaluated listing(s) for {user_name}, best overall result first."
        )
        # Score per entry, in the same best-first order as `ranked` - reused
        # below both for the badge logic and the progress-bar visual, so the
        # two never disagree.
        scores = [_rank_score(e.tiles()) if e.tiles() else None for e in ranked]
        best_score = scores[0] if scores and scores[0] is not None else None
        compare_cols = st.columns(3)
        for i, entry in enumerate(ranked):
            tiles = entry.tiles()
            score = scores[i]
            # The worst-signal-wins verdict badge (below) is deliberately
            # conservative - reliability is never "green" - so on its own
            # almost every listing reads as the same amber "worth a closer
            # look", which doesn't tell listings apart. This is what should
            # actually say "choose this one" vs "maybe not this one": the
            # whole card is tinted, not just its border - green for the top
            # score, red for the strictly-lowest score (only when it's
            # really behind the best, not just tied). Anyone in between
            # stays the plain default card - correctly reading as "no
            # strong opinion".
            is_lowest = (
                i == len(ranked) - 1 and score is not None
                and best_score is not None and score < best_score
            )
            card_style = None
            if tiles and rank_eligible > 1:
                if i == 0:
                    card_style = _CARD_TINTS["green"]
                elif is_lowest:
                    card_style = _CARD_TINTS["red"]
            with compare_cols[i % 3]:
                if card_style:
                    # st.container(border=True) only draws a neutral default
                    # border with no color/background parameter - key="..."
                    # tags the wrapper with a matching "st-key-..." CSS class
                    # (Streamlit's own supported hook for styling one
                    # specific container), which this scoped <style> block
                    # then recolors and tints.
                    st.markdown(
                        f"<style>.st-key-card_{entry.id} {{ "
                        f"background-color: {card_style['bg']} !important; "
                        f"border-color: {card_style['accent']} !important; "
                        f"border-width: 2px !important; }}</style>",
                        unsafe_allow_html=True,
                    )
                with st.container(border=True, key=f"card_{entry.id}"):
                    image_path = _vehicle_image_path(entry.make, entry.model)
                    if image_path:
                        st.image(_load_vehicle_image(str(image_path), *GRID_IMAGE_SIZE), use_container_width=True)
                    if tiles and rank_eligible > 1:
                        # Always the same widget (st.badge) regardless of
                        # rank, just different text/color/icon - using a
                        # taller st.success/st.error banner only for the
                        # extremes made those two cards visibly taller than
                        # the rest of the grid; the card tint above is what
                        # signals "choose this" / "avoid this" now, so the
                        # badge itself can stay uniform.
                        if i == 0:
                            st.badge("Recommended - best pick", icon=":material/military_tech:", color="green")
                        elif is_lowest:
                            st.badge("Lowest-ranked - use caution", icon=":material/thumb_down:", color="red")
                        else:
                            st.badge(f"#{i + 1} of your evaluated listings", icon=":material/star:", color="blue")
                    if tiles:
                        verdict_badge = {
                            "red": ("Needs caution", ":material/warning:", "red"),
                            "amber": ("Worth a closer look", ":material/visibility:", "orange"),
                            "green": ("Looks solid", ":material/thumb_up:", "green"),
                        }[_overall_color(tiles)]
                        st.badge(verdict_badge[0], icon=verdict_badge[1], color=verdict_badge[2])
                    if score is not None:
                        # A same-width bar per card is what actually makes
                        # "which one's better" jump out at a glance across
                        # the grid - reading four individual tile colors per
                        # card and comparing them mentally, card to card,
                        # doesn't.
                        st.progress(
                            score / MAX_RANK_SCORE,
                            text=f"{score}/{MAX_RANK_SCORE} signals positive",
                        )
                    st.markdown(f"**{entry.year} {entry.make} {entry.model}**")
                    # st.caption's muted secondary-gray color was hard to
                    # read against the card's own colored tint - st.markdown
                    # renders the same way st.caption already did (both are
                    # Streamlit's markdown renderer; no unsafe_allow_html
                    # here, so entry.symptom can't inject raw HTML either
                    # way), just at normal-contrast body-text size/color.
                    st.markdown(
                        f"{entry.symptom} · ${entry.asking_price:,.0f} / {entry.odometer:,} mi · "
                        f"{_format_pacific(entry.created_at)}"
                    )
                    if tiles:
                        st.markdown(_signal_squares_html(tiles), unsafe_allow_html=True)
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
                        type="primary", icon=":material/download:", use_container_width=True,
                    )
                    if st.button(
                        "Use this search", key=f"reuse_{entry.id}",
                        icon=":material/replay:", use_container_width=True,
                    ):
                        label = VIN_LABEL_BY_MAKE_MODEL.get((entry.make, entry.model), VIN_LABELS[0])
                        st.session_state["form_vin_label"] = label
                        st.session_state["form_symptom"] = entry.symptom
                        st.rerun()

with tab_run:
    selected_label = st.selectbox(
        "Choose a listing (VIN)", VIN_LABELS, index=None,
        placeholder="Select a vehicle listing to preview...", key="form_vin_label",
    )

    if selected_label is None:
        st.info("Choose a listing above to preview it and run a due-diligence check.")
        st.stop()

    listing = LISTING_BY_VIN_LABEL[selected_label]
    vehicle_index = VIN_LABELS.index(selected_label)

    make, vehicle_model, year = listing["make"], listing["model"], listing["year"]
    asking_price, odometer, condition = listing["price"], listing["odometer"], listing["condition"]
    condition_text = condition or "not stated"

    col_icon, col_form = st.columns([2, 5])
    with col_icon:
        image_path = _vehicle_image_path(make, vehicle_model)
        if image_path:
            st.image(_load_vehicle_image(str(image_path), *FEATURED_IMAGE_SIZE))
        else:
            st.markdown(_car_icon_svg(VEHICLE_ICON_COLORS[vehicle_index]), unsafe_allow_html=True)

    with col_form:
        with st.spinner("Decoding VIN..."):
            decoded = vin_decode.decode_vin(listing["vin"])
        # Two genuinely different sources, kept visually distinct so
        # neither implies it vouches for the other's data: NHTSA's vPIC
        # API only ever decodes the vehicle itself (make/model/year) from
        # the VIN - it has no idea what a listing is asking for it. Price/
        # mileage/condition always come from the curated listing data,
        # whether or not the live decode call succeeds.
        if decoded["status"] == "ok":
            st.badge(
                f"VIN decoded live via NHTSA: {decoded['year']} {decoded['make']} {decoded['model']}",
                icon=":material/verified:", color="green",
            )
        else:
            st.badge("NHTSA decode unavailable - showing the listing's own vehicle data", icon=":material/error:", color="gray")
        listing_col1, listing_col2, listing_col3 = st.columns(3)
        listing_col1.metric("Asking price", f"${asking_price:,.0f}")
        listing_col2.metric("Odometer", f"{odometer:,} mi")
        listing_col3.metric("Condition", condition_text.title())

    symptom = st.text_input("Symptom / question", key="form_symptom")

    PHASE_LABELS = {"think": "THINK", "act": "ACT", "observe": "OBSERVE"}

    if st.button("Run agent", type="primary"):
        make_clean = make.strip()
        model_clean = vehicle_model.strip()
        symptom_clean = symptom.strip()

        # Every search should be tied to someone, not saved anonymously -
        # checked before anything else runs, same as the symptom check
        # below. (tab_compare already finished rendering earlier in this
        # same script run, so st.stop() here is safe - it only cuts off the
        # rest of this tab, not the comparison grid.)
        if not user_name:
            st.error("Please enter your name in the sidebar before running a check.")
            st.stop()

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

        # Gate 4: the agent itself - only reached if both blocking gates
        # passed. run_with_progress (not run_with_trace) so the status box
        # below updates live as each tool is called, instead of a single
        # blank spinner for the whole ~15-20s run.
        status_box = st.status("Starting due-diligence checks...", expanded=True)

        def _on_event(phase, payload):
            if phase == "act":
                status_box.update(label=STATUS_LABELS.get(payload["tool"], f"Calling {payload['tool']}..."))
            elif phase == "capped":
                status_box.update(label="Hit the step cap without a confident answer.", state="error")

        trace = cla.run_with_progress(
            make_clean, model_clean, int(year), symptom_clean,
            asking_price=float(asking_price), odometer=int(odometer), condition=condition,
            on_event=_on_event,
        )
        if not trace["hit_step_cap"]:
            status_box.update(label="Due-diligence checks complete.", state="complete", expanded=False)

        final_answer = trace["final_answer"]
        if injection_check["status"] == "flagged":
            final_answer = guards.INJECTION_NOTE + final_answer

        if trace["hit_step_cap"]:
            st.session_state["last_result"] = {"hit_step_cap": True}
        else:
            # Write gate: only save runs that produced a real answer, not a
            # step-cap failure - matches the "stable, high-confidence facts
            # only" memory write policy.
            memory_store.save_search(
                memory_engine, make_clean, model_clean, int(year),
                float(asking_price), int(odometer), condition, symptom_clean,
                final_answer, user_name=user_name, tiles=trace["tiles"],
            )
            st.session_state["last_result"] = {
                "hit_step_cap": False,
                "make": make, "vehicle_model": vehicle_model, "year": year,
                "asking_price": asking_price, "odometer": odometer, "symptom": symptom_clean,
                "final_answer": final_answer, "tiles": trace["tiles"], "steps": trace["steps"],
            }

        # A plain rerun (not just falling through to render below) so
        # tab_compare - which runs earlier in this same script - reads the
        # freshly-saved row on the next pass instead of the stale one it
        # already rendered before this button's handler even ran. The
        # result itself is rendered from session_state below, outside this
        # button block, specifically so it survives that rerun instead of
        # vanishing (st.button() only returns True on the one rerun the
        # actual click triggers).
        st.rerun()

    result = st.session_state.get("last_result")
    if result:
        if result["hit_step_cap"]:
            st.warning(f"Agent hit the {cla.MAX_STEPS}-step cap without a confident final answer (failed closed).")
        else:
            st.subheader("At a glance")
            verdict_badge = {
                "red": ("Needs caution", ":material/warning:", "red"),
                "amber": ("Worth a closer look", ":material/visibility:", "orange"),
                "green": ("Looks solid", ":material/thumb_up:", "green"),
            }[_overall_color(result["tiles"])]
            st.badge(verdict_badge[0], icon=verdict_badge[1], color=verdict_badge[2])
            tile_cols = st.columns(4)
            for col, (signal, title) in zip(tile_cols, TILE_TITLES.items()):
                tile = result["tiles"].get(signal, {"color": "amber", "headline": "No data"})
                headline = _starred_headline(tile["headline"]) if signal == "safety" else tile["headline"]
                with col:
                    with st.container(border=True, height=140):
                        TILE_RENDERERS[tile["color"]](f"**{title}**\n\n{headline}", icon=TILE_ICONS[signal])

            st.subheader("Final answer")
            TILE_RENDERERS[_overall_color(result["tiles"])](_escape_for_markdown(result["final_answer"]))

            pdf_bytes = report_pdf.build_report_pdf(
                f"{result['year']} {result['make']} {result['vehicle_model']}",
                f"${result['asking_price']:,.0f} - {result['odometer']:,} mi - {result['symptom']}",
                result["tiles"], result["final_answer"],
            )
            st.download_button(
                "Download full report (PDF)",
                data=pdf_bytes,
                file_name=f"carscout_{result['make']}_{result['vehicle_model']}_{result['year']}.pdf".replace(" ", "_"),
                mime="application/pdf",
                type="primary", icon=":material/download:",
            )

            with st.expander("Show technical trace (Think → Act → Observe)", expanded=False):
                if not result["steps"]:
                    st.write("No steps recorded.")
                for i, step in enumerate(result["steps"], start=1):
                    label = PHASE_LABELS.get(step["phase"], step["phase"].upper())
                    st.caption(f"{i}. {label}")
                    if step["phase"] == "act":
                        st.code(f"{step['tool']}({step['args']})", language="python")
                    else:
                        st.write(step["text"])
