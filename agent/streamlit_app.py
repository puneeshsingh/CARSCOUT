"""
Minimal demo UI for complaint_lookup_agent.py - runs the deep agent
(GPT-4o-mini + carscout_retrieval MCP tools: reliability complaints, price
comps, recalls, safety rating) against real NHTSA and Craigslist data and
renders the Think -> Act -> Observe trace plus token cost.

For the bootcamp cohort demo. Not polished, not for production use.
"""

import base64
import html
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps

load_dotenv()

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import comparison_chat  # noqa: E402
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


# Comparison-grid cards render on a dark gradient (DARK_CARD_CSS below), not
# white - a per-signal tile tinted with its own severity color at FULL
# opacity (a past version of this) reads fine on white but camouflages
# white text on a dark background just as badly as it did on a green-tinted
# one, since these accent colors are themselves light/pastel. A border-only
# accent (an even earlier version of this fix) went too far the other way -
# too subtle to read as "colored" at a glance. `fill` is the same accent at
# low opacity over the dark card background - light enough that the base
# stays dark (white text keeps its contrast), saturated enough that the
# tile genuinely reads as red/amber/green, not just a thin outline. Plain
# Unicode icons rather than Streamlit's `:material/` ligature syntax, since
# that only resolves inside native st.* calls, not raw HTML. Every value
# interpolated in here comes from classify_tiles()'s fixed headline
# templates (TILE_TITLES keys, signal counts, star ratings) - never
# free-text user input - so this is safe to render as raw HTML.
_CHIP_COLORS = {
    "red": {"accent": "#f87171", "glow": "rgba(248,113,113,0.65)", "fill": "rgba(248,113,113,0.28)"},
    "amber": {"accent": "#fbbf24", "glow": "rgba(251,191,36,0.65)", "fill": "rgba(251,191,36,0.26)"},
    "green": {"accent": "#4ade80", "glow": "rgba(74,222,128,0.7)", "fill": "rgba(74,222,128,0.26)"},
}
SIGNAL_EMOJI = {"reliability": "🔧", "price": "🏷️", "recalls": "📢", "safety": "🛡️"}


def _signal_squares_html(tiles: dict) -> str:
    """4 signal tiles for one card, as a 2x2 grid of fixed-size squares in a
    single HTML block - a CSS grid is what actually guarantees identical
    cell sizes regardless of headline length, which four separate Streamlit
    widgets never would. Headline text that's too long to fit is clamped to
    3 lines with an ellipsis; the full text is still available as a native
    hover tooltip and, always, in "View full report" below."""
    cells = []
    for signal, title in TILE_TITLES.items():
        tile = tiles.get(signal, {"color": "amber", "headline": "No data"})
        headline = _starred_headline(tile["headline"]) if signal == "safety" else tile["headline"]
        chip = _CHIP_COLORS[tile["color"]]
        # title/headline are always fixed templates or counts/stars from
        # classify_tiles(), never free text - escaping isn't fixing a live
        # bug, just cheap insurance against that invariant changing later.
        safe_title, safe_headline = html.escape(title), html.escape(headline)
        cells.append(
            f'<div title="{safe_title}: {safe_headline}" style="display:flex;flex-direction:column;'
            f'gap:6px;height:96px;box-sizing:border-box;padding:14px 16px;border-radius:12px;'
            f'background:{chip["fill"]};border:1px solid {chip["accent"]};overflow:hidden;">'
            f'<div style="display:flex;align-items:center;gap:7px;">'
            f'<span style="font-size:15px;line-height:1;">{SIGNAL_EMOJI[signal]}</span>'
            f'<span style="font-size:13.5px;font-weight:700;line-height:1.2;color:#ffffff;">{safe_title}</span>'
            f"</div>"
            f'<span style="font-size:12.5px;line-height:1.4;color:rgba(255,255,255,0.88);'
            f'display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;">'
            f"{safe_headline}</span>"
            f"</div>"
        )
    # box-sizing:border-box on each cell makes height:96px the actual
    # rendered box (padding+border included), not padding added on top of
    # it - without it, the 4 tiles could render at very slightly different
    # total heights depending on whether a global box-sizing reset happens
    # to be present, instead of the identical height this grid is supposed
    # to guarantee.
    return (
        '<div class="dark-tile-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">'
        + "".join(cells)
        + "</div>"
    )


# Recommended/lowest-ranked cards get a colored border on top of the shared
# dark-card look below - gold for "choose this", red for "consider
# carefully". Everyone in between keeps the default glass border.
CARD_BORDER_RECOMMENDED = "#f5c542"
CARD_BORDER_LOWEST = "#e03131"

# One shared block, injected once (not per-card) - a CSS attribute selector
# ([class*=...]) matches every comparison-grid card's container by its
# "st-key-card_<id>" class prefix, so this doesn't need to repeat per card
# the way the border-color override below still does. Scoping every rule
# under that same selector prefix (rather than a bare element selector like
# `button` or `p`) is deliberate - an earlier version of the progress-bar/
# button styling used bare selectors and it bled into unrelated widgets
# elsewhere on the page (a button's own label text turned invisible when a
# blanket "color: white" rule caught it too).
DARK_CARD_CSS = """<style>
[class*="st-key-card_"] {
    background: linear-gradient(135deg, #0a2e35 0%, #124a52 50%, #1a6570 100%) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 20px !important;
    box-shadow: 0 0 24px rgba(20,180,160,0.18) !important;
    padding: 4px !important;
}
[class*="st-key-card_"] [data-testid="stExpander"] * {
    /* Wildcard, not a tag-by-tag list (p/span/summary) - the full report's
       markdown can render numbered lists (<ol>/<li>) for recall campaigns,
       and those weren't covered by the narrower rule, so they inherited
       Streamlit's default near-black body color and were invisible against
       the dark card. Same fix pattern already used for the chat panel's
       message content, applied here for the same reason: enumerating tags
       one at a time keeps missing the next markdown element type. */
    color: #ffffff !important;
}
.dark-badge-row { display: flex; gap: 10px; margin: 18px 18px; flex-wrap: wrap; }
.dark-badge-gold {
    background: linear-gradient(90deg,#f5c542,#e8a72c); color: #3a2a00;
    font-size: 13px; font-weight: 700; padding: 8px 16px; border-radius: 20px;
    display: inline-flex; align-items: center; gap: 7px;
}
.dark-badge-glass {
    background: rgba(255,255,255,0.15); color: #ffffff; font-size: 13px; font-weight: 600;
    padding: 8px 16px; border-radius: 20px; display: inline-flex; align-items: center; gap: 7px;
}
.dark-badge-red {
    /* Solid, not translucent glass - the lowest-ranked pick should read as
       an actual alert, not just another neutral pill the same weight as
       "#4 of your evaluated listings". Same deep red as CARD_BORDER_LOWEST
       so the badge and the card's own red border always agree. */
    background: linear-gradient(90deg,#e03131,#a61b1b); color: #ffffff;
    font-size: 13px; font-weight: 700; padding: 8px 16px; border-radius: 20px;
    display: inline-flex; align-items: center; gap: 7px;
}
.dark-title { color: #ffffff; font-size: 21px; font-weight: 700; margin: 4px 18px 6px; }
.dark-price-row {
    display: flex; gap: 12px; align-items: center; color: #ffffff; font-weight: 700;
    font-size: 15px; margin: 18px 18px 18px;
}
.dark-price-row span.dim { color: rgba(255,255,255,0.4); font-weight: 400; }
.dark-info-bar {
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18);
    border-radius: 12px; padding: 14px 18px; color: #ffffff; font-size: 14px;
    display: flex; align-items: flex-start; gap: 10px; margin: 0 18px 20px;
    /* Fixed height (not just the signal tiles below it) is what actually
       keeps the grid consistent card to card - a one-line symptom and a
       two-line symptom used to push everything below (including the tile
       grid) to a different vertical start on each card, which read as
       "the tiles are inconsistent" even though each tile's own height was
       already locked at 96px. */
    height: 52px; box-sizing: border-box;
}
.dark-info-bar span.symptom-text {
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.dark-tile-grid { margin: 0 18px 20px; }
[class*="st-key-card_"] [data-testid="stProgress"] {
    width: calc(100% - 36px) !important;
    margin: 0 18px 18px !important;
}
[class*="st-key-card_"] [data-testid="stProgress"] p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 13px !important;
}
[class*="st-key-card_"] [role="progressbar"] {
    background: rgba(255,255,255,0.15) !important;
}
[class*="st-key-card_"] [role="progressbar"] > div,
[class*="st-key-card_"] [role="progressbar"] > div > div {
    /* Fallback only - every card with tiles gets a per-card override below
       (progress_overrides) that recolors this to the same red/amber/green
       severity language as the tile borders and verdict badge. A fixed
       blue-cyan-green gradient here regardless of the actual score was the
       bug: it always looked vaguely "good" (green end) no matter how the
       listing actually scored, so the color carried no real information -
       exactly what was unclear about it. */
    background: #64748b !important;
    box-shadow: none !important;
}
[class*="st-key-card_"] div.stButton > button,
[class*="st-key-card_"] div.stDownloadButton > button {
    padding: 0.4rem 0.9rem !important;
    font-size: 13.5px !important;
    border-radius: 10px !important;
    min-height: 2.2rem !important;
}
[class*="st-key-card_"] div[data-testid="stHorizontalBlock"] {
    /* Streamlit sets this block's own width explicitly (not "auto"), so a
       plain margin shifts the box right without shrinking it - it just
       overflows past the card's edge on the right instead of centering
       inside it. width:calc(100% - 36px) forces the actual shrink the
       margin was supposed to cause. Bottom margin here (not on the card
       itself, which is kept near-zero so the image stays flush against
       the rounded corners) is what actually gives the buttons breathing
       room from the card's bottom edge - they used to sit 5px from it. */
    width: calc(100% - 36px) !important;
    margin: 0 18px 18px !important;
}
[class*="st-key-card_"] [data-testid="stExpander"] {
    /* Same explicit-width-not-auto issue as the button row above - this
       was overflowing 13px past the card's right edge for the same
       reason, just not measured until now. */
    width: calc(100% - 36px) !important;
    margin: 0 18px 14px;
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 10px !important;
}
</style>"""


def _rank_score(tiles: dict) -> int:
    """Higher = a better-looking listing overall - green tiles score more
    than amber, amber more than red. Sorts the comparison grid best-first;
    also shown to the user directly as "X/MAX signals positive" on each
    card's progress bar."""
    return sum(_RANK_POINTS.get(t.get("color"), 0) for t in tiles.values())


def _score_color(score: int) -> str:
    """Buckets a rank score into the same red/amber/green language as the
    tiles, by the same fraction the progress bar already renders as its
    fill width - so the bar's color is always describing the exact number
    printed on it ("X/MAX signals positive"), not a separate signal. Two
    different scores (5/8 vs 6/8) crossing a bucket boundary now visibly
    differ in color, not just in bar length - before this, the bar's color
    came from the worst-signal-wins verdict instead (_overall_color()),
    which collapses almost every real listing into the same amber bucket
    here (reliability is deliberately never green - see _overall_color's
    docstring), so most cards looked identical regardless of their actual
    score. That verdict is still the right signal for the badge text right
    above the bar (one red tile should read as "needs caution" no matter
    how good the other three are) - it's just the wrong signal for a bar
    whose whole job is to visualize this specific number."""
    fraction = score / MAX_RANK_SCORE
    if fraction >= 0.75:
        return "green"
    if fraction >= 0.5:
        return "amber"
    return "red"

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


def _combined_report_markdown(qa_history: list[dict]) -> str:
    """Every question ever asked about a listing, most recent first - a
    second (or third...) question about the same curated VIN used to
    silently overwrite the previous run's full report (only the
    comparison-grid CARD is meant to be one-per-listing; the underlying
    answers were never meant to be thrown away). Single-question listings
    (the common case) render exactly as before - no extra heading, so this
    is a no-op for every pre-existing saved search."""
    if len(qa_history) <= 1:
        return qa_history[0]["full_answer"] if qa_history else ""
    sections = []
    for entry in reversed(qa_history):
        created_at = entry.get("created_at")
        when = _format_pacific(datetime.fromisoformat(created_at)) if created_at else ""
        heading = f"### Q: {entry['symptom']}" + (f"  \n*{when}*" if when else "")
        sections.append(f"{heading}\n\n{entry['full_answer']}")
    return "\n\n---\n\n".join(sections)

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


@st.cache_data
def _logo_b64() -> str:
    return base64.b64encode(LOGO_PATH.read_bytes()).decode()


st.set_page_config(page_title="CarScout", page_icon=str(LOGO_PATH), layout="wide")

# A themed full-page login screen, not a real auth system - there's no
# account database and the password is never checked, only ever asked for
# because a login screen without one would look broken, not because it
# guards anything. Its only real job is the same thing the old sidebar
# "Your name" field did: label whose searches are whose so "Your evaluated
# listings" shows one person's history, not everyone's. Gated with
# st.stop() before the memory engine, VIN listings, or anything else below
# even initializes - nobody sees (or pays the DB-connect cost for) the rest
# of the app until they've entered a name.
LOGIN_PAGE_CSS = """<style>
[data-testid="stSidebar"] { display: none !important; }
.st-key-login_card {
    max-width: 440px;
    margin: 64px auto 0 !important;
    background: linear-gradient(135deg, #0a2e35 0%, #124a52 50%, #1a6570 100%) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 24px !important;
    box-shadow: 0 0 40px rgba(20,180,160,0.28) !important;
    padding: 44px 40px !important;
}
.st-key-login_card label,
.st-key-login_card p,
.st-key-login_card span,
.st-key-login_card div[data-testid="stMarkdownContainer"] * {
    color: #ffffff !important;
}
.st-key-login_card div[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.08) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.28) !important;
    border-radius: 10px !important;
}
.st-key-login_card div[data-testid="stTextInput"] input::placeholder {
    color: rgba(255,255,255,0.45) !important;
}
.st-key-login_card div[data-testid="stFormSubmitButton"] button {
    border-radius: 12px !important;
    padding: 0.6rem 0 !important;
    font-weight: 700 !important;
    margin-top: 6px !important;
}
</style>"""


def _render_login_page() -> None:
    st.markdown(LOGIN_PAGE_CSS, unsafe_allow_html=True)
    with st.container(key="login_card"):
        st.markdown(
            f'<div style="text-align:center;">'
            f'<img src="data:image/jpeg;base64,{_logo_b64()}" width="84" '
            f'style="border-radius:16px;box-shadow:0 0 20px rgba(20,180,160,0.4);">'
            f'<h1 style="margin:16px 0 4px;font-size:26px;">CarScout</h1>'
            f'<p style="margin:0 0 26px;opacity:0.75;font-size:14px;">'
            f"Sign in to check and compare used-car listings</p></div>",
            unsafe_allow_html=True,
        )
        with st.form("login_form", border=False):
            username = st.text_input("Username", placeholder="e.g. Alex")
            st.text_input("Password", type="password", placeholder="anything works")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if submitted:
            clean = username.strip()
            if not clean:
                st.error("Enter a username to continue.")
            else:
                st.session_state["user_name"] = clean
                st.session_state["logged_in"] = True
                st.rerun()
        st.caption(
            "Not real authentication - your password isn't checked or stored. It just labels your "
            'searches so "Your evaluated listings" shows your own history, not everyone\'s.'
        )


st.session_state.setdefault("logged_in", False)
if not st.session_state["logged_in"]:
    _render_login_page()
    st.stop()


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
    user_name = st.session_state["user_name"].strip() or None
    st.markdown(f"**{html.escape(user_name)}**" if user_name else "_Not logged in_")
    st.caption("Persists across restarts (SQLite locally, Postgres when deployed) - scoped to your name above.")
    if st.button("Log out", icon=":material/logout:"):
        st.session_state["logged_in"] = False
        st.rerun()

st.logo(str(LOGO_PATH), size="large", icon_image=str(LOGO_PATH))

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

# Chat over the user's own evaluated listings, as a right-docked panel the
# user can open/close (not an always-visible section at the bottom of the
# page, and not a centered modal covering most of the screen - a st.dialog
# was tried first and covered too much of the page). Dark-themed to match
# the comparison cards, even though the panel renders outside any
# [class*="st-key-card_"] container those styles are scoped to.
CHAT_WIDGET_CSS = """<style>
.st-key-toggle_comparison_chat button {
    position: fixed !important;
    top: 70px !important;
    right: 24px !important;
    left: auto !important;
    width: auto !important;
    z-index: 999 !important;
    border-radius: 24px !important;
    background: linear-gradient(135deg, #0a2e35 0%, #1a6570 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.35) !important;
    box-shadow: 0 0 16px rgba(20,180,160,0.35) !important;
    padding: 0.5rem 1.1rem !important;
}
.st-key-chat_panel {
    position: fixed !important;
    top: 118px !important;
    right: 24px !important;
    width: 400px !important;
    max-width: calc(100vw - 48px) !important;
    max-height: calc(100vh - 150px) !important;
    overflow-y: auto !important;
    z-index: 998 !important;
    background: linear-gradient(135deg, #0a2e35 0%, #124a52 50%, #1a6570 100%) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 16px !important;
    box-shadow: 0 0 24px rgba(20,180,160,0.25) !important;
    padding: 12px !important;
}
.st-key-chat_panel p,
.st-key-chat_panel h1,
.st-key-chat_panel h2,
.st-key-chat_panel h3,
.st-key-chat_panel span,
.st-key-chat_panel label {
    color: #ffffff !important;
}
/* st.chat_message()'s content wrapper can hold more than plain <p> text -
   the assistant's answers are LLM markdown and can include bullet lists,
   bold labels, etc. A user question asking "why" a recommendation was
   made pulled back a bulleted signal-by-signal summary, and every <li> in
   it was still Streamlit's default near-black text - not covered by the
   tag-by-tag rule above. Scoping a wildcard to just the message content
   (not the whole panel) catches every element markdown can produce
   without doing this one tag at a time again. */
.st-key-chat_panel [data-testid="stChatMessageContent"] * {
    color: #ffffff !important;
}
/* Right-align the user's own messages (avatar + bubble swap to the right
   edge) so the conversation reads like a normal chat thread instead of
   both sides stacking flush-left. Confirmed via DOM inspection that
   Streamlit tags each message's avatar with stChatMessageAvatarUser vs
   stChatMessageAvatarAssistant - a stable, purpose-built hook, not a
   guess based on visual position. */
.st-key-chat_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
}
.st-key-chat_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
    text-align: right !important;
}
</style>"""


def _build_chat_transcript(user_name: str, messages: list[dict]) -> str:
    """Plain-text export of a chat session for the download button below -
    the chat itself only ever lives in st.session_state (see
    _render_comparison_chat_panel's docstring), so this is the one way a
    user can keep a copy of it past the current browser session."""
    lines = [f"CarScout chat - {user_name}", f"Exported {_format_pacific(datetime.now(timezone.utc))}", ""]
    for msg in messages:
        speaker = "You" if msg["role"] == "user" else "CarScout"
        lines.append(f"{speaker}: {msg['content']}")
        if msg.get("sources"):
            lines.append("Sources: " + ", ".join(f"{s['title']} ({s['url']})" for s in msg["sources"]))
        lines.append("")
    return "\n".join(lines)


def _render_comparison_chat_panel(ranked_entries, user_name, rank_labels):
    """Chat over the user's own evaluated listings - grounded in the same
    saved tiles/summaries the comparison grid renders, not a second live-
    tool-calling agent (see comparison_chat.py). Chat history is keyed by
    user_name (session_state, not persisted to the DB) so switching the
    name in the sidebar doesn't show one person's chat under another's
    searches - the same scoping the searches themselves already use.

    rank_labels: {entry.id: "Recommended - best pick" | "#N of your
    evaluated listings" | "Lowest-ranked - consider carefully"} - the same
    dict the card badges above are built from, so the chat can answer
    "why did you recommend X" using the exact ranking the user is already
    looking at, instead of having no rank information at all."""
    with st.container(key="chat_panel"):
        # The toggle button itself (outside this panel) already relabels to
        # "✕ Close" while the panel is open - an in-panel close button here
        # too just duplicated it (two close controls stacked on top of each
        # other, both doing the same thing).
        st.markdown("**💬 CarScout**")

        chat_key = f"comparison_chat_history_{user_name}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = [{
                "role": "assistant",
                "content": (
                    f"Hi {user_name}! Ask me anything about your evaluated listings - "
                    "which one to pick, how they compare, or details on any signal."
                ),
            }]

        # Only once there's an actual conversation beyond the canned
        # greeting - a download button for a chat nobody has used yet has
        # nothing worth saving. The chat itself only ever lives in
        # session_state (see this function's docstring), so this is the
        # one way to keep a copy of it past the current browser session.
        if len(st.session_state[chat_key]) > 1:
            # type="primary" isn't decorative here - Streamlit's default
            # (type="secondary") button is a plain white pill with dark
            # text, which the rest of this panel never uses; against the
            # dark gradient background it rendered as a blank white box at
            # rest (only the hover ring gave it any color at all). Same
            # fix as "Download PDF" elsewhere in this file, which hits the
            # identical white-on-dark mismatch.
            st.download_button(
                "Download chat", data=_build_chat_transcript(user_name, st.session_state[chat_key]),
                file_name=f"carscout_chat_{user_name}.md".replace(" ", "_"), mime="text/markdown",
                key="download_chat", type="primary", icon=":material/download:", use_container_width=True,
            )

        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"]):
                # Same "$...$" -> LaTeX bug documented for the main report
                # view (two dollar amounts in one block get read as
                # matching math delimiters and the text between them goes
                # blank) - escape display-only, same as everywhere else
                # this pattern shows up; the stored value stays unescaped.
                st.markdown(_escape_for_markdown(msg["content"]))
                if msg.get("sources"):
                    st.caption(
                        "Sources: " + ", ".join(f"[{s['title']}]({s['url']})" for s in msg["sources"])
                    )

        if question := st.chat_input("e.g. Which one has the best safety rating?"):
            # Same moderation gate as the main due-diligence flow (fail
            # closed on flagged/error) - not the relevance check, though,
            # which is tuned to reject anything that isn't a car-symptom
            # description and would wrongly block ordinary comparison
            # questions like "which is cheapest".
            moderation = guards.check_moderation(question)
            if moderation["status"] in ("flagged", "error"):
                st.error("This input can't be processed - please ask about the listings above.")
            else:
                history_before = list(st.session_state[chat_key])
                st.session_state[chat_key].append({"role": "user", "content": question})
                # The history-rendering loop above already ran earlier in
                # this same script pass, before this question existed - so
                # the user's own bubble has to be drawn explicitly here too,
                # or it wouldn't appear until the rerun below, making the
                # reply below look like it came from nowhere.
                with st.chat_message("user"):
                    st.markdown(_escape_for_markdown(question))

                # A dedicated router decides BEFORE any answer is generated
                # whether this needs web search - never left to the answer-
                # writing call's own judgment (see comparison_chat.py's
                # classify_scope docstring). "error" is treated the same as
                # "in_scope": the safe default is answering from already-
                # verified saved data, never silently reaching out to the
                # web because a classifier call happened to fail.
                with st.spinner("Thinking..."):
                    scope = comparison_chat.classify_scope(question, ranked_entries)

                if scope["status"] == "needs_web":
                    with st.spinner("Searching the web..."):
                        web_result = comparison_chat.answer_with_web_search(
                            ranked_entries, question, history_before, rank_labels
                        )
                else:
                    web_result = {"status": "no_results"}

                if web_result["status"] == "ok":
                    # Non-streamed and moderation-checked before display -
                    # unlike the grounded path below, this answer is built
                    # from untrusted web content, so it can't be revealed
                    # live token-by-token; it has to be fully generated and
                    # pass the same fail-closed moderation gate used
                    # everywhere else in this app before the user sees any
                    # of it (see guards.check_moderation).
                    output_check = guards.check_moderation(web_result["answer"])
                    if output_check["status"] != "ok":
                        web_result = {"status": "error"}

                if web_result["status"] == "ok":
                    with st.chat_message("assistant"):
                        st.markdown(_escape_for_markdown(web_result["answer"]))
                        if web_result["sources"]:
                            st.caption(
                                "Sources: "
                                + ", ".join(f"[{s['title']}]({s['url']})" for s in web_result["sources"])
                            )
                    st.session_state[chat_key].append({
                        "role": "assistant",
                        "content": web_result["answer"],
                        "sources": web_result["sources"],
                    })
                else:
                    # Falls through to the normal grounded, streamed answer
                    # for "no_results" (Tavily unconfigured/failed/all
                    # results screened out) and "error" alike - a web-search
                    # miss should never dead-end the conversation when the
                    # saved listing data might still answer it, or at least
                    # say honestly that it doesn't.
                    with st.chat_message("assistant"):
                        try:
                            # Escaping has to happen per-chunk, live, not
                            # once on the full answer afterward - by the
                            # time st.write_stream() returns, the
                            # (unescaped) text has already been rendered
                            # chunk by chunk, so an answer with 2+ dollar
                            # amounts would flash the same "text between $
                            # signs goes blank" bug while streaming even if
                            # the final settled text looked fine.
                            # raw_chunks keeps the actual unescaped text for
                            # storage - escaping is display-only and must
                            # never leak into what's saved (same rule as
                            # _escape_for_markdown's own docstring).
                            raw_chunks = []

                            def _escaped_stream():
                                for chunk in comparison_chat.stream_comparison_answer(
                                    ranked_entries, question, history_before, rank_labels
                                ):
                                    raw_chunks.append(chunk)
                                    yield _escape_for_markdown(chunk)

                            st.write_stream(_escaped_stream())
                            full_answer = "".join(raw_chunks)
                            st.session_state[chat_key].append({"role": "assistant", "content": full_answer})
                        except Exception:
                            st.error("Something went wrong answering that - please try again.")
                            st.session_state[chat_key].append({
                                "role": "assistant",
                                "content": "Something went wrong answering that - please try again.",
                            })
                st.rerun()


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
        # The worst-signal-wins verdict badge (below, per card) is
        # deliberately conservative - reliability is never "green" - so on
        # its own almost every listing reads as the same amber "worth a
        # closer look", which doesn't tell listings apart. The border color
        # is what should actually say "choose this one" vs "maybe not this
        # one": gold for the top score, red for the strictly-lowest score
        # (only when it's really behind the best, not just tied). Anyone in
        # between keeps the default glass border - correctly reading as "no
        # strong opinion".
        is_lowest_flags = [
            i == len(ranked) - 1 and scores[i] is not None
            and best_score is not None and scores[i] < best_score
            for i in range(len(ranked))
        ]
        border_overrides = []
        # The progress bar's fill color, per card - bucketed from the same
        # score/MAX_RANK_SCORE fraction that already sets the bar's fill
        # width (see _score_color), so the color is always describing the
        # exact "X/MAX signals positive" number printed on the bar, not a
        # separate, coarser verdict that collapses most real listings into
        # one color regardless of score.
        progress_overrides = []
        # Single source of truth for each listing's rank label - used both
        # for the card badge below and passed into the comparison chat, so
        # the chat can't ever describe a listing's rank differently than
        # the badge the user is looking at. Before this, the chat had no
        # rank/recommendation info at all: it only ever saw price/mileage/
        # tiles per listing (see comparison_chat.py's _format_listing()),
        # so asking it "why did you recommend X" got "I didn't recommend
        # it" - technically accurate given what it could see, but flatly
        # contradicting the gold "Recommended" badge right next to the chat
        # button. This wasn't a tone/persona problem, it was a missing-data
        # one - a more confident-sounding prompt would've just made a wrong
        # answer sound more sure of itself.
        rank_labels: dict[int, str] = {}
        for i, entry in enumerate(ranked):
            tiles = entry.tiles()
            if tiles:
                progress_overrides.append((entry.id, _CHIP_COLORS[_score_color(scores[i])]))
            if not (tiles and rank_eligible > 1):
                continue
            if i == 0:
                rank_labels[entry.id] = "Recommended - best pick"
                border_overrides.append((entry.id, CARD_BORDER_RECOMMENDED))
            elif is_lowest_flags[i]:
                rank_labels[entry.id] = "Lowest-ranked - consider carefully"
                border_overrides.append((entry.id, CARD_BORDER_LOWEST))
            else:
                rank_labels[entry.id] = f"#{i + 1} of your evaluated listings"
        # One combined <style> block for every card that needs a border
        # override, injected once, outside the column loop entirely - an
        # earlier version injected one st.markdown() per qualifying card,
        # from inside its own column. Each of those calls became its own
        # invisible block element, and Streamlit's default inter-element
        # spacing around that block pushed the actual card (rendered right
        # after it) down by ~16px relative to cards with no override - a
        # real, measured misalignment (confirmed via getBoundingClientRect:
        # the recommended card's top sat 16px below its row neighbors),
        # not just a cosmetic nit.
        # DARK_CARD_CSS goes first, THESE per-card overrides last - both
        # sides use an equally-specific selector (a class selector and an
        # attribute selector carry identical CSS specificity), so with
        # !important on both, the browser's only remaining tiebreaker is
        # source order: whichever <style> block appears later in the DOM
        # wins. Getting this backwards is a real, previously-unnoticed bug:
        # DARK_CARD_CSS's shorthand `border: ... !important` was silently
        # winning over border_overrides's `border-color` the entire time
        # (confirmed via getComputedStyle - every card had the same default
        # border, gold/red never actually applied), because it used to be
        # injected after these override blocks instead of before.
        st.markdown(DARK_CARD_CSS, unsafe_allow_html=True)
        if border_overrides:
            rules = "".join(
                f".st-key-card_{card_id} {{ border-color: {color} !important; border-width: 2px !important; }}"
                for card_id, color in border_overrides
            )
            st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)
        if progress_overrides:
            rules = "".join(
                f'.st-key-card_{card_id} [role="progressbar"] > div,'
                f'.st-key-card_{card_id} [role="progressbar"] > div > div '
                f'{{ background: {chip["accent"]} !important; box-shadow: 0 0 10px {chip["glow"]} !important; }}'
                for card_id, chip in progress_overrides
            )
            st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)
        compare_cols = st.columns(3)
        for i, entry in enumerate(ranked):
            tiles = entry.tiles()
            score = scores[i]
            is_lowest = is_lowest_flags[i]
            with compare_cols[i % 3]:
                with st.container(border=True, key=f"card_{entry.id}"):
                    image_path = _vehicle_image_path(entry.make, entry.model)
                    if image_path:
                        st.image(_load_vehicle_image(str(image_path), *GRID_IMAGE_SIZE), use_container_width=True)

                    badges = []
                    if entry.id in rank_labels:
                        badge_class = "dark-badge-gold" if i == 0 else ("dark-badge-red" if is_lowest else "dark-badge-glass")
                        badge_icon = "🏆" if i == 0 else ("⚠️" if is_lowest else "☆")
                        badges.append(f'<span class="{badge_class}">{badge_icon} {rank_labels[entry.id]}</span>')
                    if tiles:
                        verdict_text = {
                            "red": "⚠️ Needs caution", "amber": "👁️ Worth a closer look", "green": "👍 Looks solid",
                        }[_overall_color(tiles)]
                        badges.append(f'<span class="dark-badge-glass">{verdict_text}</span>')
                    if badges:
                        st.markdown(f'<div class="dark-badge-row">{"".join(badges)}</div>', unsafe_allow_html=True)

                    st.markdown(
                        f'<div class="dark-title">{html.escape(f"{entry.year} {entry.make} {entry.model}")}</div>',
                        unsafe_allow_html=True,
                    )
                    if score is not None:
                        # A same-width bar per card is what actually makes
                        # "which one's better" jump out at a glance across
                        # the grid - reading four individual tile colors per
                        # card and comparing them mentally, card to card,
                        # doesn't. The score used to sit in its own
                        # <p> above an unlabeled bar - two disconnected
                        # elements. st.progress()'s own text= param glues
                        # the number to the bar it actually describes,
                        # instead of the bar being decoration with no
                        # information of its own.
                        st.progress(
                            score / MAX_RANK_SCORE,
                            text=f"{score}/{MAX_RANK_SCORE} signals positive",
                        )

                    st.markdown(
                        '<div class="dark-price-row">'
                        f"<span>${entry.asking_price:,.0f}</span><span class=\"dim\">&middot;</span>"
                        f"<span>{entry.odometer:,} mi</span><span class=\"dim\">&middot;</span>"
                        f"<span>{html.escape(_format_pacific(entry.created_at))}</span>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    # entry.symptom is free-text user input, unlike every
                    # other value rendered as raw HTML on this card (all of
                    # which come from fixed templates or curated listing
                    # data) - html.escape() here is load-bearing, not
                    # decorative.
                    st.markdown(
                        f'<div class="dark-info-bar"><span>ⓘ</span>'
                        f'<span class="symptom-text">{html.escape(entry.symptom)}</span></div>',
                        unsafe_allow_html=True,
                    )

                    if tiles:
                        st.markdown(_signal_squares_html(tiles), unsafe_allow_html=True)
                    else:
                        # Saved before tile classification existed - no
                        # per-signal colors to show, just the plain text.
                        st.caption("No at-a-glance summary saved for this older search.")
                    qa_history = entry.qa_history()
                    combined_report = _combined_report_markdown(qa_history)
                    with st.expander("View full report"):
                        # st.text (not st.write/st.markdown): never
                        # interprets markdown, so a preview built from any
                        # saved row (old or new format) always renders the
                        # same way, card to card.
                        st.text(memory_store.build_preview(entry.full_answer))
                        if len(qa_history) > 1:
                            st.caption(f"{len(qa_history)} questions asked about this listing.")
                        st.markdown(_escape_for_markdown(combined_report))
                    # build_report_pdf() falls back to "No data" per signal
                    # when tiles is {} (older, pre-tile-classification rows),
                    # so PDF download works for every saved search either way.
                    pdf_bytes = report_pdf.build_report_pdf(
                        f"{entry.year} {entry.make} {entry.model}",
                        f"${entry.asking_price:,.0f} - {entry.odometer:,} mi - {entry.symptom}",
                        tiles, combined_report,
                    )
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        st.download_button(
                            "Download PDF", data=pdf_bytes,
                            file_name=f"carscout_{entry.make}_{entry.model}_{entry.year}.pdf".replace(" ", "_"),
                            mime="application/pdf", key=f"pdf_{entry.id}",
                            type="primary", icon=":material/download:", use_container_width=True,
                        )
                    with btn_col2:
                        if st.button(
                            "Use this search", key=f"reuse_{entry.id}",
                            icon=":material/replay:", use_container_width=True,
                        ):
                            label = VIN_LABEL_BY_MAKE_MODEL.get((entry.make, entry.model), VIN_LABELS[0])
                            st.session_state["form_vin_label"] = label
                            st.session_state["form_symptom"] = entry.symptom
                            st.rerun()

        # Floating top-right button toggles the chat panel open/closed (see
        # CHAT_WIDGET_CSS + _render_comparison_chat_panel) - a right-docked
        # panel the user can minimize back down to just this button, rather
        # than an always-visible section at the bottom of the page or a
        # centered modal covering most of the screen.
        st.markdown(CHAT_WIDGET_CSS, unsafe_allow_html=True)
        st.session_state.setdefault("chat_panel_open", False)
        toggle_label = "✕ Close" if st.session_state["chat_panel_open"] else "💬 CarScout"
        if st.button(toggle_label, key="toggle_comparison_chat"):
            st.session_state["chat_panel_open"] = not st.session_state["chat_panel_open"]
            st.rerun()
        if st.session_state["chat_panel_open"]:
            _render_comparison_chat_panel(ranked, user_name, rank_labels)

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
