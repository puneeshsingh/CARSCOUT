"""Chat over the user's own saved evaluated listings (the "Your evaluated
listings" tab in streamlit_app.py) - answers comparison/follow-up questions
grounded in due-diligence results that are already computed and saved.

Deliberately NOT the LangGraph deep agent in complaint_lookup_agent.py - that
exists to call live tools (NHTSA/Craigslist) to research one new vehicle from
scratch. Here, the findings for every listing already exist as saved
RecentSearch rows, so this is one plain chat-completion call grounded in that
saved data, not a second agent with tools of its own.
"""

import logging

from openai import OpenAI

logger = logging.getLogger("carscout_comparison_chat")

_client = OpenAI()

COMPARISON_CHAT_MODEL = "gpt-4o-mini"

# A small local copy of the 4 signal titles, not imported from
# streamlit_app.py - that's the top-level Streamlit script; importing from it
# would re-run the whole app rather than just pulling a constant out of it.
_TILE_TITLES = {
    "reliability": "Reliability", "price": "Price fairness",
    "recalls": "Recall history", "safety": "Safety rating",
}

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant answering questions about used-car listings the \
user has already evaluated with CarScout, a due-diligence tool. Answer ONLY using the listing summaries \
below - never state or imply anything about a vehicle's reliability, price fairness, recalls, or safety \
rating beyond what's given here, and never draw on outside/training knowledge about specific vehicles or \
models. If the user asks something the data below doesn't cover, say so plainly instead of guessing.

Evaluated listings:
{listings_block}
"""


def _format_listing(entry, rank_label: str | None) -> str:
    tiles = entry.tiles()
    if tiles:
        tile_lines = "\n".join(
            f"  - {_TILE_TITLES.get(signal, signal)}: {tile.get('color', 'amber')} ({tile.get('headline', 'No data')})"
            for signal, tile in tiles.items()
        )
    else:
        tile_lines = "  - No at-a-glance summary saved for this older search."
    # Without this line, the model had no way to know which listing the
    # comparison grid actually recommends - asking "why did you recommend
    # X" got "I didn't recommend X" (technically true from what it could
    # see, but flatly contradicting the gold badge on screen). rank_label
    # is the exact same string the card badge renders - one source of
    # truth, so the chat can never disagree with what the user is looking
    # at (see rank_labels in streamlit_app.py).
    rank_line = f"Ranking among your evaluated listings: {rank_label}\n" if rank_label else ""
    return (
        f"{entry.year} {entry.make} {entry.model} - asking ${entry.asking_price:,.0f}, "
        f"{entry.odometer:,} mi, condition: {entry.condition or 'not stated'}\n"
        f"{rank_line}"
        f"Symptom/question originally asked: {entry.symptom}\n"
        f"{tile_lines}"
    )


def _build_messages(entries: list, question: str, chat_history: list[dict], rank_labels: dict) -> list[dict]:
    listings_block = "\n\n".join(_format_listing(e, rank_labels.get(e.id)) for e in entries)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(listings_block=listings_block)
    return [{"role": "system", "content": system_prompt}, *chat_history, {"role": "user", "content": question}]


def stream_comparison_answer(entries: list, question: str, chat_history: list[dict], rank_labels: dict):
    """Yields the assistant's answer as text chunks arrive, for
    st.write_stream() in streamlit_app.py - it renders each chunk live and
    hands back the fully concatenated string once the stream ends, which is
    what actually gets saved to chat history (same content either way, just
    not held back until the whole answer is done).

    entries: memory_store.RecentSearch rows for the current user, in
    whatever order the comparison grid already shows them in.
    chat_history: prior turns as [{"role": "user"|"assistant", "content": str}, ...],
    NOT including `question` itself.
    rank_labels: {entry.id: str} - see _format_listing(). Missing an id
    (older rows with no tiles saved) just omits that listing's rank line.

    Raises on failure (network/API error) instead of returning an
    {"status": "error"} dict like the previous blocking version did -
    st.write_stream() can't swap in a fallback message mid-stream, so the
    caller wraps the whole call in try/except (see streamlit_app.py) rather
    than checking a status field afterward.
    """
    messages = _build_messages(entries, question, chat_history, rank_labels)
    stream = _client.chat.completions.create(
        model=COMPARISON_CHAT_MODEL,
        temperature=0,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
    logger.info("COMPARISON_CHAT (streamed) -> question=%r", question)
