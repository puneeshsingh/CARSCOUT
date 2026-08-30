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


def _format_listing(entry) -> str:
    tiles = entry.tiles()
    if tiles:
        tile_lines = "\n".join(
            f"  - {_TILE_TITLES.get(signal, signal)}: {tile.get('color', 'amber')} ({tile.get('headline', 'No data')})"
            for signal, tile in tiles.items()
        )
    else:
        tile_lines = "  - No at-a-glance summary saved for this older search."
    return (
        f"{entry.year} {entry.make} {entry.model} - asking ${entry.asking_price:,.0f}, "
        f"{entry.odometer:,} mi, condition: {entry.condition or 'not stated'}\n"
        f"Symptom/question originally asked: {entry.symptom}\n"
        f"{tile_lines}"
    )


def answer_comparison_question(entries: list, question: str, chat_history: list[dict]) -> dict:
    """entries: memory_store.RecentSearch rows for the current user, in
    whatever order the comparison grid already shows them in.
    chat_history: prior turns as [{"role": "user"|"assistant", "content": str}, ...],
    NOT including `question` itself.

    Returns {"status": "ok", "answer": str} or {"status": "error", "error": str}.
    Fails open at the call site (see streamlit_app.py) with a plain "something
    went wrong" message - this is a convenience feature over already-saved
    data, not a safety gate, so an API hiccup here should never look like a
    real finding about a vehicle.
    """
    listings_block = "\n\n".join(_format_listing(e) for e in entries)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(listings_block=listings_block)
    messages = [{"role": "system", "content": system_prompt}, *chat_history, {"role": "user", "content": question}]
    try:
        response = _client.chat.completions.create(
            model=COMPARISON_CHAT_MODEL,
            temperature=0,
            messages=messages,
        )
        answer = response.choices[0].message.content
        logger.info("COMPARISON_CHAT -> question=%r", question)
        return {"status": "ok", "answer": answer}
    except Exception as e:
        logger.error("COMPARISON_CHAT failed: %s", e)
        return {"status": "error", "error": str(e)}
