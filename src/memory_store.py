"""Durable "recent vehicle searches" memory store (Week 5).

Same code path against SQLite (local dev) and Postgres (Render deployment) -
only DATABASE_URL changes. This is app-level memory read/written by the
Streamlit UI, not fed into the agent's own reasoning; the agent's hard rules
live in complaint_lookup_agent.py's SYSTEM_PROMPT, unrelated to this store.
"""

import os
import re
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session

from config import DATA_CACHE_DIR

MAX_RECENT_SEARCHES = 20
PREVIEW_LENGTH = 200

_TITLE_LINE_RE = re.compile(r"^#*\s*Due-Diligence Report for .*?(\n+|$)", re.IGNORECASE)
_MARKDOWN_SYNTAX_RE = re.compile(r"[#*_]+")


def build_preview(final_answer: str) -> str:
    """Plain-text preview for the Recent Searches sidebar card.

    Computed at display time from the stored full answer (not stored
    separately) - the vehicle title line is dropped (the card already shows
    the vehicle as its own heading) and markdown syntax is stripped so the
    preview always renders as flat text. final_answer is otherwise raw
    markdown, and truncating that by character count can cut mid-token (e.g.
    an unclosed "**"), rendering wildly differently card to card depending on
    exactly where the cut lands.
    """
    text = _TITLE_LINE_RE.sub("", final_answer, count=1).strip()
    text = _MARKDOWN_SYNTAX_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= PREVIEW_LENGTH:
        return text
    truncated = text[:PREVIEW_LENGTH].rsplit(" ", 1)[0]
    return truncated + "..."


class Base(DeclarativeBase):
    pass


class RecentSearch(Base):
    __tablename__ = "recent_searches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    make = Column(String, nullable=False)
    model = Column(String, nullable=False)
    year = Column(Integer, nullable=False)
    asking_price = Column(Float, nullable=False)
    odometer = Column(Integer, nullable=False)
    condition = Column(String, nullable=True)
    symptom = Column(String, nullable=False)
    full_answer = Column(String, nullable=False)


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{DATA_CACHE_DIR / 'memory.db'}"

    # Render (and most Postgres hosts) hand out postgres:// or
    # postgresql://, but SQLAlchemy needs the driver named explicitly to
    # pick psycopg3 over the legacy psycopg2 default.
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def get_engine() -> Engine:
    return create_engine(_database_url())


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def save_search(
    engine: Engine,
    make: str,
    model: str,
    year: int,
    asking_price: float,
    odometer: int,
    condition: str | None,
    symptom: str,
    final_answer: str,
) -> None:
    with Session(engine) as session:
        # Upsert on the search's identity (vehicle + listing + symptom): a
        # re-run of the exact same search - e.g. via "Use this search" with
        # nothing changed - refreshes that one entry (new answer, bumped to
        # most recent) instead of cluttering the list with a duplicate tile.
        existing = session.scalars(
            select(RecentSearch).filter_by(
                make=make, model=model, year=year, asking_price=asking_price,
                odometer=odometer, condition=condition, symptom=symptom,
            )
        ).first()

        if existing:
            existing.created_at = datetime.now(timezone.utc)
            existing.full_answer = final_answer
        else:
            session.add(
                RecentSearch(
                    created_at=datetime.now(timezone.utc),
                    make=make,
                    model=model,
                    year=year,
                    asking_price=asking_price,
                    odometer=odometer,
                    condition=condition,
                    symptom=symptom,
                    full_answer=final_answer,
                )
            )
        session.commit()

        # Forgetting policy: keep only the most recent MAX_RECENT_SEARCHES
        # rows, drop anything older.
        stale_ids = session.scalars(
            select(RecentSearch.id)
            .order_by(RecentSearch.created_at.desc())
            .offset(MAX_RECENT_SEARCHES)
        ).all()
        if stale_ids:
            session.execute(delete(RecentSearch).where(RecentSearch.id.in_(stale_ids)))
            session.commit()


def get_recent_searches(engine: Engine, limit: int = MAX_RECENT_SEARCHES) -> list[RecentSearch]:
    with Session(engine) as session:
        return list(
            session.scalars(
                select(RecentSearch).order_by(RecentSearch.created_at.desc()).limit(limit)
            )
        )
