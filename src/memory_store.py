"""Durable "recent vehicle searches" memory store (Week 5).

Same code path against SQLite (local dev) and Postgres (Render deployment) -
only DATABASE_URL changes. This is app-level memory read/written by the
Streamlit UI, not fed into the agent's own reasoning; the agent's hard rules
live in complaint_lookup_agent.py's SYSTEM_PROMPT, unrelated to this store.
"""

import os
import re
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, delete, inspect, select, text
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
    # Nullable for backward compatibility with rows written before named
    # identity existed - not real auth, just a typed label the UI scopes the
    # list by (see agent/streamlit_app.py).
    user_name = Column(String, nullable=True)
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

    # create_all() only creates missing tables - it doesn't alter an
    # existing one, and this table already existed (locally and on Render)
    # before user_name was added. No migration framework in this project
    # (Alembic would be overkill for one column), so: add it directly if
    # missing. Idempotent - a no-op once the column exists. ADD COLUMN
    # syntax here works on both SQLite and Postgres.
    inspector = inspect(engine)
    columns = {col["name"] for col in inspector.get_columns("recent_searches")}
    if "user_name" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE recent_searches ADD COLUMN user_name VARCHAR"))


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
    user_name: str | None = None,
) -> None:
    with Session(engine) as session:
        # Upsert on the search's identity (user + vehicle + listing +
        # symptom): a re-run of the exact same search - e.g. via "Use this
        # search" with nothing changed - refreshes that one entry (new
        # answer, bumped to most recent) instead of cluttering the list with
        # a duplicate tile. user_name is part of the identity so the same
        # search by two different people doesn't overwrite each other.
        existing = session.scalars(
            select(RecentSearch).filter_by(
                user_name=user_name, make=make, model=model, year=year, asking_price=asking_price,
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
                    user_name=user_name,
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
        # rows *for this user*, drop anything older - scoped per user_name so
        # one active person's searches can't push another's out of the list.
        stale_ids = session.scalars(
            select(RecentSearch.id)
            .filter_by(user_name=user_name)
            .order_by(RecentSearch.created_at.desc())
            .offset(MAX_RECENT_SEARCHES)
        ).all()
        if stale_ids:
            session.execute(delete(RecentSearch).where(RecentSearch.id.in_(stale_ids)))
            session.commit()


def get_recent_searches(
    engine: Engine, user_name: str | None = None, limit: int = MAX_RECENT_SEARCHES
) -> list[RecentSearch]:
    with Session(engine) as session:
        return list(
            session.scalars(
                select(RecentSearch)
                .filter_by(user_name=user_name)
                .order_by(RecentSearch.created_at.desc())
                .limit(limit)
            )
        )
