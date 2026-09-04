"""Inspect/clean up rows in the `recent_searches` table (src/memory_store.py).

Reads the same DATABASE_URL env var the app itself uses - point it at the
Render Postgres instance (its external connection string, from the Render
dashboard's Postgres "Connect" tab) to operate on production, or leave it
unset to operate on the local SQLite dev database instead. Never assume
which one you're pointed at - this script always prints the resolved target
(with any password masked) before doing anything, list or delete.

Usage (run from the repo root with `uv run`):
    uv run python scripts/clean_recent_searches.py --list
    uv run python scripts/clean_recent_searches.py --delete-user "SomeTestName"
    uv run python scripts/clean_recent_searches.py --delete-older-than 2
    uv run python scripts/clean_recent_searches.py --delete-all

Against production Postgres:
    DATABASE_URL="postgresql://..." uv run python scripts/clean_recent_searches.py --list
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # does not override an already-set DATABASE_URL env var

import memory_store  # noqa: E402
from sqlalchemy import delete, func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402


def _masked_target(url: str) -> str:
    """The resolved DB target with any password stripped, so it's safe to
    print - this is the one thing every mode of this script does first, so
    a user can visually confirm they're pointed at prod vs. local before
    any deletion runs."""
    parts = urlsplit(url)
    if parts.password:
        netloc = parts.netloc.replace(f":{parts.password}@", ":***@")
        parts = parts._replace(netloc=netloc)
    return urlunsplit(parts)


def _confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    answer = input(f"{prompt} Type DELETE to confirm: ")
    return answer.strip() == "DELETE"


def cmd_list(session: Session) -> None:
    rows = session.scalars(
        select(memory_store.RecentSearch).order_by(memory_store.RecentSearch.created_at.desc())
    ).all()
    if not rows:
        print("No rows in recent_searches.")
        return
    print(f"{len(rows)} row(s) total:\n")
    for r in rows:
        print(
            f"  id={r.id:<4} user={r.user_name or '(none)':<15} "
            f"{r.year} {r.make} {r.model:<12} ${r.asking_price:,.0f}  "
            f"{r.created_at.isoformat() if r.created_at else '?':<26} "
            f"symptom={r.symptom[:50]!r}"
        )
    print()
    by_user = session.execute(
        select(memory_store.RecentSearch.user_name, func.count())
        .group_by(memory_store.RecentSearch.user_name)
        .order_by(func.count().desc())
    ).all()
    print("By user_name:")
    for user_name, count in by_user:
        print(f"  {user_name or '(none)':<20} {count}")


def cmd_delete_user(session: Session, user_name: str, assume_yes: bool) -> None:
    count = session.scalar(
        select(func.count()).select_from(memory_store.RecentSearch).filter_by(user_name=user_name)
    )
    if not count:
        print(f"No rows found for user_name={user_name!r}.")
        return
    if not _confirm(f"About to delete {count} row(s) for user_name={user_name!r}.", assume_yes):
        print("Aborted - confirmation text didn't match.")
        return
    session.execute(delete(memory_store.RecentSearch).where(memory_store.RecentSearch.user_name == user_name))
    session.commit()
    print(f"Deleted {count} row(s) for user_name={user_name!r}.")


def cmd_delete_older_than(session: Session, days: int, assume_yes: bool) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    count = session.scalar(
        select(func.count()).select_from(memory_store.RecentSearch).where(memory_store.RecentSearch.created_at < cutoff)
    )
    if not count:
        print(f"No rows older than {days} day(s) ({cutoff.isoformat()}).")
        return
    if not _confirm(f"About to delete {count} row(s) older than {days} day(s) (before {cutoff.isoformat()}).", assume_yes):
        print("Aborted - confirmation text didn't match.")
        return
    session.execute(delete(memory_store.RecentSearch).where(memory_store.RecentSearch.created_at < cutoff))
    session.commit()
    print(f"Deleted {count} row(s) older than {days} day(s).")


def cmd_delete_all(session: Session, assume_yes: bool) -> None:
    count = session.scalar(select(func.count()).select_from(memory_store.RecentSearch))
    if not count:
        print("recent_searches is already empty.")
        return
    if not _confirm(f"About to delete ALL {count} row(s) in recent_searches.", assume_yes):
        print("Aborted - confirmation text didn't match.")
        return
    session.execute(delete(memory_store.RecentSearch))
    session.commit()
    print(f"Deleted all {count} row(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="List every row (default action).")
    group.add_argument("--delete-user", metavar="NAME", help="Delete every row for one user_name (exact match).")
    group.add_argument("--delete-older-than", type=int, metavar="DAYS", help="Delete rows older than N days.")
    group.add_argument("--delete-all", action="store_true", help="Delete every row in the table.")
    parser.add_argument("--yes", action="store_true", help="Skip the typed DELETE confirmation (for scripting).")
    args = parser.parse_args()

    engine = memory_store.get_engine()
    target = _masked_target(str(engine.url))
    print(f"Target database: {target}\n")

    memory_store.init_db(engine)

    with Session(engine) as session:
        if args.delete_user:
            cmd_delete_user(session, args.delete_user, args.yes)
        elif args.delete_older_than is not None:
            cmd_delete_older_than(session, args.delete_older_than, args.yes)
        elif args.delete_all:
            cmd_delete_all(session, args.yes)
        else:
            cmd_list(session)


if __name__ == "__main__":
    main()
