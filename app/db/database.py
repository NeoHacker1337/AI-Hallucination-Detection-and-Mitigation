"""SQLAlchemy engine and session factory (SQLite by default)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Create (once) and return the engine. SQLite runs in WAL mode so the API can read while
    an experiment thread writes."""
    global _engine, _SessionLocal
    if _engine is None:
        url = get_settings().database_url
        kwargs: dict = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        _engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):

            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, _record):  # type: ignore[no-untyped-def]
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()

        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db() -> None:
    """Create all tables."""
    from app.db import models  # noqa: F401  (register models)

    engine = get_engine()
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)


# Columns added after the first release; SQLite databases created earlier get them via ALTER TABLE.
_LATER_COLUMNS: dict[str, dict[str, str]] = {"documents": {"content_date": "VARCHAR(32)"}}


def _add_missing_columns(engine: Engine) -> None:
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, columns in _LATER_COLUMNS.items():
            if not inspector.has_table(table):
                continue
            existing = {c["name"] for c in inspector.get_columns(table)}
            for name, ddl in columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session context manager."""
    get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session."""
    with session_scope() as session:
        yield session
