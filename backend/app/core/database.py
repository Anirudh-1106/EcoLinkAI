"""
Database configuration.

Creates the SQLAlchemy engine, session factory,
and FastAPI dependency used throughout the application.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

# =====================================================
# SQLAlchemy Engine
# =====================================================

engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=10,
    max_overflow=20,
)

# =====================================================
# Session Factory
# =====================================================

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


# =====================================================
# Database Dependency
# =====================================================

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency.

    Creates a database session for each request
    and automatically closes it afterwards.
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()