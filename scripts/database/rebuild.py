"""
Rebuild the database from all of its sources.

Two datasets feed this platform, and a complete database needs both:

  1. datasets/csv/*.csv        - synthetic companies, listings and the
                                 exchange history the MC-GNN trains on.
  2. datasets/kinfra_data.xlsx - real companies collected from the KINFRA
                                 industrial park. They sell materials but
                                 have no recorded exchange history, because
                                 the companies did not report their buyers.

They are loaded by separate scripts, so loading only the first is an easy
mistake to make -- and a silent one, since the result still looks like a
working database. This module is the supported way to rebuild, so that
cannot happen by accident.

Usage:
    python -m scripts.database.rebuild            # load whatever is missing
    python -m scripts.database.rebuild --reset    # drop every table first
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
# The synthetic loader lives under scripts/, the KINFRA loader at the repo
# root, so both the project root and backend need to be importable.
for path in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("rebuild")


def _reset_schema() -> None:
    """Drop and recreate every table. Destroys all data."""
    from app.core.database import engine
    from app.models import Base

    logger.warning("Dropping all tables - every existing row will be lost.")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info("Schema recreated.")


def _counts() -> dict[str, int]:
    from app.core.database import SessionLocal
    from app.models.company import Company
    from app.models.exchange_request import ExchangeRequest
    from app.models.plant import Plant
    from app.models.user import User
    from app.models.waste_listing import WasteListing

    db = SessionLocal()
    try:
        return {
            "companies": db.query(Company).count(),
            "kinfra_companies": db.query(Company)
            .filter(Company.company_name.like("KINFRA%"))
            .count(),
            "plants": db.query(Plant).count(),
            "waste_listings": db.query(WasteListing).count(),
            "exchange_requests": db.query(ExchangeRequest).count(),
            "users": db.query(User).count(),
        }
    finally:
        db.close()


def rebuild(reset: bool = False) -> dict[str, int]:
    """
    Load every dataset into the database and verify the result.

    Raises RuntimeError if the real KINFRA companies are missing afterwards,
    so a half-loaded database fails here rather than surfacing later as
    recommendations that quietly exclude real sellers.
    """
    from ingest_kinfra import ingest_kinfra_data
    from scripts.database.seed import seed_database

    if reset:
        _reset_schema()

    logger.info("Step 1/2: synthetic dataset (datasets/csv)")
    seed_database()

    logger.info("Step 2/2: real KINFRA dataset (datasets/kinfra_data.xlsx)")
    # Idempotent: updates existing rows rather than duplicating them, so this
    # is safe whether or not the database was just reset.
    expected_kinfra = ingest_kinfra_data()

    counts = _counts()

    if expected_kinfra and counts["kinfra_companies"] < expected_kinfra:
        raise RuntimeError(
            f"Expected {expected_kinfra} KINFRA companies after rebuild, "
            f"found {counts['kinfra_companies']}. The real dataset did not load."
        )

    logger.info("Rebuild complete:")
    for key, value in counts.items():
        logger.info(f"  {key:18s} {value}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the EcoLinkAI database from all of its sources."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all tables before loading. Destroys existing data.",
    )
    args = parser.parse_args()

    rebuild(reset=args.reset)


if __name__ == "__main__":
    main()
