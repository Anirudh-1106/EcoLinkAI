"""
Application configuration.

Loads environment variables from the .env file and provides
centralized access to application settings.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root (backend/)
BASE_DIR = Path(__file__).resolve().parents[2]

# Load environment variables from backend/.env
load_dotenv(BASE_DIR / ".env")


class Settings:
    """
    Application settings loaded from environment variables.
    """

    def __init__(self) -> None:
        # ── Database ──────────────────────────────────────
        self.DATABASE_URL: str = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres123@localhost:5432/ecolinkai",
        )

        # ── Security / JWT ────────────────────────────────
        self.SECRET_KEY: str = os.getenv(
            "SECRET_KEY",
            "ecolinkai-dev-secret-change-in-production",
        )
        self.ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
        self.ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480")
        )

        # ── CORS ──────────────────────────────────────────
        raw_origins = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://localhost:3000",
        )
        self.CORS_ORIGINS: list[str] = [
            o.strip() for o in raw_origins.split(",") if o.strip()
        ]

        # ── AI / Model ────────────────────────────────────
        self.MODEL_PATH: str = os.getenv(
            "MODEL_PATH",
            str(BASE_DIR.parent / "ai" / "checkpoints"),
        )

        # ── Debug ─────────────────────────────────────────
        self.DEBUG: bool = (
            os.getenv("DEBUG", "False").strip().lower() == "true"
        )


settings = Settings()