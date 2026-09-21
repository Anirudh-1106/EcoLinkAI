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

# Convenience defaults for a developer machine. They are in version control,
# so treating any of them as a real setting means running on credentials the
# whole internet can read -- _assert_production_ready refuses to start with
# them once ENVIRONMENT=production.
DEV_SECRET_KEY = "ecolinkai-dev-secret-change-in-production"
DEV_DATABASE_URL = "postgresql://postgres:postgres123@localhost:5432/ecolinkai"

# A token is only as unguessable as the key that signs it.
MIN_SECRET_KEY_LENGTH = 32


class Settings:
    """
    Application settings loaded from environment variables.
    """

    def __init__(self) -> None:
        # ── Environment ───────────────────────────────────
        # Anything other than "production" is treated as a developer machine
        # and keeps the convenience defaults below.
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").strip().lower()

        # ── Database ──────────────────────────────────────
        self.DATABASE_URL: str = os.getenv(
            "DATABASE_URL",
            DEV_DATABASE_URL,
        )

        # ── Security / JWT ────────────────────────────────
        self.SECRET_KEY: str = os.getenv(
            "SECRET_KEY",
            DEV_SECRET_KEY,
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
        # Resolved against BASE_DIR (backend/), not the process's cwd, so a
        # relative MODEL_PATH in .env works regardless of where the server
        # is launched from.
        raw_model_path = os.getenv(
            "MODEL_PATH",
            str(BASE_DIR.parent / "ai" / "checkpoints"),
        )
        model_path = Path(raw_model_path)
        if not model_path.is_absolute():
            model_path = (BASE_DIR / model_path).resolve()
        self.MODEL_PATH: str = str(model_path)

        # How long the cached industrial graph and its GNN node embeddings
        # stay valid before being rebuilt from the database.
        self.GRAPH_CACHE_TTL_SECONDS: int = int(
            os.getenv("GRAPH_CACHE_TTL_SECONDS", "600")
        )

        # ── Debug ─────────────────────────────────────────
        self.DEBUG: bool = (
            os.getenv("DEBUG", "False").strip().lower() == "true"
        )

        if self.is_production:
            self._assert_production_ready()

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    def _assert_production_ready(self) -> None:
        """
        Refuse to start on a development default in production.

        SECRET_KEY signs every access token, so a server falling back to the
        committed default will happily accept tokens minted by anyone who has
        read this file -- for any account, including an administrator. Nothing
        about that looks wrong from the outside: the app starts, serves
        traffic and logs nothing unusual.

        Failing at startup turns a silent authentication bypass into an
        obvious crash. Every problem is collected first so a misconfigured
        deploy is fixed in one pass rather than one restart at a time.
        """
        problems: list[str] = []

        if self.SECRET_KEY == DEV_SECRET_KEY:
            problems.append(
                "SECRET_KEY is still the development default, which is in version "
                "control -- anyone who has seen the repository can forge a login "
                "token for any account."
            )
        elif len(self.SECRET_KEY) < MIN_SECRET_KEY_LENGTH:
            problems.append(
                f"SECRET_KEY is {len(self.SECRET_KEY)} characters; use at least "
                f"{MIN_SECRET_KEY_LENGTH} so it cannot be guessed."
            )

        if self.DATABASE_URL == DEV_DATABASE_URL:
            problems.append(
                "DATABASE_URL is still the local development database, password "
                "and all."
            )

        local_origins = [o for o in self.CORS_ORIGINS if "localhost" in o or "127.0.0.1" in o]
        if local_origins:
            problems.append(
                f"CORS_ORIGINS still allows {', '.join(local_origins)}; set it to "
                "the real site origin."
            )

        if self.DEBUG:
            problems.append("DEBUG is on, which exposes internals in error responses.")

        if not problems:
            return

        raise RuntimeError(
            "Refusing to start with ENVIRONMENT=production:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nGenerate a key with:\n"
            '  python -c "import secrets; print(secrets.token_urlsafe(64))"\n'
            "Set these in the environment (or backend/.env) and start again."
        )


settings = Settings()