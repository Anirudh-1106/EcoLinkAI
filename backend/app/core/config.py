"""
Application configuration.

Loads environment variables from the .env file and provides
centralized access to application settings.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """
    Application settings loaded from environment variables.
    """

    def __init__(self) -> None:
        self.DATABASE_URL: str | None = os.getenv("DATABASE_URL")

        self.DEBUG: bool = (
            os.getenv("DEBUG", "False").strip().lower() == "true"
        )

        if not self.DATABASE_URL:
            raise ValueError(
                "DATABASE_URL environment variable is not set."
            )


settings = Settings()