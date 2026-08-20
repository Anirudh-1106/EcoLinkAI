"""Common Pydantic schemas used across the application."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    detail: str | None = None


class PaginatedResponse(BaseModel):
    """Paginated list wrapper."""
    items: list
    total: int
    page: int
    page_size: int
    total_pages: int


class IDResponse(BaseModel):
    """Response containing just an ID."""
    id: uuid.UUID


class TimestampMixin(BaseModel):
    """Mixin providing created_at and updated_at."""
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
