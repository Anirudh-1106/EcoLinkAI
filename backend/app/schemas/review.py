"""Review schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ReviewCreate(BaseModel):
    """Schema for creating a review after exchange completion."""
    exchange_id: uuid.UUID
    supplier_rating: int = Field(ge=1, le=5)
    buyer_rating: int = Field(ge=1, le=5)
    supplier_feedback: str | None = Field(None, max_length=1000)
    buyer_feedback: str | None = Field(None, max_length=1000)


class ReviewResponse(BaseModel):
    """Review response schema."""
    id: uuid.UUID
    exchange_id: uuid.UUID
    supplier_rating: int
    buyer_rating: int
    supplier_feedback: str | None = None
    buyer_feedback: str | None = None
    created_at: datetime

    # Joined
    supplier_plant_name: str | None = None
    buyer_plant_name: str | None = None

    model_config = {"from_attributes": True}


class ReviewListResponse(BaseModel):
    """Paginated review list."""
    items: list[ReviewResponse]
    total: int
    page: int
    page_size: int
