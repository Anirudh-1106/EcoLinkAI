"""Waste listing schemas."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class WasteListingCreate(BaseModel):
    """Schema for creating a waste listing."""
    plant_id: uuid.UUID
    material_id: uuid.UUID
    description: str | None = Field(None, max_length=1000)
    quantity: Decimal = Field(gt=0)
    unit: str
    purity_percentage: Decimal | None = Field(None, ge=0, le=100)
    moisture_percentage: Decimal | None = Field(None, ge=0, le=100)
    quality_grade: str | None = Field(None, max_length=20)
    price_per_unit: Decimal | None = Field(None, ge=0)
    available_from: date
    available_until: date


class WasteListingUpdate(BaseModel):
    """Schema for updating a waste listing."""
    description: str | None = Field(None, max_length=1000)
    quantity: Decimal | None = Field(None, gt=0)
    purity_percentage: Decimal | None = Field(None, ge=0, le=100)
    moisture_percentage: Decimal | None = Field(None, ge=0, le=100)
    quality_grade: str | None = Field(None, max_length=20)
    price_per_unit: Decimal | None = Field(None, ge=0)
    available_until: date | None = None
    status: str | None = None


class WasteListingResponse(BaseModel):
    """Waste listing response schema."""
    id: uuid.UUID
    plant_id: uuid.UUID
    material_id: uuid.UUID
    description: str | None = None
    quantity: Decimal
    unit: str
    purity_percentage: Decimal | None = None
    moisture_percentage: Decimal | None = None
    quality_grade: str | None = None
    price_per_unit: Decimal | None = None
    available_from: date
    available_until: date
    status: str
    created_at: datetime
    updated_at: datetime

    # Joined fields
    plant_name: str | None = None
    material_name: str | None = None
    company_name: str | None = None
    company_id: uuid.UUID | None = None

    model_config = {"from_attributes": True}


class WasteListingListResponse(BaseModel):
    """Paginated waste listing list."""
    items: list[WasteListingResponse]
    total: int
    page: int
    page_size: int
