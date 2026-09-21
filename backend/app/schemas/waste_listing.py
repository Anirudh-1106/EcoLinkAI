"""Waste listing schemas."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, Field, model_validator

from app.enums.common import QuantityUnit


class WasteListingCreate(BaseModel):
    """Schema for creating a waste listing."""
    plant_id: uuid.UUID
    material_id: uuid.UUID
    description: str | None = Field(None, max_length=1000)
    quantity: Decimal = Field(gt=0)
    # Constrained to the enum rather than free text: the unit decides how a
    # quantity converts, what a price means per unit, and how freight is
    # costed, so an unrecognised one is not a cosmetic problem.
    unit: QuantityUnit
    purity_percentage: Decimal | None = Field(None, ge=0, le=100)
    moisture_percentage: Decimal | None = Field(None, ge=0, le=100)
    quality_grade: str | None = Field(None, max_length=20)
    price_per_unit: Decimal | None = Field(None, ge=0)
    available_from: date
    available_until: date

    @model_validator(mode="after")
    def _check_availability_window(self) -> "WasteListingCreate":
        # The database has always enforced this, so a reversed window used to
        # surface as an IntegrityError and a 500. Checking here returns the
        # ordinary 422 that tells the caller which field is wrong.
        if self.available_until < self.available_from:
            raise ValueError("available_until must be on or after available_from")
        return self


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
    plant_district: str | None = None
    plant_latitude: Decimal | None = None
    plant_longitude: Decimal | None = None
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
