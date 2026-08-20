"""Plant schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class PlantCreate(BaseModel):
    """Schema for creating a plant."""
    plant_name: str = Field(max_length=255)
    plant_type: str = Field(max_length=100)
    address: str = Field(max_length=500)
    district: str = Field(max_length=100)
    state: str = Field(max_length=100, default="Kerala")
    country: str = Field(max_length=100, default="India")
    latitude: Decimal = Field(ge=Decimal("-90"), le=Decimal("90"))
    longitude: Decimal = Field(ge=Decimal("-180"), le=Decimal("180"))


class PlantUpdate(BaseModel):
    """Schema for updating a plant."""
    plant_name: str | None = Field(None, max_length=255)
    plant_type: str | None = Field(None, max_length=100)
    address: str | None = Field(None, max_length=500)
    district: str | None = Field(None, max_length=100)
    state: str | None = Field(None, max_length=100)
    latitude: Decimal | None = Field(None, ge=Decimal("-90"), le=Decimal("90"))
    longitude: Decimal | None = Field(None, ge=Decimal("-180"), le=Decimal("180"))
    is_active: bool | None = None


class PlantResponse(BaseModel):
    """Plant response schema."""
    id: uuid.UUID
    company_id: uuid.UUID
    plant_name: str
    plant_type: str
    address: str
    district: str
    state: str
    country: str
    latitude: Decimal
    longitude: Decimal
    is_active: bool
    created_at: datetime
    updated_at: datetime
    company_name: str | None = None

    model_config = {"from_attributes": True}


class PlantListResponse(BaseModel):
    """Paginated plant list."""
    items: list[PlantResponse]
    total: int
    page: int
    page_size: int
