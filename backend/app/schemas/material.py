"""Material schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class MaterialCreate(BaseModel):
    """Schema for creating a material."""
    material_name: str = Field(max_length=255)
    material_category: str
    hazard_class: str = "Non-Hazardous"
    default_density: Decimal | None = Field(None, ge=0)
    carbon_factor: Decimal | None = Field(None, ge=0)
    description: str | None = Field(None, max_length=1000)


class MaterialUpdate(BaseModel):
    """Schema for updating a material."""
    material_name: str | None = Field(None, max_length=255)
    material_category: str | None = None
    hazard_class: str | None = None
    default_density: Decimal | None = Field(None, ge=0)
    carbon_factor: Decimal | None = Field(None, ge=0)
    description: str | None = Field(None, max_length=1000)
    is_active: bool | None = None


class MaterialResponse(BaseModel):
    """Material response schema."""
    id: uuid.UUID
    material_name: str
    material_category: str
    hazard_class: str
    default_density: Decimal | None = None
    carbon_factor: Decimal | None = None
    description: str | None = None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class MaterialListResponse(BaseModel):
    """Paginated material list."""
    items: list[MaterialResponse]
    total: int
    page: int
    page_size: int
