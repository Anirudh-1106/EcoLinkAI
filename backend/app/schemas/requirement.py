"""Requirement schemas."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class RequirementCreate(BaseModel):
    """Schema for creating a material requirement."""
    plant_id: uuid.UUID
    material_id: uuid.UUID
    quantity: Decimal = Field(gt=0)
    unit: str
    minimum_purity: Decimal | None = Field(None, ge=0, le=100)
    maximum_budget_per_unit: Decimal | None = Field(None, ge=0)
    required_before: date
    preferred_max_distance_km: Decimal | None = Field(None, ge=0)
    notes: str | None = Field(None, max_length=1000)


class RequirementUpdate(BaseModel):
    """Schema for updating a requirement."""
    quantity: Decimal | None = Field(None, gt=0)
    minimum_purity: Decimal | None = Field(None, ge=0, le=100)
    maximum_budget_per_unit: Decimal | None = Field(None, ge=0)
    required_before: date | None = None
    preferred_max_distance_km: Decimal | None = Field(None, ge=0)
    notes: str | None = Field(None, max_length=1000)
    status: str | None = None


class RequirementResponse(BaseModel):
    """Requirement response schema."""
    id: uuid.UUID
    plant_id: uuid.UUID
    material_id: uuid.UUID
    quantity: Decimal
    unit: str
    minimum_purity: Decimal | None = None
    maximum_budget_per_unit: Decimal | None = None
    required_before: date
    preferred_max_distance_km: Decimal | None = None
    notes: str | None = None
    status: str
    created_at: datetime

    # Joined
    plant_name: str | None = None
    material_name: str | None = None
    company_name: str | None = None

    model_config = {"from_attributes": True}


class RequirementListResponse(BaseModel):
    """Paginated requirement list."""
    items: list[RequirementResponse]
    total: int
    page: int
    page_size: int
