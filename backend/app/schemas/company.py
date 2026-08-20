"""Company schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class CompanyCreate(BaseModel):
    """Schema for creating a company."""
    company_name: str = Field(max_length=255)
    industry_type: str = Field(max_length=100)
    registration_number: str = Field(max_length=100)
    gst_number: str = Field(max_length=15)
    license_number: str = Field(max_length=100)


class CompanyUpdate(BaseModel):
    """Schema for updating a company."""
    company_name: str | None = Field(None, max_length=255)
    industry_type: str | None = Field(None, max_length=100)
    is_active: bool | None = None


class CompanyResponse(BaseModel):
    """Company response schema."""
    id: uuid.UUID
    company_name: str
    industry_type: str
    registration_number: str
    gst_number: str
    license_number: str
    verification_status: str
    trust_score: float
    is_active: bool
    created_at: datetime
    updated_at: datetime
    plant_count: int | None = None

    model_config = {"from_attributes": True}


class CompanyListResponse(BaseModel):
    """Paginated company list."""
    items: list[CompanyResponse]
    total: int
    page: int
    page_size: int
