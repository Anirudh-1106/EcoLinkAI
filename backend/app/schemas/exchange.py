"""Exchange request and exchange schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


# ─── Exchange Request ─────────────────────────────────

class ExchangeRequestCreate(BaseModel):
    """Schema for sending an exchange request."""
    waste_listing_id: uuid.UUID
    requirement_id: uuid.UUID | None = None
    buyer_plant_id: uuid.UUID
    offered_price_per_unit: Decimal | None = Field(None, ge=0)
    requested_quantity: Decimal = Field(gt=0)
    remarks: str | None = Field(None, max_length=1000)
    transportation_mode: str | None = None
    recommendation_rank: int | None = Field(None, ge=1)


class ExchangeRequestResponse(BaseModel):
    """Exchange request response schema."""
    id: uuid.UUID
    supplier_plant_id: uuid.UUID
    buyer_plant_id: uuid.UUID
    waste_listing_id: uuid.UUID
    requirement_id: uuid.UUID | None = None
    compatibility_score: Decimal
    ai_confidence_score: Decimal
    recommendation_rank: int
    distance_km: Decimal
    estimated_transport_cost: Decimal
    estimated_carbon_emission: Decimal
    estimated_carbon_saving: Decimal | None = None
    recommendation_reason: str | None = None
    status: str
    created_at: datetime
    updated_at: datetime

    # Joined
    supplier_plant_name: str | None = None
    buyer_plant_name: str | None = None
    supplier_company_id: uuid.UUID | None = None
    buyer_company_id: uuid.UUID | None = None
    supplier_company_name: str | None = None
    buyer_company_name: str | None = None
    material_name: str | None = None
    waste_quantity: Decimal | None = None

    model_config = {"from_attributes": True}


class ExchangeRequestAction(BaseModel):
    """Accept/reject an exchange request."""
    action: str = Field(pattern="^(accept|reject)$")
    reason: str | None = Field(None, max_length=500)


# ─── Exchange (Transaction) ──────────────────────────

class ExchangeCreate(BaseModel):
    """Schema for creating an exchange from accepted request."""
    exchange_request_id: uuid.UUID
    agreed_price: Decimal = Field(ge=0)
    transport_cost: Decimal = Field(ge=0)
    expected_delivery_date: datetime | None = None


class ExchangeUpdate(BaseModel):
    """Update exchange status."""
    exchange_status: str | None = None
    shipment_status: str | None = None
    actual_quantity: Decimal | None = Field(None, ge=0)
    actual_carbon_emission: Decimal | None = Field(None, ge=0)
    actual_carbon_saving: Decimal | None = Field(None, ge=0)
    delivered_at: datetime | None = None
    completion_notes: str | None = Field(None, max_length=1000)


class ExchangeResponse(BaseModel):
    """Exchange response schema."""
    id: uuid.UUID
    exchange_request_id: uuid.UUID
    exchange_status: str
    shipment_status: str
    agreed_price: Decimal
    transport_cost: Decimal
    actual_quantity: Decimal | None = None
    actual_carbon_emission: Decimal | None = None
    actual_carbon_saving: Decimal | None = None
    expected_delivery_date: datetime | None = None
    delivered_at: datetime | None = None
    completion_notes: str | None = None
    created_at: datetime
    updated_at: datetime

    # Joined context
    supplier_plant_name: str | None = None
    buyer_plant_name: str | None = None
    material_name: str | None = None

    model_config = {"from_attributes": True}


class ExchangeListResponse(BaseModel):
    """Paginated exchange list."""
    items: list[ExchangeResponse]
    total: int
    page: int
    page_size: int
