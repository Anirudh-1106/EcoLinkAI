"""Recommendation schemas — the core AI output."""

from __future__ import annotations

import uuid
from decimal import Decimal

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """Request for AI partner recommendations."""
    waste_listing_id: uuid.UUID
    max_results: int = Field(default=10, ge=1, le=50)
    max_distance_km: Decimal | None = Field(None, ge=0)


class PartnerExplanation(BaseModel):
    """Feature-level explanation of why a partner was recommended."""
    material_compatibility: str
    quantity_match_pct: float
    quality_match_pct: float | None = None
    distance_km: float
    transport_feasibility: str
    estimated_transport_cost: float
    trust_score: float
    carbon_benefit_kg: float
    historical_exchange_count: int = 0
    recommendation_summary: str


class PartnerCard(BaseModel):
    """A single recommended partner with full context."""
    rank: int
    ai_score: float = Field(description="MC-GNN composite score 0-100")
    model_type: str = Field(description="mc_gnn or baseline")

    # Partner info
    company_id: uuid.UUID
    company_name: str
    plant_id: uuid.UUID
    plant_name: str
    plant_district: str
    plant_state: str

    # Requirement / demand info
    requirement_id: uuid.UUID | None = None
    required_quantity: Decimal | None = None
    required_purity: Decimal | None = None

    # Match metrics
    material_name: str
    compatibility_score: float
    distance_km: float
    estimated_transport_cost: float
    estimated_carbon_saving: float

    # Explanation
    explanation: PartnerExplanation


class RecommendationResponse(BaseModel):
    """Full recommendation response with multiple partners."""
    waste_listing_id: uuid.UUID
    material_name: str
    supplier_plant_name: str
    total_candidates: int
    recommendations: list[PartnerCard]
    model_version: str = "v1.0"
    inference_time_ms: float | None = None
