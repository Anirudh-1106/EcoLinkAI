"""Analytics and dashboard schemas."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel


class DashboardStats(BaseModel):
    """Company dashboard overview stats."""
    total_plants: int = 0
    active_waste_listings: int = 0
    open_requirements: int = 0
    pending_exchange_requests: int = 0
    completed_exchanges: int = 0
    total_carbon_saved_kg: Decimal = Decimal("0.00")
    trust_score: float = 0.0
    average_rating: float = 0.0


class PlatformAnalytics(BaseModel):
    """Platform-wide analytics for admin dashboard."""
    total_companies: int = 0
    total_plants: int = 0
    total_materials: int = 0
    total_waste_listings: int = 0
    total_requirements: int = 0
    total_exchange_requests: int = 0
    total_exchanges: int = 0
    total_reviews: int = 0
    total_waste_exchanged_tons: Decimal = Decimal("0.00")
    total_carbon_saved_kg: Decimal = Decimal("0.00")
    total_revenue: Decimal = Decimal("0.00")
    average_rating: float = 0.0
    exchange_success_rate: float = 0.0


class MaterialDistribution(BaseModel):
    """Material category distribution for charts."""
    category: str
    count: int
    total_quantity: Decimal = Decimal("0.00")


class MonthlyTrend(BaseModel):
    """Monthly exchange/listing trend data."""
    month: str
    exchanges: int = 0
    listings: int = 0
    carbon_saved: Decimal = Decimal("0.00")


class TopPartner(BaseModel):
    """Top exchange partner by volume."""
    company_name: str
    plant_name: str
    exchange_count: int
    total_quantity: Decimal


class AIModelMetrics(BaseModel):
    """AI model performance metrics for admin."""
    model_version: str = "v1.0"
    last_trained: str | None = None
    training_samples: int = 0
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    ndcg_at_5: float = 0.0
    baseline_ndcg_at_5: float = 0.0
    total_recommendations: int = 0
    recommendation_to_exchange_rate: float = 0.0
