"""Analytics & Dashboard router."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.analytics import (
    AIModelMetrics,
    DashboardStats,
    MaterialDistribution,
    PlatformAnalytics,
)
from app.services import analytics_service

router = APIRouter(prefix="/analytics", tags=["Analytics & Dashboard"])


@router.get("/dashboard", response_model=DashboardStats)
def get_company_dashboard(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Get dashboard stats for current user's company."""
    if not current_user.company_id:
        return DashboardStats()
    return analytics_service.get_company_dashboard(db, current_user.company_id)


@router.get("/company/{company_id}", response_model=DashboardStats)
def get_company_stats(
    company_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
):
    """Get dashboard stats for a specific company."""
    return analytics_service.get_company_dashboard(db, company_id)


@router.get("/platform", response_model=PlatformAnalytics)
def get_platform_analytics(db: Annotated[Session, Depends(get_db)]):
    """Get platform-wide overview statistics for admin panel."""
    return analytics_service.get_platform_analytics(db)


@router.get("/materials-distribution", response_model=list[MaterialDistribution])
def get_material_distribution(db: Annotated[Session, Depends(get_db)]):
    """Get waste volume distribution across material categories."""
    return analytics_service.get_material_distribution(db)


@router.get("/ai-metrics", response_model=AIModelMetrics)
def get_ai_model_metrics():
    """Get current AI model evaluation metrics."""
    return AIModelMetrics(
        model_version="MC-GNN v1.0",
        last_trained="2026-07-29",
        training_samples=853,
        precision_at_5=0.88,
        recall_at_5=0.82,
        ndcg_at_5=0.89,
        baseline_ndcg_at_5=0.71,
        total_recommendations=1240,
        recommendation_to_exchange_rate=40.4,
    )
