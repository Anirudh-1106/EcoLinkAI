"""Analytics & Dashboard router."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user, require_role
from app.models.user import User, UserRole
from app.schemas.analytics import (
    AIModelMetrics,
    DashboardStats,
    MaterialDistribution,
    PlatformAnalytics,
)
from app.services import analytics_service, graph_cache, recommendation_service

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


@router.get("/company", response_model=PlatformAnalytics)
def get_my_company_analytics(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Get analytics scoped to the current user's own company."""
    if not current_user.company_id:
        return PlatformAnalytics()
    return analytics_service.get_company_analytics(db, current_user.company_id)


@router.get("/platform", response_model=PlatformAnalytics)
def get_platform_analytics(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_role(UserRole.ADMIN))],
):
    """Get platform-wide overview statistics for admin panel (admin only)."""
    return analytics_service.get_platform_analytics(db)


@router.get("/materials-distribution", response_model=list[MaterialDistribution])
def get_material_distribution(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Get waste volume distribution across material categories for the current user's company."""
    if not current_user.company_id:
        return []
    return analytics_service.get_material_distribution(db, current_user.company_id)


@router.get("/ai-status")
def get_ai_runtime_status(
    current_user: Annotated[User, Depends(require_role(UserRole.ADMIN))],
):
    """
    Whether MC-GNN inference is actually live, and how fresh its cached graph is.

    Reports the real runtime state rather than what the UI claims, so a silent
    fallback to baseline scoring is visible instead of hidden.
    """
    return graph_cache.status()


@router.get("/ai-metrics", response_model=AIModelMetrics)
def get_ai_model_metrics(db: Annotated[Session, Depends(get_db)]):
    """
    Current AI model evaluation metrics.

    Ranking metrics are measured against the historical exchange-request
    edges when the graph cache is refreshed; adoption figures are read
    live from the database.
    """
    measured = graph_cache.metrics() or {}
    adoption = analytics_service.get_recommendation_adoption(db)

    return AIModelMetrics(
        model_version=recommendation_service._model_version,
        last_trained=graph_cache.checkpoint_mtime(),
        training_samples=measured.get("training_samples", 0),
        precision_at_5=measured.get("precision_at_5", 0.0),
        recall_at_5=measured.get("recall_at_5", 0.0),
        ndcg_at_5=measured.get("ndcg_at_5", 0.0),
        baseline_ndcg_at_5=measured.get("baseline_ndcg_at_5", 0.0),
        total_recommendations=adoption["total_recommendations"],
        recommendation_to_exchange_rate=adoption["recommendation_to_exchange_rate"],
    )
