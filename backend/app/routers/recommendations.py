"""Recommendations router — the primary AI endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse
from app.services import recommendation_service

router = APIRouter(prefix="/recommendations", tags=["AI Recommendations"])


@router.post("", response_model=RecommendationResponse)
def get_partner_recommendations(
    request: RecommendationRequest,
    db: Annotated[Session, Depends(get_db)],
):
    """
    Get ranked partner recommendations for a waste listing using MC-GNN / baseline AI model.
    Returns explainable partner cards with scores, distance, transport cost, and carbon benefit.
    """
    try:
        return recommendation_service.get_recommendations(db, request)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation engine error: {str(e)}",
        )
