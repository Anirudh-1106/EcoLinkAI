"""Reviews router."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.authorization import ensure_company_access
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.review import ReviewCreate, ReviewListResponse, ReviewResponse
from app.services import exchange_service, review_service

router = APIRouter(prefix="/reviews", tags=["Reviews"])


@router.get("", response_model=ReviewListResponse)
def list_reviews(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """List reviews belonging to the current user's company."""
    if not current_user.company_id:
        return ReviewListResponse(items=[], total=0, page=page, page_size=page_size)

    items, total = review_service.get_reviews(
        db, page=page, page_size=page_size, company_id=current_user.company_id
    )
    return ReviewListResponse(
        items=[ReviewResponse.model_validate(r) for r in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("", response_model=ReviewResponse, status_code=status.HTTP_201_CREATED)
def create_review(
    data: ReviewCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Submit a review for a completed exchange the caller took part in."""
    # Reviews move both parties' trust scores, which in turn feed partner
    # recommendations, so only the two companies involved may leave one.
    exchange = exchange_service.get_exchange(db, data.exchange_id)
    if not exchange:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Exchange not found",
        )

    req = exchange.exchange_request
    ensure_company_access(
        current_user,
        req.supplier_plant.company_id if req and req.supplier_plant else None,
        req.buyer_plant.company_id if req and req.buyer_plant else None,
    )

    existing = review_service.get_reviews_for_exchange(db, data.exchange_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review already exists for this exchange",
        )

    return review_service.create_review(db, data)
