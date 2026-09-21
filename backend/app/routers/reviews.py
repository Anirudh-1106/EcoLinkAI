"""Reviews router."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.authorization import ensure_company_access
from app.core.database import get_db
from app.enums.exchange import ExchangeStatus
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.review import ReviewCreate, ReviewListResponse, ReviewResponse
from app.services import exchange_service, review_service

router = APIRouter(prefix="/reviews", tags=["Reviews"])


def _to_response(review, *, role: str | None) -> ReviewResponse:
    """
    Build a review response, resolving what was actually traded.

    A rating on its own says little: two companies may have dealt with each
    other repeatedly, so the material and quantity are what tell the reader
    which exchange is being rated.
    """
    resp = ReviewResponse.model_validate(review)

    request = review.exchange.exchange_request if review.exchange else None
    if request:
        if request.supplier_plant:
            resp.supplier_plant_name = request.supplier_plant.plant_name
            if request.supplier_plant.company:
                resp.supplier_company_name = request.supplier_plant.company.company_name
        if request.buyer_plant:
            resp.buyer_plant_name = request.buyer_plant.plant_name
            if request.buyer_plant.company:
                resp.buyer_company_name = request.buyer_plant.company.company_name

        listing = request.waste_listing
        if listing:
            resp.unit = listing.unit.value if listing.unit else None
            if listing.material:
                resp.material_name = listing.material.material_name
        # What the deal was actually for, falling back to the whole listing
        # for requests raised before the quantity was recorded.
        resp.quantity = request.requested_quantity or (listing.quantity if listing else None)

    resp.my_role = role
    if role:
        resp.my_rating_submitted = review_service.has_already_rated(review, role)
        other = "supplier" if role == "buyer" else "buyer"
        resp.counterparty_rating_submitted = review_service.has_already_rated(review, other)

    return resp


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
        items=[
            _to_response(
                r,
                role=review_service.role_in_exchange(
                    db, r.exchange_id, current_user.company_id
                ),
            )
            for r in items
        ],
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

    # A review rates how an exchange actually went, so there has to be an
    # outcome to rate. Without this an in-transit exchange could be rated --
    # and since reviews move trust scores, that would let a party mark a
    # counterpart down before they had a chance to deliver.
    if exchange.exchange_status != ExchangeStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Only completed exchanges can be reviewed "
                f"(this one is '{exchange.exchange_status.value}')"
            ),
        )

    # Which side the caller is on decides which rating they may give. The
    # request never says, so a party cannot rate itself.
    role = review_service.role_in_exchange(
        db, data.exchange_id, current_user.company_id
    )
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the two companies involved in an exchange can review it",
        )

    existing = review_service.get_reviews_for_exchange(db, data.exchange_id)
    if review_service.has_already_rated(existing, role):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You have already rated this exchange",
        )

    review = review_service.submit_review(
        db, data, reviewer_company_id=current_user.company_id, role=role
    )
    return _to_response(review, role=role)
