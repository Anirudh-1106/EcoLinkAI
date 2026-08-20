"""
Review service — business logic for post-exchange reviews.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.exchange import Exchange
from app.models.exchange_request import ExchangeRequest
from app.models.review import Review
from app.schemas.review import ReviewCreate
from app.services import company_service


def get_review(db: Session, review_id: uuid.UUID) -> Review | None:
    """Get a single review by ID."""
    return db.query(Review).filter(Review.id == review_id).first()


def get_reviews_for_exchange(db: Session, exchange_id: uuid.UUID) -> Review | None:
    """Get review for a specific exchange."""
    return db.query(Review).filter(Review.exchange_id == exchange_id).first()


def get_reviews(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    company_id: uuid.UUID | None = None,
) -> tuple[list[Review], int]:
    """Get paginated reviews."""
    query = db.query(Review)

    if company_id:
        from app.models.plant import Plant
        plant_ids = (
            db.query(Plant.id).filter(Plant.company_id == company_id).subquery()
        )
        query = query.join(Exchange, Review.exchange_id == Exchange.id).join(
            ExchangeRequest, Exchange.exchange_request_id == ExchangeRequest.id
        ).filter(
            (ExchangeRequest.supplier_plant_id.in_(plant_ids))
            | (ExchangeRequest.buyer_plant_id.in_(plant_ids))
        )

    total = query.count()
    items = (
        query.order_by(Review.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def create_review(db: Session, data: ReviewCreate) -> Review:
    """Create a review and update trust scores."""
    review = Review(
        exchange_id=data.exchange_id,
        supplier_rating=data.supplier_rating,
        buyer_rating=data.buyer_rating,
        supplier_feedback=data.supplier_feedback,
        buyer_feedback=data.buyer_feedback,
    )
    db.add(review)
    db.commit()
    db.refresh(review)

    # Update trust scores for both parties
    exchange = db.query(Exchange).filter(Exchange.id == data.exchange_id).first()
    if exchange:
        req = db.query(ExchangeRequest).filter(
            ExchangeRequest.id == exchange.exchange_request_id
        ).first()
        if req:
            from app.models.plant import Plant
            supplier_plant = db.query(Plant).filter(Plant.id == req.supplier_plant_id).first()
            buyer_plant = db.query(Plant).filter(Plant.id == req.buyer_plant_id).first()
            if supplier_plant:
                company_service.update_trust_score(db, supplier_plant.company_id)
            if buyer_plant:
                company_service.update_trust_score(db, buyer_plant.company_id)

    return review
