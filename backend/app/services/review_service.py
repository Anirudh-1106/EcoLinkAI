"""
Review service — business logic for post-exchange reviews.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

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
    from app.models.plant import Plant
    from app.models.waste_listing import WasteListing

    # The response reports what was traded and between whom, so the whole
    # chain is loaded up front rather than lazily per row.
    query = db.query(Review).options(
        joinedload(Review.exchange)
        .joinedload(Exchange.exchange_request)
        .joinedload(ExchangeRequest.supplier_plant)
        .joinedload(Plant.company),
        joinedload(Review.exchange)
        .joinedload(Exchange.exchange_request)
        .joinedload(ExchangeRequest.buyer_plant)
        .joinedload(Plant.company),
        joinedload(Review.exchange)
        .joinedload(Exchange.exchange_request)
        .joinedload(ExchangeRequest.waste_listing)
        .joinedload(WasteListing.material),
    )

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


def role_in_exchange(
    db: Session, exchange_id: uuid.UUID, company_id: uuid.UUID
) -> str | None:
    """
    Which side of an exchange a company sits on: "supplier", "buyer" or None.

    None means the company was not party to it and has nothing to say about
    how it went.
    """
    from app.models.plant import Plant

    exchange = db.query(Exchange).filter(Exchange.id == exchange_id).first()
    if not exchange:
        return None

    req = (
        db.query(ExchangeRequest)
        .filter(ExchangeRequest.id == exchange.exchange_request_id)
        .first()
    )
    if not req:
        return None

    supplier = db.query(Plant).filter(Plant.id == req.supplier_plant_id).first()
    buyer = db.query(Plant).filter(Plant.id == req.buyer_plant_id).first()

    if supplier and supplier.company_id == company_id:
        return "supplier"
    if buyer and buyer.company_id == company_id:
        return "buyer"
    return None


def submit_review(
    db: Session, data: ReviewCreate, *, reviewer_company_id: uuid.UUID, role: str
) -> Review:
    """
    Record one party's rating of the other, and refresh trust scores.

    The role decides which half is written: a buyer rates the supplier, a
    supplier rates the buyer. Nobody writes their own.

    An exchange holds a single review row, so the second party to respond
    fills the half the first left empty rather than starting a new record.
    """
    review = get_reviews_for_exchange(db, data.exchange_id)
    if review is None:
        review = Review(exchange_id=data.exchange_id)
        db.add(review)

    if role == "buyer":
        # The buyer is rating the supplier.
        review.supplier_rating = data.rating
        review.supplier_feedback = data.feedback
    else:
        review.buyer_rating = data.rating
        review.buyer_feedback = data.feedback

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


def has_already_rated(review: Review | None, role: str) -> bool:
    """Whether this side has already given its rating."""
    if review is None:
        return False
    return (review.supplier_rating if role == "buyer" else review.buyer_rating) is not None
