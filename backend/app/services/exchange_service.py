"""
Exchange service — business logic for exchange requests and exchanges.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy.orm import Session, joinedload

from app.enums.exchange import (
    ExchangeRequestStatus,
    ExchangeStatus,
    ShipmentStatus,
    WasteStatus,
)
from app.models.exchange import Exchange
from app.models.exchange_request import ExchangeRequest
from app.models.plant import Plant
from app.models.waste_listing import WasteListing
from app.schemas.exchange import (
    ExchangeCreate,
    ExchangeRequestCreate,
    ExchangeUpdate,
)


# ─── Exchange Requests ────────────────────────────────

def get_exchange_request(
    db: Session, request_id: uuid.UUID
) -> ExchangeRequest | None:
    """Get a single exchange request with relations."""
    return (
        db.query(ExchangeRequest)
        .options(
            joinedload(ExchangeRequest.supplier_plant).joinedload(Plant.company),
            joinedload(ExchangeRequest.buyer_plant).joinedload(Plant.company),
            joinedload(ExchangeRequest.waste_listing),
        )
        .filter(ExchangeRequest.id == request_id)
        .first()
    )


def get_exchange_requests(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    company_id: uuid.UUID | None = None,
    plant_id: uuid.UUID | None = None,
    status: str | None = None,
    role: str = "all",  # "supplier", "buyer", "all"
) -> tuple[list[ExchangeRequest], int]:
    """Get paginated exchange requests with filters."""
    query = db.query(ExchangeRequest)

    if company_id and role == "supplier":
        query = query.join(
            Plant, ExchangeRequest.supplier_plant_id == Plant.id
        ).filter(Plant.company_id == company_id)
    elif company_id and role == "buyer":
        query = query.join(
            Plant, ExchangeRequest.buyer_plant_id == Plant.id
        ).filter(Plant.company_id == company_id)
    elif company_id:
        # Both supplier and buyer
        supplier_plants = (
            db.query(Plant.id).filter(Plant.company_id == company_id).subquery()
        )
        query = query.filter(
            (ExchangeRequest.supplier_plant_id.in_(supplier_plants))
            | (ExchangeRequest.buyer_plant_id.in_(supplier_plants))
        )

    if plant_id:
        query = query.filter(
            (ExchangeRequest.supplier_plant_id == plant_id)
            | (ExchangeRequest.buyer_plant_id == plant_id)
        )

    if status:
        query = query.filter(ExchangeRequest.status == status)

    total = query.count()
    items = (
        query.options(
            joinedload(ExchangeRequest.supplier_plant).joinedload(Plant.company),
            joinedload(ExchangeRequest.buyer_plant).joinedload(Plant.company),
            joinedload(ExchangeRequest.waste_listing),
        )
        .order_by(ExchangeRequest.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def create_exchange_request(
    db: Session,
    data: ExchangeRequestCreate,
    *,
    supplier_plant_id: uuid.UUID,
    compatibility_score: float,
    ai_confidence_score: float,
    recommendation_rank: int,
    distance_km: float,
    estimated_transport_cost: float,
    estimated_carbon_emission: float,
    estimated_carbon_saving: float | None = None,
    recommendation_reason: str | None = None,
    material_compatibility: float | None = None,
    quantity_compatibility: float | None = None,
    quality_compatibility: float | None = None,
) -> ExchangeRequest:
    """Create an exchange request (called after recommendation selection)."""
    req = ExchangeRequest(
        supplier_plant_id=supplier_plant_id,
        buyer_plant_id=data.buyer_plant_id,
        waste_listing_id=data.waste_listing_id,
        requirement_id=data.requirement_id,
        # Recorded rather than dropped: the freight and carbon figures above
        # are computed from this amount, so without it a stored request cannot
        # be interpreted, let alone trained on.
        requested_quantity=data.requested_quantity,
        compatibility_score=compatibility_score,
        material_compatibility=material_compatibility,
        quantity_compatibility=quantity_compatibility,
        quality_compatibility=quality_compatibility,
        ai_confidence_score=ai_confidence_score,
        recommendation_rank=recommendation_rank,
        distance_km=distance_km,
        estimated_transport_cost=estimated_transport_cost,
        estimated_carbon_emission=estimated_carbon_emission,
        estimated_carbon_saving=estimated_carbon_saving,
        recommendation_reason=recommendation_reason,
        status=ExchangeRequestStatus.PENDING,
    )
    db.add(req)
    db.commit()
    db.refresh(req)
    return req


def accept_exchange_request(
    db: Session, request_id: uuid.UUID
) -> ExchangeRequest | None:
    """Accept an exchange request and create an exchange."""
    req = db.query(ExchangeRequest).filter(ExchangeRequest.id == request_id).first()
    if not req or req.status != ExchangeRequestStatus.PENDING:
        return None

    # Locked for update so two sellers accepting at once cannot both read the
    # same stock and between them commit more than exists.
    listing = (
        db.query(WasteListing)
        .filter(WasteListing.id == req.waste_listing_id)
        .with_for_update()
        .first()
    )

    agreed_price = Decimal("0")

    if listing:
        # A request may be for part of the listing. Draw down only what was
        # agreed and leave the remainder on the market; reserve the listing
        # solely once nothing is left, rather than taking a seller's whole
        # stock off the board because someone wanted a fraction of it.
        agreed = req.requested_quantity or listing.quantity
        if agreed > listing.quantity:
            raise ValueError(
                f"Cannot accept: {agreed} {listing.unit.value} requested but only "
                f"{listing.quantity} {listing.unit.value} remains"
            )

        # What the material itself costs: the quantity agreed, at the
        # listing's asking price. This was previously set to the transport
        # cost, so every completed exchange recorded the price of the lorry
        # rather than the price of the goods, and the analytics built on it
        # followed. Computed before the stock is drawn down, since afterwards
        # listing.quantity is what is left rather than what was bought.
        #
        # A listing with no asking price is open to negotiation and nothing
        # has been agreed yet, so it records zero rather than inventing a
        # figure.
        if listing.price_per_unit is not None:
            agreed_price = agreed * listing.price_per_unit

        listing.quantity = listing.quantity - agreed
        if listing.quantity <= 0:
            listing.status = WasteStatus.RESERVED

    req.status = ExchangeRequestStatus.ACCEPTED

    # Auto-create exchange
    exchange = Exchange(
        exchange_request_id=req.id,
        exchange_status=ExchangeStatus.INITIATED,
        shipment_status=ShipmentStatus.PENDING,
        agreed_price=agreed_price,
        transport_cost=req.estimated_transport_cost,
    )
    db.add(exchange)
    db.commit()
    db.refresh(req)
    return req


def reject_exchange_request(
    db: Session, request_id: uuid.UUID
) -> ExchangeRequest | None:
    """Reject an exchange request."""
    req = db.query(ExchangeRequest).filter(ExchangeRequest.id == request_id).first()
    if not req or req.status != ExchangeRequestStatus.PENDING:
        return None

    req.status = ExchangeRequestStatus.REJECTED
    db.commit()
    db.refresh(req)
    return req


# ─── Exchanges (Transactions) ────────────────────────

def get_exchange(db: Session, exchange_id: uuid.UUID) -> Exchange | None:
    """Get a single exchange with relations."""
    return (
        db.query(Exchange)
        .options(
            joinedload(Exchange.exchange_request)
            .joinedload(ExchangeRequest.supplier_plant)
            .joinedload(Plant.company),
            joinedload(Exchange.exchange_request)
            .joinedload(ExchangeRequest.buyer_plant)
            .joinedload(Plant.company),
            joinedload(Exchange.review),
        )
        .filter(Exchange.id == exchange_id)
        .first()
    )


def get_exchanges(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    company_id: uuid.UUID | None = None,
    status: str | None = None,
) -> tuple[list[Exchange], int]:
    """Get paginated exchanges."""
    query = db.query(Exchange).join(
        ExchangeRequest, Exchange.exchange_request_id == ExchangeRequest.id
    )

    if company_id:
        supplier_plants = (
            db.query(Plant.id).filter(Plant.company_id == company_id).subquery()
        )
        query = query.filter(
            (ExchangeRequest.supplier_plant_id.in_(supplier_plants))
            | (ExchangeRequest.buyer_plant_id.in_(supplier_plants))
        )

    if status:
        query = query.filter(Exchange.exchange_status == status)

    total = query.count()
    items = (
        query.options(
            joinedload(Exchange.exchange_request)
            .joinedload(ExchangeRequest.supplier_plant)
            .joinedload(Plant.company),
            joinedload(Exchange.exchange_request)
            .joinedload(ExchangeRequest.buyer_plant),
        )
        .order_by(Exchange.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def update_exchange(
    db: Session,
    exchange_id: uuid.UUID,
    data: ExchangeUpdate,
) -> Exchange | None:
    """Update exchange status/shipment/details."""
    exchange = db.query(Exchange).filter(Exchange.id == exchange_id).first()
    if not exchange:
        return None

    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(exchange, field, value)

    # If completed, mark listing as exchanged
    if data.exchange_status == ExchangeStatus.COMPLETED.value:
        req = db.query(ExchangeRequest).filter(
            ExchangeRequest.id == exchange.exchange_request_id
        ).first()
        if req:
            listing = db.query(WasteListing).filter(
                WasteListing.id == req.waste_listing_id
            ).first()
            if listing:
                listing.status = WasteStatus.EXCHANGED

    db.commit()
    db.refresh(exchange)
    return exchange
