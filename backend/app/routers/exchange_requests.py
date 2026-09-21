"""Exchange requests router."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.authorization import ensure_plant_access
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.exchange_request import ExchangeRequest
from app.models.user import User
from app.schemas.exchange import (
    ExchangeRequestAction,
    ExchangeRequestCreate,
    ExchangeRequestPreview,
    ExchangeRequestResponse,
)
from app.utils.quantity import to_kg
from app.services import exchange_service, recommendation_service, waste_listing_service

router = APIRouter(prefix="/exchange-requests", tags=["Exchange Requests"])


def _req_to_response(r) -> ExchangeRequestResponse:
    resp = ExchangeRequestResponse.model_validate(r)
    if r.supplier_plant:
        resp.supplier_plant_name = r.supplier_plant.plant_name
        if r.supplier_plant.company:
            resp.supplier_company_id = r.supplier_plant.company.id
            resp.supplier_company_name = r.supplier_plant.company.company_name
    if r.buyer_plant:
        resp.buyer_plant_name = r.buyer_plant.plant_name
        if r.buyer_plant.company:
            resp.buyer_company_id = r.buyer_plant.company.id
            resp.buyer_company_name = r.buyer_plant.company.company_name
    if r.waste_listing:
        resp.waste_quantity = r.waste_listing.quantity
        if r.waste_listing.material:
            resp.material_name = r.waste_listing.material.material_name
    return resp


def _user_owns_request(current_user: User, r) -> bool:
    if not current_user.company_id:
        return False
    supplier_company_id = r.supplier_plant.company_id if r.supplier_plant else None
    buyer_company_id = r.buyer_plant.company_id if r.buyer_plant else None
    return current_user.company_id in (supplier_company_id, buyer_company_id)


@router.get("", response_model=list[ExchangeRequestResponse])
def list_exchange_requests(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    plant_id: uuid.UUID | None = None,
    status_filter: str | None = Query(None, alias="status"),
    role: str = "all",
):
    """List exchange requests belonging to the current user's company."""
    if not current_user.company_id:
        return []

    items, total = exchange_service.get_exchange_requests(
        db,
        page=page,
        page_size=page_size,
        company_id=current_user.company_id,
        plant_id=plant_id,
        status=status_filter,
        role=role,
    )
    return [_req_to_response(r) for r in items]


@router.get("/{request_id}", response_model=ExchangeRequestResponse)
def get_exchange_request(
    request_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Get single exchange request details."""
    r = exchange_service.get_exchange_request(db, request_id)
    if not r:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found")
    if not _user_owns_request(current_user, r):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this request")
    return _req_to_response(r)


@router.post("/preview", response_model=ExchangeRequestPreview)
def preview_exchange_request(
    data: ExchangeRequestCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Cost and carbon for a given quantity, without creating anything.

    The buyer's quantity dialog reads its figures from here rather than
    recomputing them in the browser. Freight is priced through tapering
    volume brackets and carbon depends on the material, so a second
    implementation in TypeScript would drift from this one and quietly show
    buyers numbers the server disagrees with.
    """
    ensure_plant_access(db, current_user, data.buyer_plant_id)

    listing = waste_listing_service.get_waste_listing(db, data.waste_listing_id)
    if not listing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Waste listing not found")

    if data.requested_quantity > listing.quantity:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Requested {data.requested_quantity} {listing.unit.value} "
                f"but only {listing.quantity} {listing.unit.value} is available"
            ),
        )

    try:
        match = recommendation_service.score_candidate_pair(
            db,
            waste_listing_id=data.waste_listing_id,
            buyer_plant_id=data.buyer_plant_id,
            requirement_id=data.requirement_id,
            traded_quantity=float(data.requested_quantity),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    quantity_kg = to_kg(float(data.requested_quantity), listing.unit)
    total_price = (
        data.requested_quantity * listing.price_per_unit
        if listing.price_per_unit is not None
        else None
    )

    return ExchangeRequestPreview(
        requested_quantity=data.requested_quantity,
        unit=listing.unit.value,
        available_quantity=listing.quantity,
        quantity_kg=Decimal(str(round(quantity_kg, 2))) if quantity_kg is not None else None,
        estimated_transport_cost=Decimal(str(round(match["estimated_transport_cost"], 2))),
        estimated_carbon_emission=Decimal(str(round(match["estimated_carbon_emission"], 2))),
        estimated_carbon_saving=Decimal(str(round(match["estimated_carbon_saving"], 2))),
        estimated_total_price=total_price,
        distance_km=Decimal(str(round(match["distance_km"], 2))),
    )


@router.post("", response_model=ExchangeRequestResponse, status_code=status.HTTP_201_CREATED)
def create_exchange_request(
    data: ExchangeRequestCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Create an exchange request for a recommended waste listing."""
    # The caller must own the plant they are buying for, otherwise a company
    # could raise requests in another company's name.
    ensure_plant_access(db, current_user, data.buyer_plant_id)

    listing = waste_listing_service.get_waste_listing(db, data.waste_listing_id)
    if not listing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Waste listing not found")

    if db.query(ExchangeRequest).filter(
        ExchangeRequest.waste_listing_id == data.waste_listing_id,
        ExchangeRequest.requirement_id == data.requirement_id,
    ).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An exchange request already exists for this listing and requirement",
        )

    supplier_plant_id = listing.plant_id

    # Checked server-side because the browser's copy of the listing may be
    # stale, and because a client is free not to ask.
    if data.requested_quantity > listing.quantity:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Requested {data.requested_quantity} {listing.unit.value} "
                f"but only {listing.quantity} {listing.unit.value} is available"
            ),
        )

    try:
        match = recommendation_service.score_candidate_pair(
            db,
            waste_listing_id=data.waste_listing_id,
            buyer_plant_id=data.buyer_plant_id,
            requirement_id=data.requirement_id,
            traded_quantity=float(data.requested_quantity),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    try:
        req = exchange_service.create_exchange_request(
            db,
            data,
            supplier_plant_id=supplier_plant_id,
            compatibility_score=match["compatibility_score"],
            ai_confidence_score=match["ai_score"],
            recommendation_rank=data.recommendation_rank or 1,
            distance_km=match["distance_km"],
            estimated_transport_cost=match["estimated_transport_cost"],
            estimated_carbon_emission=match["estimated_carbon_emission"],
            estimated_carbon_saving=match["estimated_carbon_saving"],
            recommendation_reason=data.remarks,
            material_compatibility=match["material_compatibility"],
            quantity_compatibility=match["quantity_compatibility"],
            quality_compatibility=match["quality_compatibility"],
        )
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An exchange request already exists for this listing and requirement",
        )

    full_req = exchange_service.get_exchange_request(db, req.id)
    return _req_to_response(full_req or req)


@router.post("/{request_id}/action", response_model=ExchangeRequestResponse)
def handle_exchange_request_action(
    request_id: uuid.UUID,
    action_data: ExchangeRequestAction,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Accept or reject an exchange request."""
    existing = exchange_service.get_exchange_request(db, request_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found")
    supplier_company_id = existing.supplier_plant.company_id if existing.supplier_plant else None
    if not current_user.company_id or current_user.company_id != supplier_company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the seller (supplier company) can accept or reject this request",
        )

    if action_data.action == "accept":
        try:
            req = exchange_service.accept_exchange_request(db, request_id)
        except ValueError as e:
            # Stock ran out between the request being raised and accepted.
            db.rollback()
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    else:
        req = exchange_service.reject_exchange_request(db, request_id)

    if not req:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request cannot be processed (invalid status or not found)",
        )

    full_req = exchange_service.get_exchange_request(db, request_id)
    return _req_to_response(full_req or req)
