"""Exchange requests router."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.exchange import (
    ExchangeRequestAction,
    ExchangeRequestCreate,
    ExchangeRequestResponse,
)
from app.services import exchange_service, waste_listing_service

router = APIRouter(prefix="/exchange-requests", tags=["Exchange Requests"])


def _req_to_response(r) -> ExchangeRequestResponse:
    resp = ExchangeRequestResponse.model_validate(r)
    if r.supplier_plant:
        resp.supplier_plant_name = r.supplier_plant.plant_name
        if r.supplier_plant.company:
            resp.supplier_company_name = r.supplier_plant.company.company_name
    if r.buyer_plant:
        resp.buyer_plant_name = r.buyer_plant.plant_name
        if r.buyer_plant.company:
            resp.buyer_company_name = r.buyer_plant.company.company_name
    if r.waste_listing:
        resp.waste_quantity = r.waste_listing.quantity
        if r.waste_listing.material:
            resp.material_name = r.waste_listing.material.material_name
    return resp


@router.get("", response_model=list[ExchangeRequestResponse])
def list_exchange_requests(
    db: Annotated[Session, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    company_id: uuid.UUID | None = None,
    plant_id: uuid.UUID | None = None,
    status_filter: str | None = Query(None, alias="status"),
    role: str = "all",
):
    """List exchange requests."""
    items, total = exchange_service.get_exchange_requests(
        db,
        page=page,
        page_size=page_size,
        company_id=company_id,
        plant_id=plant_id,
        status=status_filter,
        role=role,
    )
    return [_req_to_response(r) for r in items]


@router.get("/{request_id}", response_model=ExchangeRequestResponse)
def get_exchange_request(
    request_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
):
    """Get single exchange request details."""
    r = exchange_service.get_exchange_request(db, request_id)
    if not r:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found")
    return _req_to_response(r)


@router.post("", response_model=ExchangeRequestResponse, status_code=status.HTTP_201_CREATED)
def create_exchange_request(
    data: ExchangeRequestCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Create an exchange request for a recommended waste listing."""
    listing = waste_listing_service.get_waste_listing(db, data.waste_listing_id)
    if not listing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Waste listing not found")

    supplier_plant_id = listing.plant_id

    req = exchange_service.create_exchange_request(
        db,
        data,
        supplier_plant_id=supplier_plant_id,
        compatibility_score=95.0,  # Default fallback if direct request
        ai_confidence_score=90.0,
        recommendation_rank=1,
        distance_km=50.0,
        estimated_transport_cost=float(data.offered_price_per_unit or 100) * float(data.requested_quantity),
        estimated_carbon_emission=25.0,
        estimated_carbon_saving=float(data.requested_quantity) * 1.2,
        recommendation_reason="User selected partner recommendation",
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
    if action_data.action == "accept":
        req = exchange_service.accept_exchange_request(db, request_id)
    else:
        req = exchange_service.reject_exchange_request(db, request_id)

    if not req:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request cannot be processed (invalid status or not found)",
        )

    full_req = exchange_service.get_exchange_request(db, request_id)
    return _req_to_response(full_req or req)
