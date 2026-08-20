"""Exchanges (Transactions) router."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.exchange import (
    ExchangeListResponse,
    ExchangeResponse,
    ExchangeUpdate,
)
from app.services import exchange_service

router = APIRouter(prefix="/exchanges", tags=["Exchanges / Transactions"])


def _ex_to_response(e) -> ExchangeResponse:
    resp = ExchangeResponse.model_validate(e)
    if e.exchange_request:
        req = e.exchange_request
        if req.supplier_plant:
            resp.supplier_plant_name = req.supplier_plant.plant_name
        if req.buyer_plant:
            resp.buyer_plant_name = req.buyer_plant.plant_name
        if req.waste_listing and req.waste_listing.material:
            resp.material_name = req.waste_listing.material.material_name
    return resp


@router.get("", response_model=ExchangeListResponse)
def list_exchanges(
    db: Annotated[Session, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    company_id: uuid.UUID | None = None,
    status_filter: str | None = Query(None, alias="status"),
):
    """List exchanges with optional filters."""
    items, total = exchange_service.get_exchanges(
        db, page=page, page_size=page_size, company_id=company_id, status=status_filter
    )
    return ExchangeListResponse(
        items=[_ex_to_response(e) for e in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{exchange_id}", response_model=ExchangeResponse)
def get_exchange(
    exchange_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
):
    """Get single exchange details."""
    e = exchange_service.get_exchange(db, exchange_id)
    if not e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exchange not found")
    return _ex_to_response(e)


@router.put("/{exchange_id}", response_model=ExchangeResponse)
def update_exchange(
    exchange_id: uuid.UUID,
    data: ExchangeUpdate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Update exchange lifecycle or shipment status."""
    e = exchange_service.get_exchange(db, exchange_id)
    if not e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exchange not found")

    updated = exchange_service.update_exchange(db, exchange_id, data)
    full_e = exchange_service.get_exchange(db, exchange_id)
    return _ex_to_response(full_e or updated)
