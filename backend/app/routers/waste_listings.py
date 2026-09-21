"""Waste listings router."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.authorization import ensure_company_access, ensure_plant_access
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User, UserRole
from app.schemas.waste_listing import (
    WasteListingCreate,
    WasteListingListResponse,
    WasteListingResponse,
    WasteListingUpdate,
)
from app.services import waste_listing_service

router = APIRouter(prefix="/waste-listings", tags=["Waste Listings"])


def _to_response(w) -> WasteListingResponse:
    resp = WasteListingResponse.model_validate(w)
    if w.plant:
        resp.plant_name = w.plant.plant_name
        resp.plant_district = w.plant.district
        resp.plant_latitude = w.plant.latitude
        resp.plant_longitude = w.plant.longitude
        if w.plant.company:
            resp.company_name = w.plant.company.company_name
            resp.company_id = w.plant.company.id
    if w.material:
        resp.material_name = w.material.material_name
    return resp


@router.get("", response_model=WasteListingListResponse)
def list_waste_listings(
    db: Annotated[Session, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    plant_id: uuid.UUID | None = None,
    company_id: uuid.UUID | None = None,
    material_id: uuid.UUID | None = None,
    status_filter: str | None = Query(None, alias="status"),
    available_only: bool = False,
):
    """List waste listings with optional filters."""
    items, total = waste_listing_service.get_waste_listings(
        db,
        page=page,
        page_size=page_size,
        plant_id=plant_id,
        company_id=company_id,
        material_id=material_id,
        status=status_filter,
        available_only=available_only,
    )
    return WasteListingListResponse(
        items=[_to_response(w) for w in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{listing_id}", response_model=WasteListingResponse)
def get_waste_listing(
    listing_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
):
    """Get single waste listing details."""
    w = waste_listing_service.get_waste_listing(db, listing_id)
    if not w:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found")
    return _to_response(w)


@router.post("", response_model=WasteListingResponse, status_code=status.HTTP_201_CREATED)
def create_waste_listing(
    data: WasteListingCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Create a new waste listing on a plant the caller's company owns."""
    ensure_plant_access(db, current_user, data.plant_id)

    w = waste_listing_service.create_waste_listing(db, data)
    full_w = waste_listing_service.get_waste_listing(db, w.id)
    return _to_response(full_w or w)


@router.put("/{listing_id}", response_model=WasteListingResponse)
def update_waste_listing(
    listing_id: uuid.UUID,
    data: WasteListingUpdate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Update waste listing details."""
    w = waste_listing_service.get_waste_listing(db, listing_id)
    if not w:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found")

    ensure_company_access(current_user, w.plant.company_id if w.plant else None)

    # Checked against the stored start date, which the update body does not
    # carry, so the schema cannot do it alone. Without this the database
    # constraint rejects it as a 500 instead of a readable error.
    if data.available_until is not None and data.available_until < w.available_from:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"available_until ({data.available_until}) is before this listing's "
                f"available_from ({w.available_from})"
            ),
        )

    updated = waste_listing_service.update_waste_listing(db, listing_id, data)
    full_w = waste_listing_service.get_waste_listing(db, listing_id)
    return _to_response(full_w or updated)
