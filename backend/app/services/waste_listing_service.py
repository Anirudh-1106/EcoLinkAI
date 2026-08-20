"""
Waste listing service — business logic for waste listing operations.
"""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import and_
from sqlalchemy.orm import Session, joinedload

from app.enums.exchange import WasteStatus
from app.models.material import Material
from app.models.plant import Plant
from app.models.company import Company
from app.models.waste_listing import WasteListing
from app.schemas.waste_listing import WasteListingCreate, WasteListingUpdate


def get_waste_listing(db: Session, listing_id: uuid.UUID) -> WasteListing | None:
    """Get a single waste listing with eager-loaded relations."""
    return (
        db.query(WasteListing)
        .options(
            joinedload(WasteListing.plant).joinedload(Plant.company),
            joinedload(WasteListing.material),
        )
        .filter(WasteListing.id == listing_id)
        .first()
    )


def get_waste_listings(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    plant_id: uuid.UUID | None = None,
    company_id: uuid.UUID | None = None,
    material_id: uuid.UUID | None = None,
    status: str | None = None,
    available_only: bool = False,
) -> tuple[list[WasteListing], int]:
    """Get paginated waste listings with filters."""
    query = (
        db.query(WasteListing)
        .join(Plant, WasteListing.plant_id == Plant.id)
        .join(Material, WasteListing.material_id == Material.id)
    )

    if plant_id:
        query = query.filter(WasteListing.plant_id == plant_id)
    if company_id:
        query = query.filter(Plant.company_id == company_id)
    if material_id:
        query = query.filter(WasteListing.material_id == material_id)
    if status:
        query = query.filter(WasteListing.status == status)
    if available_only:
        query = query.filter(
            WasteListing.status == WasteStatus.AVAILABLE,
            WasteListing.available_until >= date.today(),
        )

    total = query.count()
    items = (
        query.options(
            joinedload(WasteListing.plant).joinedload(Plant.company),
            joinedload(WasteListing.material),
        )
        .order_by(WasteListing.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def create_waste_listing(db: Session, data: WasteListingCreate) -> WasteListing:
    """Create a new waste listing."""
    listing = WasteListing(
        plant_id=data.plant_id,
        material_id=data.material_id,
        description=data.description,
        quantity=data.quantity,
        unit=data.unit,
        purity_percentage=data.purity_percentage,
        moisture_percentage=data.moisture_percentage,
        quality_grade=data.quality_grade,
        price_per_unit=data.price_per_unit,
        available_from=data.available_from,
        available_until=data.available_until,
        status=WasteStatus.AVAILABLE,
    )
    db.add(listing)
    db.commit()
    db.refresh(listing)
    return listing


def update_waste_listing(
    db: Session,
    listing_id: uuid.UUID,
    data: WasteListingUpdate,
) -> WasteListing | None:
    """Update waste listing fields."""
    listing = db.query(WasteListing).filter(WasteListing.id == listing_id).first()
    if not listing:
        return None

    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(listing, field, value)

    db.commit()
    db.refresh(listing)
    return listing
