"""
Plant service — business logic for plant operations.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.models.plant import Plant
from app.models.company import Company
from app.schemas.plant import PlantCreate, PlantUpdate


def get_plant(db: Session, plant_id: uuid.UUID) -> Plant | None:
    """Get a single plant by ID."""
    return db.query(Plant).filter(Plant.id == plant_id).first()


def get_plants_by_company(
    db: Session,
    company_id: uuid.UUID,
    *,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[Plant], int]:
    """Get paginated plants for a specific company."""
    query = db.query(Plant).filter(
        Plant.company_id == company_id,
        Plant.is_active.is_(True),
    )
    total = query.count()
    items = (
        query.order_by(Plant.plant_name)
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def get_all_plants(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    district: str | None = None,
) -> tuple[list[Plant], int]:
    """Get all active plants with optional filters."""
    query = db.query(Plant).filter(Plant.is_active.is_(True))
    if district:
        query = query.filter(Plant.district == district)
    total = query.count()
    items = (
        query.order_by(Plant.plant_name)
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def create_plant(
    db: Session,
    company_id: uuid.UUID,
    data: PlantCreate,
) -> Plant:
    """Create a new plant under a company."""
    plant = Plant(
        company_id=company_id,
        plant_name=data.plant_name,
        plant_type=data.plant_type,
        address=data.address,
        district=data.district,
        state=data.state,
        country=data.country,
        latitude=data.latitude,
        longitude=data.longitude,
    )
    db.add(plant)
    db.commit()
    db.refresh(plant)
    return plant


def update_plant(
    db: Session,
    plant_id: uuid.UUID,
    data: PlantUpdate,
) -> Plant | None:
    """Update plant fields."""
    plant = get_plant(db, plant_id)
    if not plant:
        return None

    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(plant, field, value)

    db.commit()
    db.refresh(plant)
    return plant
