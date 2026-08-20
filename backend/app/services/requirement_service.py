"""
Requirement service — business logic for material requirements.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session, joinedload

from app.enums.exchange import RequirementStatus
from app.models.plant import Plant
from app.models.requirement import Requirement
from app.schemas.requirement import RequirementCreate, RequirementUpdate


def get_requirement(db: Session, requirement_id: uuid.UUID) -> Requirement | None:
    """Get a single requirement with relations."""
    return (
        db.query(Requirement)
        .options(
            joinedload(Requirement.plant).joinedload(Plant.company),
            joinedload(Requirement.material),
        )
        .filter(Requirement.id == requirement_id)
        .first()
    )


def get_requirements(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    plant_id: uuid.UUID | None = None,
    company_id: uuid.UUID | None = None,
    material_id: uuid.UUID | None = None,
    status: str | None = None,
) -> tuple[list[Requirement], int]:
    """Get paginated requirements with filters."""
    query = db.query(Requirement).join(Plant, Requirement.plant_id == Plant.id)

    if plant_id:
        query = query.filter(Requirement.plant_id == plant_id)
    if company_id:
        query = query.filter(Plant.company_id == company_id)
    if material_id:
        query = query.filter(Requirement.material_id == material_id)
    if status:
        query = query.filter(Requirement.status == status)

    total = query.count()
    items = (
        query.options(
            joinedload(Requirement.plant).joinedload(Plant.company),
            joinedload(Requirement.material),
        )
        .order_by(Requirement.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def create_requirement(db: Session, data: RequirementCreate) -> Requirement:
    """Create a new material requirement."""
    requirement = Requirement(
        plant_id=data.plant_id,
        material_id=data.material_id,
        quantity=data.quantity,
        unit=data.unit,
        minimum_purity=data.minimum_purity,
        maximum_budget_per_unit=data.maximum_budget_per_unit,
        required_before=data.required_before,
        preferred_max_distance_km=data.preferred_max_distance_km,
        notes=data.notes,
        status=RequirementStatus.OPEN,
    )
    db.add(requirement)
    db.commit()
    db.refresh(requirement)
    return requirement


def update_requirement(
    db: Session,
    requirement_id: uuid.UUID,
    data: RequirementUpdate,
) -> Requirement | None:
    """Update requirement fields."""
    req = db.query(Requirement).filter(Requirement.id == requirement_id).first()
    if not req:
        return None

    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(req, field, value)

    db.commit()
    db.refresh(req)
    return req
