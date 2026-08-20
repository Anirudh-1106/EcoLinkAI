"""
Material service — business logic for material operations.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.models.material import Material
from app.schemas.material import MaterialCreate, MaterialUpdate


def get_material(db: Session, material_id: uuid.UUID) -> Material | None:
    """Get a single material by ID."""
    return db.query(Material).filter(Material.id == material_id).first()


def get_materials(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 50,
    search: str | None = None,
    category: str | None = None,
) -> tuple[list[Material], int]:
    """Get paginated materials with optional filters."""
    query = db.query(Material).filter(Material.is_active.is_(True))

    if search:
        query = query.filter(Material.material_name.ilike(f"%{search}%"))
    if category:
        query = query.filter(Material.material_category == category)

    total = query.count()
    items = (
        query.order_by(Material.material_name)
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def create_material(db: Session, data: MaterialCreate) -> Material:
    """Create a new material."""
    material = Material(
        material_name=data.material_name,
        material_category=data.material_category,
        hazard_class=data.hazard_class,
        default_density=data.default_density,
        carbon_factor=data.carbon_factor,
        description=data.description,
    )
    db.add(material)
    db.commit()
    db.refresh(material)
    return material


def update_material(
    db: Session,
    material_id: uuid.UUID,
    data: MaterialUpdate,
) -> Material | None:
    """Update material fields."""
    material = get_material(db, material_id)
    if not material:
        return None

    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(material, field, value)

    db.commit()
    db.refresh(material)
    return material


def get_material_categories(db: Session) -> list[str]:
    """Get distinct material categories."""
    rows = (
        db.query(Material.material_category)
        .distinct()
        .order_by(Material.material_category)
        .all()
    )
    return [r[0] for r in rows]
