"""
Ownership authorization helpers.

Company data on this platform hangs off the plants a company owns, so most
write authorization reduces to "does the caller's company own this plant".
Centralising it here keeps every endpoint enforcing the same rule instead of
re-deriving it, and makes a missing check obvious by its absence.

Admins bypass these checks, matching the convention already used by the
plants router.
"""

from __future__ import annotations

import uuid

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.plant import Plant
from app.models.user import User, UserRole


def is_admin(user: User) -> bool:
    """Whether the user is a platform administrator."""
    return user.role == UserRole.ADMIN


def ensure_company_access(user: User, *owner_company_ids: uuid.UUID | None) -> None:
    """
    Raise 403 unless the caller's company is one of the resource's owners.

    Accepts several owners so two-party records (an exchange has both a
    seller and a buyer) can be authorised with the same call.
    """
    if is_admin(user):
        return

    owners = {owner for owner in owner_company_ids if owner is not None}
    if not user.company_id or user.company_id not in owners:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access another company's data",
        )


def ensure_plant_access(db: Session, user: User, plant_id: uuid.UUID) -> Plant:
    """
    Load a plant, rejecting it if the caller's company does not own it.

    Raises 404 when the plant does not exist and 403 when it belongs to
    another company. Returns the plant so callers can reuse it.
    """
    plant = db.query(Plant).filter(Plant.id == plant_id).first()
    if not plant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Plant not found",
        )

    ensure_company_access(user, plant.company_id)
    return plant
