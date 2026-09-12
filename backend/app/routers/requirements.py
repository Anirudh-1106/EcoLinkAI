"""Requirements router."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.authorization import ensure_company_access, ensure_plant_access
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.requirement import (
    RequirementCreate,
    RequirementListResponse,
    RequirementResponse,
    RequirementUpdate,
)
from app.services import requirement_service

router = APIRouter(prefix="/requirements", tags=["Requirements"])


def _to_response(r) -> RequirementResponse:
    resp = RequirementResponse.model_validate(r)
    if r.plant:
        resp.plant_name = r.plant.plant_name
        if r.plant.company:
            resp.company_name = r.plant.company.company_name
    if r.material:
        resp.material_name = r.material.material_name
    return resp


@router.get("", response_model=RequirementListResponse)
def list_requirements(
    db: Annotated[Session, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    plant_id: uuid.UUID | None = None,
    company_id: uuid.UUID | None = None,
    material_id: uuid.UUID | None = None,
    status_filter: str | None = Query(None, alias="status"),
):
    """List material requirements with optional filters."""
    items, total = requirement_service.get_requirements(
        db,
        page=page,
        page_size=page_size,
        plant_id=plant_id,
        company_id=company_id,
        material_id=material_id,
        status=status_filter,
    )
    return RequirementListResponse(
        items=[_to_response(r) for r in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{requirement_id}", response_model=RequirementResponse)
def get_requirement(
    requirement_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
):
    """Get requirement details."""
    r = requirement_service.get_requirement(db, requirement_id)
    if not r:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Requirement not found")
    return _to_response(r)


@router.post("", response_model=RequirementResponse, status_code=status.HTTP_201_CREATED)
def create_requirement(
    data: RequirementCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Post a new material requirement for a plant the caller's company owns."""
    ensure_plant_access(db, current_user, data.plant_id)

    r = requirement_service.create_requirement(db, data)
    full_r = requirement_service.get_requirement(db, r.id)
    return _to_response(full_r or r)


@router.put("/{requirement_id}", response_model=RequirementResponse)
def update_requirement(
    requirement_id: uuid.UUID,
    data: RequirementUpdate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Update requirement status or details."""
    r = requirement_service.get_requirement(db, requirement_id)
    if not r:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Requirement not found")

    ensure_company_access(current_user, r.plant.company_id if r.plant else None)

    updated = requirement_service.update_requirement(db, requirement_id, data)
    full_r = requirement_service.get_requirement(db, requirement_id)
    return _to_response(full_r or updated)
