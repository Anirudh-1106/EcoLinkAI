"""Plants router."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User, UserRole
from app.schemas.plant import PlantCreate, PlantListResponse, PlantResponse, PlantUpdate
from app.services import plant_service

router = APIRouter(prefix="/plants", tags=["Plants"])


@router.get("", response_model=PlantListResponse)
def list_plants(
    db: Annotated[Session, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    company_id: uuid.UUID | None = None,
    district: str | None = None,
):
    """List plants with optional company or district filter."""
    if company_id:
        items, total = plant_service.get_plants_by_company(db, company_id, page=page, page_size=page_size)
    else:
        items, total = plant_service.get_all_plants(db, page=page, page_size=page_size, district=district)

    response_items = []
    for p in items:
        resp = PlantResponse.model_validate(p)
        if p.company:
            resp.company_name = p.company.company_name
        response_items.append(resp)

    return PlantListResponse(items=response_items, total=total, page=page, page_size=page_size)


@router.get("/{plant_id}", response_model=PlantResponse)
def get_plant(
    plant_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
):
    """Get details of a plant."""
    plant = plant_service.get_plant(db, plant_id)
    if not plant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plant not found")
    resp = PlantResponse.model_validate(plant)
    if plant.company:
        resp.company_name = plant.company.company_name
    return resp


@router.post("", response_model=PlantResponse, status_code=status.HTTP_201_CREATED)
def create_plant(
    data: PlantCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Create a new plant under user's company."""
    if not current_user.company_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User is not associated with any company")

    company_id = current_user.company_id
    plant = plant_service.create_plant(db, company_id, data)
    resp = PlantResponse.model_validate(plant)
    if plant and getattr(plant, "company", None):
        resp.company_name = plant.company.company_name
    return resp


@router.put("/{plant_id}", response_model=PlantResponse)
def update_plant(
    plant_id: uuid.UUID,
    data: PlantUpdate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Update plant details."""
    plant = plant_service.get_plant(db, plant_id)
    if not plant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plant not found")

    if current_user.role != UserRole.ADMIN and current_user.company_id != plant.company_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    updated = plant_service.update_plant(db, plant_id, data)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plant update failed")

    resp = PlantResponse.model_validate(updated)
    if getattr(updated, "company", None):
        resp.company_name = updated.company.company_name
    return resp
