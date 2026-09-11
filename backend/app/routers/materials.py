"""Materials router."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user, require_role
from app.models.user import User, UserRole
from app.schemas.material import (
    MaterialCreate,
    MaterialListResponse,
    MaterialResponse,
    MaterialUpdate,
)
from app.services import material_service

router = APIRouter(prefix="/materials", tags=["Materials"])


@router.get("", response_model=MaterialListResponse)
def list_materials(
    db: Annotated[Session, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: str | None = None,
    category: str | None = None,
):
    """List materials with optional category and search filters."""
    items, total = material_service.get_materials(
        db, page=page, page_size=page_size, search=search, category=category
    )
    return MaterialListResponse(
        items=[MaterialResponse.model_validate(m) for m in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/categories", response_model=list[str])
def get_categories(db: Annotated[Session, Depends(get_db)]):
    """Get list of distinct material categories."""
    return material_service.get_material_categories(db)


@router.get("/autocomplete")
def autocomplete_materials(
    db: Annotated[Session, Depends(get_db)],
    q: str = Query("", min_length=2, max_length=100),
):
    """Autocomplete material names for search. Returns up to 10 matching names."""
    import re
    from app.models.material import Material
    
    # Fetch a larger pool since we will deduplicate in Python
    results = (
        db.query(Material.material_name)
        .filter(
            Material.material_name.ilike(f"%{q}%"),
            Material.is_active.is_(True),
        )
        .limit(100)
        .all()
    )
    
    # Clean the names by removing synthetic (M###) suffixes and deduplicate
    unique_suggestions = []
    seen = set()
    
    for r in results:
        raw_name = r[0]
        # Strip out suffixes like " (M005)", " (M114)", etc.
        clean_name = re.sub(r'\s*\([A-Z0-9]+\)$', '', raw_name).strip()
        
        if clean_name not in seen:
            seen.add(clean_name)
            unique_suggestions.append(clean_name)
            
        if len(unique_suggestions) >= 10:
            break
            
    return {"suggestions": unique_suggestions}



@router.get("/{material_id}", response_model=MaterialResponse)
def get_material(
    material_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
):
    """Get material by ID."""
    mat = material_service.get_material(db, material_id)
    if not mat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")
    return mat


@router.post("", response_model=MaterialResponse, status_code=status.HTTP_201_CREATED)
def create_material(
    data: MaterialCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_role(UserRole.ADMIN, UserRole.COMPANY))],
):
    """Create a new material."""
    return material_service.create_material(db, data)
