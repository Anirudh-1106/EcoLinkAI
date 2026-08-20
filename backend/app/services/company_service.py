"""
Company service — business logic for company operations.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from app.models.company import Company
from app.models.plant import Plant
from app.models.review import Review
from app.models.exchange import Exchange
from app.models.exchange_request import ExchangeRequest
from app.schemas.company import CompanyCreate, CompanyUpdate


def get_company(db: Session, company_id: uuid.UUID) -> Company | None:
    """Get a single company by ID."""
    return db.query(Company).filter(Company.id == company_id).first()


def get_companies(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    search: str | None = None,
    industry_type: str | None = None,
) -> tuple[list[Company], int]:
    """Get paginated list of companies with optional filters."""
    query = db.query(Company).filter(Company.is_active.is_(True))

    if search:
        query = query.filter(Company.company_name.ilike(f"%{search}%"))
    if industry_type:
        query = query.filter(Company.industry_type == industry_type)

    total = query.count()
    items = (
        query.order_by(Company.company_name)
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def create_company(db: Session, data: CompanyCreate) -> Company:
    """Create a new company."""
    company = Company(
        company_name=data.company_name,
        industry_type=data.industry_type,
        registration_number=data.registration_number,
        gst_number=data.gst_number,
        license_number=data.license_number,
    )
    db.add(company)
    db.commit()
    db.refresh(company)
    return company


def update_company(
    db: Session,
    company_id: uuid.UUID,
    data: CompanyUpdate,
) -> Company | None:
    """Update company fields."""
    company = get_company(db, company_id)
    if not company:
        return None

    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(company, field, value)

    db.commit()
    db.refresh(company)
    return company


def update_trust_score(db: Session, company_id: uuid.UUID) -> None:
    """
    Recalculate company trust score from review ratings
    of all plants belonging to this company.
    """
    company = get_company(db, company_id)
    if not company:
        return

    # Get average ratings from reviews involving this company's plants
    plant_ids = [p.id for p in company.plants]
    if not plant_ids:
        return

    # Average of supplier ratings when company is supplier
    avg_rating = (
        db.query(func.avg(Review.supplier_rating))
        .join(Exchange, Review.exchange_id == Exchange.id)
        .join(ExchangeRequest, Exchange.exchange_request_id == ExchangeRequest.id)
        .filter(ExchangeRequest.supplier_plant_id.in_(plant_ids))
        .scalar()
    )

    if avg_rating is not None:
        # Scale 1-5 rating to 0-100 trust score
        company.trust_score = float(avg_rating) * 20.0
        db.commit()
