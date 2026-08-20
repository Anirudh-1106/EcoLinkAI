"""
Analytics service — aggregation queries for dashboards.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy import func, case, extract
from sqlalchemy.orm import Session

from app.enums.exchange import (
    ExchangeRequestStatus,
    ExchangeStatus,
    RequirementStatus,
    WasteStatus,
)
from app.models.company import Company
from app.models.exchange import Exchange
from app.models.exchange_request import ExchangeRequest
from app.models.material import Material
from app.models.plant import Plant
from app.models.requirement import Requirement
from app.models.review import Review
from app.models.waste_listing import WasteListing
from app.schemas.analytics import (
    DashboardStats,
    MaterialDistribution,
    MonthlyTrend,
    PlatformAnalytics,
)


def get_company_dashboard(db: Session, company_id: uuid.UUID) -> DashboardStats:
    """Get dashboard statistics for a specific company."""
    plant_ids = [
        r[0] for r in
        db.query(Plant.id).filter(Plant.company_id == company_id).all()
    ]

    if not plant_ids:
        return DashboardStats()

    total_plants = len(plant_ids)

    active_listings = (
        db.query(func.count(WasteListing.id))
        .filter(
            WasteListing.plant_id.in_(plant_ids),
            WasteListing.status == WasteStatus.AVAILABLE,
        )
        .scalar()
        or 0
    )

    open_requirements = (
        db.query(func.count(Requirement.id))
        .filter(
            Requirement.plant_id.in_(plant_ids),
            Requirement.status == RequirementStatus.OPEN,
        )
        .scalar()
        or 0
    )

    pending_requests = (
        db.query(func.count(ExchangeRequest.id))
        .filter(
            (ExchangeRequest.supplier_plant_id.in_(plant_ids))
            | (ExchangeRequest.buyer_plant_id.in_(plant_ids)),
            ExchangeRequest.status == ExchangeRequestStatus.PENDING,
        )
        .scalar()
        or 0
    )

    completed_exchanges = (
        db.query(func.count(Exchange.id))
        .join(ExchangeRequest, Exchange.exchange_request_id == ExchangeRequest.id)
        .filter(
            (ExchangeRequest.supplier_plant_id.in_(plant_ids))
            | (ExchangeRequest.buyer_plant_id.in_(plant_ids)),
            Exchange.exchange_status == ExchangeStatus.COMPLETED,
        )
        .scalar()
        or 0
    )

    carbon_saved = (
        db.query(func.coalesce(func.sum(Exchange.actual_carbon_saving), 0))
        .join(ExchangeRequest, Exchange.exchange_request_id == ExchangeRequest.id)
        .filter(
            (ExchangeRequest.supplier_plant_id.in_(plant_ids))
            | (ExchangeRequest.buyer_plant_id.in_(plant_ids)),
        )
        .scalar()
        or Decimal("0.00")
    )

    company = db.query(Company).filter(Company.id == company_id).first()
    trust = company.trust_score if company else 0.0

    avg_rating = (
        db.query(func.avg(Review.supplier_rating))
        .join(Exchange, Review.exchange_id == Exchange.id)
        .join(ExchangeRequest, Exchange.exchange_request_id == ExchangeRequest.id)
        .filter(ExchangeRequest.supplier_plant_id.in_(plant_ids))
        .scalar()
    )

    return DashboardStats(
        total_plants=total_plants,
        active_waste_listings=active_listings,
        open_requirements=open_requirements,
        pending_exchange_requests=pending_requests,
        completed_exchanges=completed_exchanges,
        total_carbon_saved_kg=carbon_saved,
        trust_score=trust,
        average_rating=float(avg_rating) if avg_rating else 0.0,
    )


def get_platform_analytics(db: Session) -> PlatformAnalytics:
    """Get platform-wide analytics for admin dashboard."""
    total_companies = db.query(func.count(Company.id)).scalar() or 0
    total_plants = db.query(func.count(Plant.id)).scalar() or 0
    total_materials = db.query(func.count(Material.id)).scalar() or 0
    total_listings = db.query(func.count(WasteListing.id)).scalar() or 0
    total_requirements = db.query(func.count(Requirement.id)).scalar() or 0
    total_requests = db.query(func.count(ExchangeRequest.id)).scalar() or 0
    total_exchanges = db.query(func.count(Exchange.id)).scalar() or 0
    total_reviews = db.query(func.count(Review.id)).scalar() or 0

    waste_exchanged = (
        db.query(func.coalesce(func.sum(Exchange.actual_quantity), 0))
        .filter(Exchange.exchange_status == ExchangeStatus.COMPLETED)
        .scalar()
        or Decimal("0.00")
    )

    carbon_saved = (
        db.query(func.coalesce(func.sum(Exchange.actual_carbon_saving), 0))
        .scalar()
        or Decimal("0.00")
    )

    revenue = (
        db.query(func.coalesce(func.sum(Exchange.agreed_price), 0))
        .filter(Exchange.exchange_status == ExchangeStatus.COMPLETED)
        .scalar()
        or Decimal("0.00")
    )

    avg_rating = db.query(func.avg(Review.supplier_rating)).scalar()

    accepted = (
        db.query(func.count(ExchangeRequest.id))
        .filter(ExchangeRequest.status == ExchangeRequestStatus.ACCEPTED)
        .scalar()
        or 0
    )
    success_rate = (accepted / total_requests * 100) if total_requests > 0 else 0

    return PlatformAnalytics(
        total_companies=total_companies,
        total_plants=total_plants,
        total_materials=total_materials,
        total_waste_listings=total_listings,
        total_requirements=total_requirements,
        total_exchange_requests=total_requests,
        total_exchanges=total_exchanges,
        total_reviews=total_reviews,
        total_waste_exchanged_tons=waste_exchanged / 1000,
        total_carbon_saved_kg=carbon_saved,
        total_revenue=revenue,
        average_rating=float(avg_rating) if avg_rating else 0.0,
        exchange_success_rate=success_rate,
    )


def get_material_distribution(db: Session) -> list[MaterialDistribution]:
    """Get waste listing distribution by material category."""
    rows = (
        db.query(
            Material.material_category,
            func.count(WasteListing.id),
            func.coalesce(func.sum(WasteListing.quantity), 0),
        )
        .join(WasteListing, Material.id == WasteListing.material_id)
        .group_by(Material.material_category)
        .order_by(func.count(WasteListing.id).desc())
        .all()
    )
    return [
        MaterialDistribution(
            category=row[0].value if hasattr(row[0], 'value') else str(row[0]),
            count=row[1],
            total_quantity=row[2],
        )
        for row in rows
    ]
