from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Integer,
    Numeric,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.plant import Plant


class Analytics(BaseModel):
    """
    Stores aggregated analytics for an industrial plant.

    Analytics are updated as exchanges are completed and are
    used for dashboards, reporting, and AI insights.
    """

    __tablename__ = "analytics"

    __table_args__ = (
        CheckConstraint(
            "total_waste_generated >= 0",
            name="ck_analytics_waste_generated",
        ),
        CheckConstraint(
            "total_waste_exchanged >= 0",
            name="ck_analytics_waste_exchanged",
        ),
        CheckConstraint(
            "total_exchanges_completed >= 0",
            name="ck_analytics_exchanges_completed",
        ),
        CheckConstraint(
            "total_revenue_generated >= 0",
            name="ck_analytics_revenue",
        ),
        CheckConstraint(
            "total_transport_cost >= 0",
            name="ck_analytics_transport_cost",
        ),
        CheckConstraint(
            "total_carbon_emission >= 0",
            name="ck_analytics_carbon_emission",
        ),
        CheckConstraint(
            "total_carbon_saved >= 0",
            name="ck_analytics_carbon_saved",
        ),
        CheckConstraint(
            "average_ai_match_score BETWEEN 0 AND 100",
            name="ck_analytics_ai_score",
        ),
        CheckConstraint(
            "average_supplier_rating BETWEEN 0 AND 5",
            name="ck_analytics_supplier_rating",
        ),
        CheckConstraint(
            "average_buyer_rating BETWEEN 0 AND 5",
            name="ck_analytics_buyer_rating",
        ),
    )

    # =====================================================
    # Foreign Key
    # =====================================================

    plant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("plants.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # =====================================================
    # Analytics
    # =====================================================

    total_waste_generated: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        default=Decimal("0.00"),
        nullable=False,
    )

    total_waste_exchanged: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        default=Decimal("0.00"),
        nullable=False,
    )

    total_exchanges_completed: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    total_revenue_generated: Mapped[Decimal] = mapped_column(
        Numeric(14, 2),
        default=Decimal("0.00"),
        nullable=False,
    )

    total_transport_cost: Mapped[Decimal] = mapped_column(
        Numeric(14, 2),
        default=Decimal("0.00"),
        nullable=False,
    )

    total_carbon_emission: Mapped[Decimal] = mapped_column(
        Numeric(14, 2),
        default=Decimal("0.00"),
        nullable=False,
    )

    total_carbon_saved: Mapped[Decimal] = mapped_column(
        Numeric(14, 2),
        default=Decimal("0.00"),
        nullable=False,
    )

    average_ai_match_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 2),
        default=Decimal("0.00"),
        nullable=False,
    )

    average_supplier_rating: Mapped[Decimal] = mapped_column(
        Numeric(3, 2),
        default=Decimal("0.00"),
        nullable=False,
    )

    average_buyer_rating: Mapped[Decimal] = mapped_column(
        Numeric(3, 2),
        default=Decimal("0.00"),
        nullable=False,
    )

    # =====================================================
    # Relationships
    # =====================================================

    plant: Mapped["Plant"] = relationship(
        back_populates="analytics",
    )

    def __repr__(self) -> str:
        return (
            f"<Analytics("
            f"id={self.id}, "
            f"plant_id={self.plant_id}, "
            f"completed_exchanges={self.total_exchanges_completed}"
            f")>"
        )