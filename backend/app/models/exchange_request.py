from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    Enum,
    ForeignKey,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.enums.exchange import ExchangeRequestStatus
from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.exchange import Exchange
    from app.models.plant import Plant
    from app.models.requirement import Requirement
    from app.models.waste_listing import WasteListing


class ExchangeRequest(BaseModel):
    """
    Represents an AI-generated match between a waste listing
    and a material requirement.

    An exchange request stores recommendation metrics and
    serves as the approval workflow before an exchange is
    finalized.
    """

    __tablename__ = "exchange_requests"

    __table_args__ = (
        UniqueConstraint(
            "waste_listing_id",
            "requirement_id",
            name="uq_exchange_request_match",
        ),
        CheckConstraint(
            "supplier_plant_id <> buyer_plant_id",
            name="ck_exchange_request_different_plants",
        ),
        CheckConstraint(
            "compatibility_score BETWEEN 0 AND 100",
            name="ck_exchange_request_compatibility",
        ),
        CheckConstraint(
            "ai_confidence_score BETWEEN 0 AND 100",
            name="ck_exchange_request_ai_confidence",
        ),
        CheckConstraint(
            "recommendation_rank > 0",
            name="ck_exchange_request_rank",
        ),
        CheckConstraint(
            "distance_km >= 0",
            name="ck_exchange_request_distance",
        ),
        CheckConstraint(
            "estimated_transport_cost >= 0",
            name="ck_exchange_request_transport_cost",
        ),
        CheckConstraint(
            "estimated_carbon_emission >= 0",
            name="ck_exchange_request_carbon_emission",
        ),
        CheckConstraint(
            "estimated_carbon_saving IS NULL OR estimated_carbon_saving >= 0",
            name="ck_exchange_request_carbon_saving",
        ),
    )

    # =====================================================
    # Foreign Keys
    # =====================================================

    supplier_plant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("plants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    buyer_plant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("plants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    waste_listing_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("waste_listings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    requirement_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("requirements.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # =====================================================
    # AI Recommendation Metrics
    # =====================================================

    compatibility_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 2),
        nullable=False,
    )

    ai_confidence_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 2),
        nullable=False,
    )

    recommendation_rank: Mapped[int] = mapped_column(
        nullable=False,
    )

    distance_km: Mapped[Decimal] = mapped_column(
        Numeric(10, 2),
        nullable=False,
    )

    estimated_transport_cost: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
    )

    estimated_carbon_emission: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
    )

    estimated_carbon_saving: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2),
        nullable=True,
    )

    recommendation_reason: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    status: Mapped[ExchangeRequestStatus] = mapped_column(
        Enum(ExchangeRequestStatus),
        default=ExchangeRequestStatus.PENDING,
        nullable=False,
        index=True,
    )

    # =====================================================
    # Relationships
    # =====================================================

    supplier_plant: Mapped["Plant"] = relationship(
        foreign_keys=[supplier_plant_id],
    )

    buyer_plant: Mapped["Plant"] = relationship(
        foreign_keys=[buyer_plant_id],
    )

    waste_listing: Mapped["WasteListing"] = relationship(
        back_populates="exchange_requests",
    )

    requirement: Mapped["Requirement"] = relationship(
        back_populates="exchange_requests",
    )

    exchange: Mapped["Exchange" | None] = relationship(
        back_populates="exchange_request",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<ExchangeRequest("
            f"id={self.id}, "
            f"supplier={self.supplier_plant_id}, "
            f"buyer={self.buyer_plant_id}, "
            f"rank={self.recommendation_rank}, "
            f"status='{self.status.value}'"
            f")>"
        )