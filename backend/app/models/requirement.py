from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    Date,
    Enum,
    ForeignKey,
    Numeric,
    String,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.enums.common import QuantityUnit
from app.enums.exchange import RequirementStatus
from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.exchange_request import ExchangeRequest
    from app.models.material import Material
    from app.models.plant import Plant


class Requirement(BaseModel):
    """
    Represents a material requirement posted by a plant.

    Requirements are the demand-side entities in the EcoLinkAI
    network and are matched against available waste listings.
    """

    __tablename__ = "requirements"

    __table_args__ = (
        CheckConstraint(
            "quantity > 0",
            name="ck_requirement_quantity_positive",
        ),
        CheckConstraint(
            "minimum_purity IS NULL OR (minimum_purity BETWEEN 0 AND 100)",
            name="ck_requirement_minimum_purity",
        ),
        CheckConstraint(
            "maximum_budget_per_unit IS NULL OR maximum_budget_per_unit >= 0",
            name="ck_requirement_budget",
        ),
        CheckConstraint(
            "preferred_max_distance_km IS NULL OR preferred_max_distance_km >= 0",
            name="ck_requirement_distance",
        ),
    )

    # =====================================================
    # Foreign Keys
    # =====================================================

    plant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("plants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    material_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("materials.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    # =====================================================
    # Requirement Details
    # =====================================================

    quantity: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
    )

    unit: Mapped[QuantityUnit] = mapped_column(
        Enum(QuantityUnit),
        nullable=False,
    )

    minimum_purity: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2),
        nullable=True,
    )

    maximum_budget_per_unit: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2),
        nullable=True,
    )

    required_before: Mapped[Date] = mapped_column(
        Date,
        nullable=False,
    )

    preferred_max_distance_km: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 2),
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    status: Mapped[RequirementStatus] = mapped_column(
        Enum(RequirementStatus),
        default=RequirementStatus.OPEN,
        nullable=False,
        index=True,
    )

    # =====================================================
    # Relationships
    # =====================================================

    plant: Mapped["Plant"] = relationship(
        back_populates="requirements",
    )

    material: Mapped["Material"] = relationship(
        back_populates="requirements",
    )

    exchange_requests: Mapped[list["ExchangeRequest"]] = relationship(
        back_populates="requirement",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<Requirement("
            f"id={self.id}, "
            f"plant_id={self.plant_id}, "
            f"quantity={self.quantity} {self.unit.value}, "
            f"status='{self.status.value}'"
            f")>"
        )