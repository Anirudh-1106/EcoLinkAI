from __future__ import annotations

import uuid
from datetime import date
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
from app.enums.exchange import WasteStatus
from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.exchange_request import ExchangeRequest
    from app.models.material import Material
    from app.models.plant import Plant


class WasteListing(BaseModel):
    """
    Represents a waste listing created by a plant.

    Waste listings are the primary supply-side entities in the
    EcoLinkAI network and are matched against material
    requirements by the recommendation engine.
    """

    __tablename__ = "waste_listings"

    __table_args__ = (
        CheckConstraint(
            "quantity > 0",
            name="ck_waste_quantity_positive",
        ),
        CheckConstraint(
            "purity_percentage IS NULL OR (purity_percentage BETWEEN 0 AND 100)",
            name="ck_waste_purity",
        ),
        CheckConstraint(
            "moisture_percentage IS NULL OR (moisture_percentage BETWEEN 0 AND 100)",
            name="ck_waste_moisture",
        ),
        CheckConstraint(
            "price_per_unit IS NULL OR price_per_unit >= 0",
            name="ck_waste_price",
        ),
        CheckConstraint(
            "available_until >= available_from",
            name="ck_waste_availability_dates",
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
    # Waste Information
    # =====================================================

    description: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    quantity: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
    )

    unit: Mapped[QuantityUnit] = mapped_column(
        Enum(QuantityUnit),
        nullable=False,
    )

    purity_percentage: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2),
        nullable=True,
    )

    moisture_percentage: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2),
        nullable=True,
    )

    quality_grade: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    price_per_unit: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2),
        nullable=True,
    )

    available_from: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    available_until: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    status: Mapped[WasteStatus] = mapped_column(
        Enum(WasteStatus),
        default=WasteStatus.AVAILABLE,
        nullable=False,
        index=True,
    )

    # =====================================================
    # Relationships
    # =====================================================

    plant: Mapped["Plant"] = relationship(
        back_populates="waste_listings",
    )

    material: Mapped["Material"] = relationship(
        back_populates="waste_listings",
    )

    exchange_requests: Mapped[list["ExchangeRequest"]] = relationship(
        back_populates="waste_listing",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<WasteListing("
            f"id={self.id}, "
            f"plant_id={self.plant_id}, "
            f"quantity={self.quantity} {self.unit.value}, "
            f"status='{self.status.value}'"
            f")>"
        )