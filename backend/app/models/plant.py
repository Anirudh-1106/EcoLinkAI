from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.analytics import Analytics
    from app.models.company import Company
    from app.models.requirement import Requirement
    from app.models.waste_listing import WasteListing


class Plant(BaseModel):
    """
    Represents an industrial plant belonging to a company.

    Each company can own multiple plants. Plants generate waste,
    request materials, and are represented as nodes in the MC-GNN
    recommendation graph.
    """

    __tablename__ = "plants"

    __table_args__ = (
        CheckConstraint(
            "latitude BETWEEN -90 AND 90",
            name="ck_plant_latitude",
        ),
        CheckConstraint(
            "longitude BETWEEN -180 AND 180",
            name="ck_plant_longitude",
        ),
        UniqueConstraint(
            "company_id",
            "plant_name",
            name="uq_company_plant_name",
        ),
    )

    # =====================================================
    # Foreign Key
    # =====================================================

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # =====================================================
    # Plant Information
    # =====================================================

    plant_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    plant_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    address: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    district: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    state: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    country: Mapped[str] = mapped_column(
        String(100),
        default="India",
        nullable=False,
    )

    latitude: Mapped[Decimal] = mapped_column(
        Numeric(10, 7),
        nullable=False,
    )

    longitude: Mapped[Decimal] = mapped_column(
        Numeric(10, 7),
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    # =====================================================
    # Relationships
    # =====================================================

    company: Mapped["Company"] = relationship(
        back_populates="plants",
    )

    waste_listings: Mapped[list["WasteListing"]] = relationship(
        back_populates="plant",
        cascade="all, delete-orphan",
    )

    requirements: Mapped[list["Requirement"]] = relationship(
        back_populates="plant",
        cascade="all, delete-orphan",
    )

    analytics: Mapped["Analytics"] = relationship(
        back_populates="plant",
        cascade="all, delete-orphan",
        uselist=False,
    )

    def __repr__(self) -> str:
        return (
            f"<Plant("
            f"id={self.id}, "
            f"name='{self.plant_name}', "
            f"district='{self.district}'"
            f")>"
        )