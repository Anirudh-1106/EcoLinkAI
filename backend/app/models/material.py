from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Enum,
    Numeric,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.enums.material import HazardClass, MaterialCategory
from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.requirement import Requirement
    from app.models.waste_listing import WasteListing


class Material(BaseModel):
    """
    Master table containing all supported materials in the
    EcoLinkAI network.

    Materials are referenced by both waste listings and
    material requirements to ensure consistent classification
    and matching.
    """

    __tablename__ = "materials"

    __table_args__ = (
        CheckConstraint(
            "default_density >= 0",
            name="ck_material_density",
        ),
        CheckConstraint(
            "carbon_factor >= 0",
            name="ck_material_carbon_factor",
        ),
    )

    # =====================================================
    # Material Information
    # =====================================================

    material_name: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    material_category: Mapped[MaterialCategory] = mapped_column(
        Enum(MaterialCategory),
        nullable=False,
    )

    hazard_class: Mapped[HazardClass] = mapped_column(
        Enum(HazardClass),
        default=HazardClass.NON_HAZARDOUS,
        nullable=False,
    )

    default_density: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 3),
        nullable=True,
    )

    carbon_factor: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )

    description: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    # =====================================================
    # Relationships
    # =====================================================

    waste_listings: Mapped[list["WasteListing"]] = relationship(
        back_populates="material",
    )

    requirements: Mapped[list["Requirement"]] = relationship(
        back_populates="material",
    )

    def __repr__(self) -> str:
        return (
            f"<Material("
            f"id={self.id}, "
            f"name='{self.material_name}', "
            f"category='{self.material_category.value}', "
            f"hazard='{self.hazard_class.value}'"
            f")>"
        )