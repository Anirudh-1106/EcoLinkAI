from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Enum,
    Float,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.enums.company import VerificationStatus
from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.industry_contact import IndustryContact
    from app.models.plant import Plant
    from app.models.verification import Verification


class Company(BaseModel):
    """
    Represents a registered company participating in the
    EcoLinkAI circular economy network.

    A company can own multiple manufacturing plants, have
    multiple contacts, and maintain verification records.
    """

    __tablename__ = "companies"

    __table_args__ = (
        CheckConstraint(
            "trust_score >= 0 AND trust_score <= 100",
            name="ck_company_trust_score",
        ),
    )

    # =====================================================
    # Company Information
    # =====================================================

    company_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    industry_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    registration_number: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    gst_number: Mapped[str] = mapped_column(
        String(15),
        unique=True,
        nullable=False,
    )

    license_number: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    verification_status: Mapped[VerificationStatus] = mapped_column(
        Enum(VerificationStatus),
        default=VerificationStatus.PENDING,
        nullable=False,
    )

    trust_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
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

    plants: Mapped[list["Plant"]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan",
    )

    contacts: Mapped[list["IndustryContact"]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan",
    )

    verifications: Mapped[list["Verification"]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<Company("
            f"id={self.id}, "
            f"name='{self.company_name}', "
            f"industry='{self.industry_type}', "
            f"status='{self.verification_status.value}'"
            f")>"
        )