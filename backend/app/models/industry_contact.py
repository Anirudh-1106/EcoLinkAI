from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.company import Company


class IndustryContact(BaseModel):
    """
    Represents a contact person for a company.

    A company may have multiple contacts, but only one
    should be marked as the primary contact.
    """

    __tablename__ = "industry_contacts"

    # =====================================================
    # Foreign Keys
    # =====================================================

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # =====================================================
    # Contact Information
    # =====================================================

    full_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    designation: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    department: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )

    phone: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    is_primary: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
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
        back_populates="contacts",
    )

    def __repr__(self) -> str:
        return (
            f"<IndustryContact("
            f"id={self.id}, "
            f"name='{self.full_name}', "
            f"email='{self.email}'"
            f")>"
        )