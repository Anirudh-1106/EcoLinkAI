from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Date, Enum, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.enums.company import DocumentType, VerificationStatus
from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.company import Company


class Verification(BaseModel):
    """
    Stores verification records for company documents.

    A company may have multiple verification documents,
    each with its own verification status.
    """

    __tablename__ = "verifications"

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
    # Document Information
    # =====================================================

    document_type: Mapped[DocumentType] = mapped_column(
        Enum(DocumentType),
        nullable=False,
    )

    document_number: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    document_url: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    issued_date: Mapped[Date] = mapped_column(
        nullable=True,
    )

    expiry_date: Mapped[Date] = mapped_column(
        nullable=True,
    )

    verification_status: Mapped[VerificationStatus] = mapped_column(
        Enum(VerificationStatus),
        default=VerificationStatus.PENDING,
        nullable=False,
        index=True,
    )

    verified_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    remarks: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    # =====================================================
    # Relationships
    # =====================================================

    company: Mapped["Company"] = relationship(
        back_populates="verifications",
    )

    def __repr__(self) -> str:
        return (
            f"<Verification("
            f"id={self.id}, "
            f"company_id={self.company_id}, "
            f"document='{self.document_type.value}', "
            f"status='{self.verification_status.value}'"
            f")>"
        )