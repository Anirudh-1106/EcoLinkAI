from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.exchange import Exchange


class Review(BaseModel):
    """
    Stores mutual feedback for a completed exchange.

    Each exchange has exactly one review record containing
    ratings and feedback from both the supplier and buyer.
    """

    __tablename__ = "reviews"

    __table_args__ = (
        CheckConstraint(
            "supplier_rating BETWEEN 1 AND 5",
            name="ck_review_supplier_rating",
        ),
        CheckConstraint(
            "buyer_rating BETWEEN 1 AND 5",
            name="ck_review_buyer_rating",
        ),
    )

    # =====================================================
    # Foreign Key
    # =====================================================

    exchange_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("exchanges.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # =====================================================
    # Review Details
    # =====================================================

    supplier_rating: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    buyer_rating: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    supplier_feedback: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    buyer_feedback: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    # =====================================================
    # Relationships
    # =====================================================

    exchange: Mapped["Exchange"] = relationship(
        back_populates="review",
    )

    def __repr__(self) -> str:
        return (
            f"<Review("
            f"id={self.id}, "
            f"exchange_id={self.exchange_id}, "
            f"supplier_rating={self.supplier_rating}, "
            f"buyer_rating={self.buyer_rating}"
            f")>"
        )