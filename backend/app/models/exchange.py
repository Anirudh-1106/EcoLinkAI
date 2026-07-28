from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Numeric,
    String,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.enums.exchange import ExchangeStatus, ShipmentStatus
from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.exchange_request import ExchangeRequest
    from app.models.review import Review


class Exchange(BaseModel):
    """
    Represents a finalized exchange between two plants.

    An Exchange is created only after an ExchangeRequest
    has been accepted and stores the actual transaction,
    shipment, and completion details.
    """

    __tablename__ = "exchanges"

    __table_args__ = (
        CheckConstraint(
            "agreed_price >= 0",
            name="ck_exchange_price",
        ),
        CheckConstraint(
            "transport_cost >= 0",
            name="ck_exchange_transport_cost",
        ),
        CheckConstraint(
            "actual_quantity IS NULL OR actual_quantity >= 0",
            name="ck_exchange_actual_quantity",
        ),
        CheckConstraint(
            "actual_carbon_emission IS NULL OR actual_carbon_emission >= 0",
            name="ck_exchange_carbon_emission",
        ),
        CheckConstraint(
            "actual_carbon_saving IS NULL OR actual_carbon_saving >= 0",
            name="ck_exchange_carbon_saving",
        ),
    )

    # =====================================================
    # Foreign Key
    # =====================================================

    exchange_request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("exchange_requests.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # =====================================================
    # Exchange Details
    # =====================================================

    exchange_status: Mapped[ExchangeStatus] = mapped_column(
        Enum(ExchangeStatus),
        default=ExchangeStatus.INITIATED,
        nullable=False,
        index=True,
    )

    shipment_status: Mapped[ShipmentStatus] = mapped_column(
        Enum(ShipmentStatus),
        default=ShipmentStatus.PENDING,
        nullable=False,
        index=True,
    )

    agreed_price: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
    )

    transport_cost: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
    )

    actual_quantity: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2),
        nullable=True,
    )

    actual_carbon_emission: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2),
        nullable=True,
    )

    actual_carbon_saving: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2),
        nullable=True,
    )

    expected_delivery_date: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    delivered_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    completion_notes: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    # =====================================================
    # Relationships
    # =====================================================

    exchange_request: Mapped["ExchangeRequest"] = relationship(
        back_populates="exchange",
    )

    review: Mapped["Review | None"] = relationship(
        back_populates="exchange",
        cascade="all, delete-orphan",
        uselist=False,
    )

    def __repr__(self) -> str:
        return (
            f"<Exchange("
            f"id={self.id}, "
            f"exchange_request_id={self.exchange_request_id}, "
            f"status='{self.exchange_status.value}'"
            f")>"
        )