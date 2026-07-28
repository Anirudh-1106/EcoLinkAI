from __future__ import annotations

from sqlalchemy import Boolean, Enum, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel
from app.enums.transport import VehicleType

# ==========================================
# ENUMS
# ==========================================

from enum import Enum as PyEnum


class VehicleType(str, PyEnum):
    MINI_TRUCK = "Mini Truck"
    LIGHT_TRUCK = "Light Truck"
    MEDIUM_TRUCK = "Medium Truck"
    HEAVY_TRUCK = "Heavy Truck"
    CONTAINER = "Container"


# ==========================================
# MODEL
# ==========================================


class TransportRate(BaseModel):
    __tablename__ = "transport_rates"

    vehicle_type: Mapped[VehicleType] = mapped_column(
        Enum(VehicleType),
        nullable=False,
        unique=True,
    )

    max_capacity_tons: Mapped[float] = mapped_column(
        Numeric(10, 2),
        nullable=False,
    )

    cost_per_km: Mapped[float] = mapped_column(
        Numeric(10, 2),
        nullable=False,
    )

    carbon_emission_per_km: Mapped[float] = mapped_column(
        Numeric(10, 4),
        nullable=False,
    )

    average_speed_kmph: Mapped[float] = mapped_column(
        Numeric(6, 2),
        nullable=True,
    )

    fuel_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<TransportRate("
            f"vehicle='{self.vehicle_type.value}', "
            f"cost_per_km={self.cost_per_km}"
            f")>"
        )