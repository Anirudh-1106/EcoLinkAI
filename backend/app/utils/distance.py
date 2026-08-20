"""
Distance and transport utilities.
"""

from __future__ import annotations

import math
from decimal import Decimal

# Earth radius in km
EARTH_RADIUS_KM = 6371.0

# Transport cost per km per ton (INR) — default estimates
TRANSPORT_RATES = {
    "road": {"cost_per_km_per_ton": 8.0, "carbon_per_km_per_ton": 0.062},
    "rail": {"cost_per_km_per_ton": 4.0, "carbon_per_km_per_ton": 0.022},
    "sea": {"cost_per_km_per_ton": 2.5, "carbon_per_km_per_ton": 0.016},
}


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Calculate the great-circle distance between two points
    on Earth using the Haversine formula.

    Args:
        lat1, lon1: Latitude and longitude of point 1 (degrees).
        lat2, lon2: Latitude and longitude of point 2 (degrees).

    Returns:
        Distance in kilometers.
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_KM * c


def estimate_transport_cost(
    distance_km: float,
    quantity_tons: float,
    mode: str = "road",
) -> float:
    """Estimate transport cost in INR."""
    rates = TRANSPORT_RATES.get(mode.lower(), TRANSPORT_RATES["road"])
    return distance_km * quantity_tons * rates["cost_per_km_per_ton"]


def estimate_transport_emission(
    distance_km: float,
    quantity_tons: float,
    mode: str = "road",
) -> float:
    """Estimate CO2 emission in kg for transport."""
    rates = TRANSPORT_RATES.get(mode.lower(), TRANSPORT_RATES["road"])
    return distance_km * quantity_tons * rates["carbon_per_km_per_ton"]


def estimate_carbon_saving(
    quantity_kg: float,
    carbon_factor: float | None = None,
) -> float:
    """
    Estimate carbon saving from reusing material instead of
    virgin production. Uses material-specific carbon factor
    if available, otherwise a default estimate.
    """
    factor = carbon_factor if carbon_factor else 1.2  # kg CO2e per kg material
    return quantity_kg * factor
