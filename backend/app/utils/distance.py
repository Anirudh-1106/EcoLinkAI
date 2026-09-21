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

# Marginal rate multipliers by consignment size, in the style of tax brackets.
#
# Charging one flat per-ton rate prices a 2,500 t bulk movement as though it
# were 1,400 separate 1.7 t deliveries. Freight is not sold that way: the cost
# of dispatching a vehicle is largely fixed, so once a consignment fills a
# truck and then several, the rate per ton falls steeply.
#
# Each entry is (upper bound in tons, multiplier on the mode's base rate), and
# a multiplier applies only to the tonnage falling inside its own bracket. That
# keeps the total continuous across bracket edges and strictly increasing --
# an extra ton always costs more, just less than the ton before it.
FREIGHT_BRACKETS: list[tuple[float, float]] = [
    (5.0, 1.00),           # part load, shares a vehicle, no consolidation gain
    (25.0, 0.75),          # up to roughly one full truckload
    (100.0, 0.55),         # several trucks, contracted movement
    (float("inf"), 0.38),  # bulk haulage, competitive with rail
]


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


def billable_ton_rate_units(quantity_tons: float) -> float:
    """
    Tonnage repriced through FREIGHT_BRACKETS, in "effective tons".

    Multiply by a mode's base per-ton-km rate and by distance to get a cost.
    For consignments at or below the first bracket this returns the tonnage
    unchanged, so small movements keep costing exactly what they did before.
    """
    if quantity_tons <= 0:
        return 0.0

    units = 0.0
    lower = 0.0
    for upper, multiplier in FREIGHT_BRACKETS:
        if quantity_tons <= lower:
            break
        units += (min(quantity_tons, upper) - lower) * multiplier
        lower = upper

    return units


def estimate_transport_cost(
    distance_km: float,
    quantity_tons: float,
    mode: str = "road",
) -> float:
    """
    Estimate transport cost in INR, with a volume discount on large loads.

    See FREIGHT_BRACKETS: the per-ton rate tapers as the consignment grows,
    rather than a 2,500 t movement costing 1,400x a 1.7 t one over the same
    road.
    """
    rates = TRANSPORT_RATES.get(mode.lower(), TRANSPORT_RATES["road"])
    return distance_km * billable_ton_rate_units(quantity_tons) * rates["cost_per_km_per_ton"]


def estimate_transport_emission(
    distance_km: float,
    quantity_tons: float,
    mode: str = "road",
) -> float:
    """
    Estimate CO2 emission in kg for transport.

    Deliberately linear in tonnage, unlike cost: the freight brackets model
    how haulage is *priced*, and moving twice the mass still burns close to
    twice the fuel.
    """
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
