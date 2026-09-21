"""
Quantity unit conversion.

A listing and the requirement it's matched against each carry their own unit
(kg, ton, liter, cubic_meter, piece, meter) with no guarantee they're the
same one. Comparing their raw numbers directly -- "2458.16 ton" vs "723.67
kg" as if both were just "2458.16" and "723.67" -- silently produces a
compatibility score that's off by orders of magnitude.

kg and ton are the same physical quantity (mass) at different scales, so
converting between them is always safe and exact. liter, cubic_meter, piece
and meter measure something else entirely -- volume, count, length -- and
cannot be converted to a mass without the material's density, which this
system does not track anywhere. Every function here treats that case as
"cannot be compared" rather than guessing.
"""

from __future__ import annotations

from app.enums.common import QuantityUnit
from app.utils.distance import estimate_carbon_saving

# kg per unit, for the units that are genuinely mass measurements.
_KG_PER_UNIT: dict[QuantityUnit, float] = {
    QuantityUnit.KG: 1.0,
    QuantityUnit.TON: 1000.0,
}


def to_kg(quantity: float, unit: QuantityUnit) -> float | None:
    """
    Convert a quantity to kilograms.

    Returns None when the unit isn't a convertible mass unit (liter,
    cubic_meter, piece, meter) -- callers must treat that as "unknown", never
    substitute the raw number as if it were already in kilograms.
    """
    factor = _KG_PER_UNIT.get(unit)
    if factor is None:
        return None
    return quantity * factor


def quantity_fit_pct(
    quantity_a: float,
    unit_a: QuantityUnit,
    quantity_b: float,
    unit_b: QuantityUnit,
    *,
    default: float = 50.0,
) -> float:
    """
    How closely two quantities match, as a 0-100 score.

    100 when they're equal after converting both to kilograms; falls off
    symmetrically the further apart they are. Returns `default` -- the same
    "unknown" fallback already used elsewhere for missing data -- when either
    quantity is non-positive or the two units can't both be expressed as a
    mass, instead of comparing numbers that don't mean the same thing.
    """
    if quantity_a <= 0 or quantity_b <= 0:
        return default

    kg_a = to_kg(quantity_a, unit_a)
    kg_b = to_kg(quantity_b, unit_b)
    if kg_a is None or kg_b is None:
        return default

    ratio = min(kg_a / kg_b, kg_b / kg_a)
    return ratio * 100.0


def price_per_kg(price_per_unit: float, unit: QuantityUnit) -> float | None:
    """
    Normalise a "price per {unit}" figure to price-per-kilogram.

    A price quoted per ton is a thousandth of the same price quoted per kg;
    comparing the two raw numbers directly makes a per-ton price look 1000x
    cheaper than it is. Returns None when the unit isn't a mass unit, same
    convention as to_kg().
    """
    kg = to_kg(1.0, unit)
    if kg is None or kg <= 0:
        return None
    return price_per_unit / kg


def transport_tons(quantity: float, unit: QuantityUnit, *, floor_tons: float = 0.1) -> float:
    """
    Weight in tons for the transport-cost / transport-emission formulas.

    Falls back to `floor_tons` (matching the floor these formulas already
    apply to avoid a zero-cost estimate) when the unit can't be expressed as
    a mass -- a conservative placeholder rather than treating a piece-count
    or a volume as if it were already in tons.
    """
    kg = to_kg(quantity, unit)
    if kg is None:
        return floor_tons
    return max(kg / 1000.0, floor_tons)


def carbon_saving_for_quantity(
    quantity: float, unit: QuantityUnit, carbon_factor: float | None
) -> float:
    """
    Estimated CO2e avoided by reusing this quantity instead of virgin production.

    Returns 0.0 when the unit can't be expressed as a mass -- an honest
    "unknown benefit" rather than crediting a piece-count or volume with a
    carbon saving computed as though it were a kilogram figure.
    """
    kg = to_kg(quantity, unit)
    if kg is None:
        return 0.0
    return estimate_carbon_saving(kg, carbon_factor)
