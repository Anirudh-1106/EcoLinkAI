"""
Ingest the real KINFRA company dataset.

Unlike the synthetic files under datasets/csv, this workbook holds data
collected from actual companies in the KINFRA industrial park, Trivandrum.
They are sellers: each has a plant, a waste material and a live listing, but
no recorded buyers and no exchange history, because the companies did not
report who they sell to.

Run this after any database reseed. scripts/database/seed.py only knows about
the synthetic CSVs and will drop these companies, so they have to be put back
separately. Re-running is safe: rows are matched on their natural keys and
updated in place rather than duplicated.
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal

import openpyxl

# Add backend to path so we can import models and database
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from app.core.database import SessionLocal
from app.core.security import hash_password
from app.enums.common import QuantityUnit
from app.enums.company import VerificationStatus
from app.enums.exchange import WasteStatus
from app.enums.material import HazardClass, MaterialCategory
from app.enums.exchange import (
    ExchangeRequestStatus,
    ExchangeStatus,
    RequirementStatus,
    ShipmentStatus,
)
from app.models.analytics import Analytics
from app.models.company import Company
from app.models.exchange import Exchange
from app.models.exchange_request import ExchangeRequest
from app.models.material import Material
from app.models.plant import Plant
from app.models.requirement import Requirement
from app.models.review import Review
from app.models.user import User, UserRole
from app.models.waste_listing import WasteListing
from app.utils.distance import (
    estimate_transport_cost,
    estimate_transport_emission,
    haversine_distance,
)
from app.utils.quantity import carbon_saving_for_quantity, transport_tons

WORKBOOK = os.path.join(os.path.dirname(__file__), "datasets", "kinfra_data.xlsx")

# The park's real location; plants are nudged apart slightly so they do not
# stack on the exact same pixel on the map.
KINFRA_LAT = 8.554790
KINFRA_LON = 76.858660

# Matches the value used for seeded materials, so carbon savings estimated for
# KINFRA listings are comparable with the rest of the catalogue.
DEFAULT_CARBON_FACTOR = Decimal("1.85")

DEFAULT_PASSWORD = "password123"

CATEGORY_MAP = {
    "rubber": MaterialCategory.RUBBER,
    "wood": MaterialCategory.WOOD,
    "plastic": MaterialCategory.PLASTIC,
    "metal": MaterialCategory.METAL,
    "aluminium": MaterialCategory.METAL,
    "aluminum": MaterialCategory.METAL,
    "paper": MaterialCategory.PAPER,
    "glass": MaterialCategory.GLASS,
    "textile": MaterialCategory.TEXTILE,
    "organic": MaterialCategory.ORGANIC,
    "chemical": MaterialCategory.CHEMICAL,
    "electronic": MaterialCategory.ELECTRONIC,
    "construction": MaterialCategory.CONSTRUCTION,
}

LISTING_STATUS_MAP = {
    "active": WasteStatus.AVAILABLE,
    "available": WasteStatus.AVAILABLE,
    "reserved": WasteStatus.RESERVED,
    "sold": WasteStatus.EXCHANGED,
    "exchanged": WasteStatus.EXCHANGED,
    "expired": WasteStatus.EXPIRED,
}


def _find_or_create_material(db, *, name: str, category_label: str, created: dict):
    """
    Look a material up by name, creating it if the catalogue lacks it.

    Matching on name rather than the sheet's id deliberately: a requirement
    for "Plywood Waste" has to resolve to the same catalogue row as the
    listing that sells it, or buyer and seller would never be matched.
    """
    material = db.query(Material).filter(Material.material_name == name).first()
    if material is not None:
        return material

    material = Material(
        id=uuid.uuid4(),
        material_name=name,
        material_category=CATEGORY_MAP.get(
            category_label.strip().lower(), MaterialCategory.OTHER
        ),
        hazard_class=HazardClass.NON_HAZARDOUS,
        default_density=Decimal("1.2"),
        carbon_factor=DEFAULT_CARBON_FACTOR,
        description=f"Waste - {name}",
    )
    db.add(material)
    db.flush()
    created["materials"] += 1
    return material


def _plant_distance(db, plant_a: uuid.UUID, plant_b: uuid.UUID) -> float:
    """Kilometres between two plants, 0 when either lacks coordinates."""
    a = db.query(Plant).filter(Plant.id == plant_a).first()
    b = db.query(Plant).filter(Plant.id == plant_b).first()
    if not a or not b or a.latitude is None or b.latitude is None:
        return 0.0
    return haversine_distance(
        float(a.latitude), float(a.longitude), float(b.latitude), float(b.longitude)
    )


def _sheet_rows(workbook, name: str, *, required: bool = True) -> list[dict]:
    """Read a worksheet into a list of header-keyed dicts."""
    if name not in workbook.sheetnames and not required:
        return []
    if name not in workbook.sheetnames:
        raise ValueError(
            f"Sheet {name!r} missing from {WORKBOOK}. "
            f"Found: {workbook.sheetnames}"
        )
    sheet = workbook[name]
    headers = [sheet.cell(row=1, column=c).value for c in range(1, sheet.max_column + 1)]
    rows = []
    for r in range(2, sheet.max_row + 1):
        values = [sheet.cell(row=r, column=c).value for c in range(1, sheet.max_column + 1)]
        if all(v is None for v in values):
            continue
        rows.append(dict(zip(headers, values)))
    return rows


def _as_date(value, fallback: date) -> date:
    """Dates arrive as datetimes or strings depending on how the sheet was saved."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        text = value.strip().split(" ")[0]
        try:
            return date.fromisoformat(text)
        except ValueError:
            return fallback
    return fallback


def _as_decimal(value, fallback: str = "0") -> Decimal:
    try:
        return Decimal(str(value)) if value is not None else Decimal(fallback)
    except Exception:
        return Decimal(fallback)


def ingest_kinfra_data() -> int:
    """Load the workbook into the database. Returns the number of companies handled."""
    workbook = openpyxl.load_workbook(WORKBOOK, data_only=True)

    companies = _sheet_rows(workbook, "companies")
    plants = _sheet_rows(workbook, "plants")
    materials = _sheet_rows(workbook, "materials")
    listings = _sheet_rows(workbook, "waste_listings")
    # Optional: the workbook only gained a buyer side once a company confirmed
    # what it purchases, so older copies will not have these.
    requirements = _sheet_rows(workbook, "requirements", required=False)
    exchanges = _sheet_rows(workbook, "exchanges", required=False)

    if not companies:
        raise ValueError(f"No company rows found in {WORKBOOK}; nothing to ingest.")

    db = SessionLocal()
    created = {
        "companies": 0, "plants": 0, "materials": 0, "listings": 0, "users": 0,
        "requirements": 0, "exchanges": 0,
    }
    updated = {"companies": 0, "plants": 0, "listings": 0, "requirements": 0, "exchanges": 0}

    try:
        # ── Companies ─────────────────────────────────
        company_ids: dict[str, uuid.UUID] = {}
        for idx, row in enumerate(companies):
            sheet_id = str(row["company_id"])
            name = f"KINFRA - {row['company_name']}"
            registration = str(row.get("registration_number") or f"KIN-{sheet_id}")

            company = (
                db.query(Company)
                .filter(Company.registration_number == registration)
                .first()
            )
            verification = (
                VerificationStatus.VERIFIED
                if str(row.get("verification_status", "")).lower() == "verified"
                else VerificationStatus.PENDING
            )

            if company is None:
                company = Company(
                    id=uuid.uuid4(),
                    company_name=name,
                    registration_number=registration,
                    gst_number=f"32{registration[-10:]}1Z5"[:15],
                    license_number=f"LIC-{sheet_id}",
                )
                db.add(company)
                created["companies"] += 1
            else:
                updated["companies"] += 1

            company.company_name = name
            company.industry_type = str(row.get("industry_sector") or "Manufacturing")
            company.verification_status = verification
            company.trust_score = float(row.get("trust_score") or 50.0)
            company.is_active = True

            db.flush()
            company_ids[sheet_id] = company.id

            # A login for each company, otherwise nobody can act on their
            # behalf -- accepting an exchange request requires a user of the
            # selling company.
            email = str(row.get("email") or f"{sheet_id.lower()}@kinfra.in")
            if not db.query(User).filter(User.email == email).first():
                db.add(
                    User(
                        email=email,
                        hashed_password=hash_password(DEFAULT_PASSWORD),
                        full_name=f"{row['company_name']} Manager",
                        role=UserRole.COMPANY,
                        company_id=company.id,
                    )
                )
                created["users"] += 1

        db.flush()

        # ── Plants ────────────────────────────────────
        plant_ids: dict[str, uuid.UUID] = {}
        for idx, row in enumerate(plants):
            sheet_id = str(row["plant_id"])
            company_uuid = company_ids.get(str(row["company_id"]))
            if company_uuid is None:
                continue

            plant_name = str(row["plant_name"])
            plant = (
                db.query(Plant)
                .filter(Plant.company_id == company_uuid, Plant.plant_name == plant_name)
                .first()
            )

            is_new = plant is None
            if plant is None:
                plant = Plant(id=uuid.uuid4(), company_id=company_uuid, plant_name=plant_name)
                created["plants"] += 1
            else:
                updated["plants"] += 1

            # Populate before flushing; several of these columns are NOT NULL.
            plant.plant_type = str(row.get("plant_type") or "Manufacturing Unit")
            plant.address = "KINFRA Industrial Park"
            plant.district = str(row.get("district") or "Trivandrum")
            plant.state = str(row.get("state") or "Kerala")
            plant.country = str(row.get("country") or "India")
            # Spread plants slightly around the park's real coordinates.
            plant.latitude = Decimal(str(round(KINFRA_LAT + (idx % 5) * 0.004, 6)))
            plant.longitude = Decimal(str(round(KINFRA_LON + (idx % 4) * 0.004, 6)))
            plant.is_active = True

            if is_new:
                db.add(plant)
            db.flush()
            if is_new:
                db.add(Analytics(plant_id=plant.id))
            plant_ids[sheet_id] = plant.id

        db.flush()

        # ── Materials ─────────────────────────────────
        # Reused by name where one already exists, so a buyer's requirement for
        # e.g. "Plastic Scrap" matches these sellers rather than sitting beside
        # a duplicate material nobody is linked to.
        material_ids: dict[str, uuid.UUID] = {}
        for row in materials:
            sheet_id = str(row["material_id"])
            name = str(row["material_name"])

            material = db.query(Material).filter(Material.material_name == name).first()
            if material is None:
                category = CATEGORY_MAP.get(
                    str(row.get("material_category") or "").strip().lower(),
                    MaterialCategory.OTHER,
                )
                hazardous = str(row.get("hazardous", "No")).strip().lower() == "yes"
                material = Material(
                    id=uuid.uuid4(),
                    material_name=name,
                    material_category=category,
                    hazard_class=(
                        HazardClass.HAZARDOUS if hazardous else HazardClass.NON_HAZARDOUS
                    ),
                    default_density=Decimal("1.2"),
                    carbon_factor=DEFAULT_CARBON_FACTOR,
                    description=f"{row.get('material_type', 'Waste')} - {name}",
                )
                db.add(material)
                db.flush()
                created["materials"] += 1

            material_ids[sheet_id] = material.id

        db.flush()

        # ── Waste listings (the products these companies sell) ──
        today = date.today()
        listing_ids: dict[str, uuid.UUID] = {}
        for row in listings:
            plant_uuid = plant_ids.get(str(row["plant_id"]))
            material_uuid = material_ids.get(str(row["material_id"]))
            if plant_uuid is None or material_uuid is None:
                continue

            # Keyed on the availability date as well as the plant and
            # material. A plant's waste is a recurring stream, not a single
            # consignment: RC Interiors generates plywood offcuts every month,
            # and the batch sold to Al-Fas has to stay on record as history
            # while this month's sits on the market. Matching on plant and
            # material alone would make the second batch overwrite the first.
            available_from = _as_date(row.get("availability_date"), today)
            listing = (
                db.query(WasteListing)
                .filter(
                    WasteListing.plant_id == plant_uuid,
                    WasteListing.material_id == material_uuid,
                    WasteListing.available_from == available_from,
                )
                .first()
            )
            if listing is None:
                listing = WasteListing(
                    id=uuid.uuid4(), plant_id=plant_uuid, material_id=material_uuid
                )
                db.add(listing)
                created["listings"] += 1
            else:
                updated["listings"] += 1
            listing_ids[str(row["waste_id"])] = listing.id

            listing.quantity = _as_decimal(row.get("quantity_kg"))
            try:
                listing.unit = QuantityUnit(str(row.get("unit") or "kg").strip().lower())
            except ValueError:
                listing.unit = QuantityUnit.KG
            listing.purity_percentage = _as_decimal(row.get("purity_percentage"), "0")
            listing.price_per_unit = _as_decimal(row.get("asking_price_per_kg_inr"))
            listing.available_from = available_from
            listing.available_until = _as_date(
                row.get("expiry_date"), today + timedelta(days=365)
            )
            listing.status = LISTING_STATUS_MAP.get(
                str(row.get("listing_status") or "").strip().lower(),
                WasteStatus.AVAILABLE,
            )
            listing.description = str(
                row.get("notes") or f"Waste generated at KINFRA facility."
            )

        db.flush()

        # ── Requirements (what these companies want to buy) ──
        # The park's companies were originally recorded as sellers only. Where
        # one has since confirmed what it buys, that belongs here: a seller
        # with no stated demand can never be matched *to*, only *from*.
        requirement_ids: dict[str, uuid.UUID] = {}
        for row in requirements:
            plant_uuid = plant_ids.get(str(row["plant_id"]))
            if plant_uuid is None:
                continue

            material = _find_or_create_material(
                db,
                name=str(row["material_name"]),
                category_label=str(row.get("material_category") or ""),
                created=created,
            )

            requirement = (
                db.query(Requirement)
                .filter(
                    Requirement.plant_id == plant_uuid,
                    Requirement.material_id == material.id,
                )
                .first()
            )
            if requirement is None:
                requirement = Requirement(
                    id=uuid.uuid4(), plant_id=plant_uuid, material_id=material.id
                )
                db.add(requirement)
                created["requirements"] += 1
            else:
                updated["requirements"] += 1

            requirement.quantity = _as_decimal(row.get("quantity_kg"))
            requirement.unit = QuantityUnit.KG
            requirement.minimum_purity = _as_decimal(row.get("minimum_purity"), "0")
            requirement.maximum_budget_per_unit = _as_decimal(
                row.get("maximum_budget_per_kg_inr")
            )
            requirement.required_before = _as_date(
                row.get("required_before"), today + timedelta(days=180)
            )
            requirement.status = (
                RequirementStatus.OPEN
                if str(row.get("status") or "Open").strip().lower() == "open"
                else RequirementStatus.CLOSED
            )
            requirement.notes = str(row.get("notes") or "")
            requirement_ids[str(row["requirement_id"])] = requirement.id

        db.flush()

        # ── Exchanges (deals confirmed to have happened) ──
        # Real trades, unlike the seeded ones. Each becomes an accepted
        # exchange request plus a completed exchange, so it appears in the
        # transaction history and, because the graph propagates over accepted
        # deals, connects these two plants in the network the model reads.
        for row in exchanges:
            supplier_plant = plant_ids.get(str(row["supplier_plant_id"]))
            buyer_plant = plant_ids.get(str(row["buyer_plant_id"]))
            listing_uuid = listing_ids.get(str(row["waste_id"]))
            requirement_uuid = requirement_ids.get(str(row.get("requirement_id") or ""))
            if not (supplier_plant and buyer_plant and listing_uuid):
                continue

            existing = (
                db.query(ExchangeRequest)
                .filter(
                    ExchangeRequest.waste_listing_id == listing_uuid,
                    ExchangeRequest.buyer_plant_id == buyer_plant,
                )
                .first()
            )
            if existing is not None:
                updated["exchanges"] += 1
                continue

            quantity = _as_decimal(row.get("quantity_kg"))
            distance_km = _plant_distance(db, supplier_plant, buyer_plant)
            tons = transport_tons(float(quantity), QuantityUnit.KG)

            request = ExchangeRequest(
                id=uuid.uuid4(),
                supplier_plant_id=supplier_plant,
                buyer_plant_id=buyer_plant,
                waste_listing_id=listing_uuid,
                requirement_id=requirement_uuid,
                requested_quantity=quantity,
                compatibility_score=Decimal("100.00"),
                material_compatibility=Decimal("100.00"),
                quantity_compatibility=Decimal("100.00"),
                quality_compatibility=Decimal("100.00"),
                ai_confidence_score=Decimal("100.00"),
                recommendation_rank=1,
                distance_km=Decimal(str(round(distance_km, 2))),
                estimated_transport_cost=Decimal(
                    str(round(estimate_transport_cost(distance_km, tons), 2))
                ),
                estimated_carbon_emission=Decimal(
                    str(round(estimate_transport_emission(distance_km, tons), 2))
                ),
                estimated_carbon_saving=Decimal(
                    str(round(
                        carbon_saving_for_quantity(
                            float(quantity), QuantityUnit.KG, float(DEFAULT_CARBON_FACTOR)
                        ), 2,
                    ))
                ),
                recommendation_reason=str(row.get("notes") or "Confirmed KINFRA exchange"),
                status=ExchangeRequestStatus.ACCEPTED,
            )
            requested_on = _as_date(row.get("request_date"), today)
            request.created_at = datetime(
                requested_on.year, requested_on.month, requested_on.day
            )
            db.add(request)
            db.flush()

            exchange = Exchange(
                id=uuid.uuid4(),
                exchange_request_id=request.id,
                exchange_status=ExchangeStatus.COMPLETED,
                shipment_status=ShipmentStatus.DELIVERED,
                agreed_price=quantity * _as_decimal(row.get("price_per_kg_inr")),
                transport_cost=request.estimated_transport_cost,
                actual_quantity=quantity,
                actual_carbon_emission=request.estimated_carbon_emission,
                actual_carbon_saving=request.estimated_carbon_saving,
                completion_notes=str(row.get("notes") or ""),
            )
            db.add(exchange)
            db.flush()
            created["exchanges"] += 1

            # Both sides rated each other, which is what a confirmed deal
            # between two willing parties looks like.
            supplier_rating = row.get("supplier_rating")
            buyer_rating = row.get("buyer_rating")
            if supplier_rating or buyer_rating:
                db.add(Review(
                    id=uuid.uuid4(),
                    exchange_id=exchange.id,
                    supplier_rating=int(supplier_rating) if supplier_rating else None,
                    buyer_rating=int(buyer_rating) if buyer_rating else None,
                    supplier_feedback="Confirmed KINFRA park exchange.",
                    buyer_feedback="Confirmed KINFRA park exchange.",
                ))

        db.commit()
    finally:
        db.close()

    print("KINFRA ingest complete.")
    print(
        f"  companies : {created['companies']} created, {updated['companies']} updated"
    )
    print(f"  users     : {created['users']} created")
    print(f"  plants    : {created['plants']} created, {updated['plants']} updated")
    print(f"  materials : {created['materials']} created (existing names reused)")
    print(f"  listings  : {created['listings']} created, {updated['listings']} updated")
    print(f"  requirements: {created['requirements']} created, {updated['requirements']} updated")
    print(f"  exchanges : {created['exchanges']} created, {updated['exchanges']} already present")
    return len(companies)


if __name__ == "__main__":
    handled = ingest_kinfra_data()
    if handled == 0:
        raise SystemExit("No KINFRA companies were ingested - check the workbook.")
