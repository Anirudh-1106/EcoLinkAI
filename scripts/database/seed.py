"""
Database Seeding Script for EcoLinkAI.

Imports synthetic CSV seed data into PostgreSQL database,
mapping CSV string IDs (C001, P001, M001) to PostgreSQL UUIDs
while preserving all relationships and foreign keys.

Also seeds:
- Default transport rates
- Default admin user (admin@ecolink.ai / admin123)
- Demo company users for easy testing
"""

from __future__ import annotations

import csv
import logging
import math
import random
import sys
import uuid
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path

# Add backend directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.core.database import SessionLocal
from app.core.security import hash_password
from app.enums.common import QuantityUnit
from app.enums.company import VerificationStatus
from app.enums.exchange import (
    ExchangeRequestStatus,
    ExchangeStatus,
    RequirementStatus,
    ShipmentStatus,
    WasteStatus,
)
from app.enums.material import HazardClass, MaterialCategory
from app.enums.transport import VehicleType
from app.models.analytics import Analytics
from app.models.company import Company
from app.models.exchange import Exchange
from app.models.exchange_request import ExchangeRequest
from app.models.material import Material
from app.models.plant import Plant
from app.models.requirement import Requirement
from app.models.review import Review
from app.models.transport_rate import TransportRate
from app.models.user import User, UserRole
from app.models.waste_listing import WasteListing
from app.utils.distance import (
    estimate_transport_cost,
    estimate_transport_emission,
    haversine_distance,
)
from app.utils.quantity import (
    carbon_saving_for_quantity,
    quantity_fit_pct,
    transport_tons,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("seed")

CSV_DIR = PROJECT_ROOT / "datasets" / "csv"

# Fixed seed so re-seeding reproduces an identical dataset.
OUTCOME_SEED = 42

# Tuned so the share of accepted requests lands near 40%, matching the mix the
# platform's analytics were built around.
ACCEPT_INTERCEPT = -5.1


def _quality_fit(listing_purity: float | None, required_purity: float | None) -> float:
    """Whether the listing's purity clears the buyer's minimum, 0-100."""
    if not required_purity or not listing_purity:
        return 100.0
    if listing_purity >= required_purity:
        return 100.0
    return (listing_purity / required_purity) * 100.0


def _compatibility_score(quantity_compat: float, quality_compat: float) -> float:
    """
    How well a listing fits the request overall, as a 0-100 score.

    Blends the three things a buyer weighs: the materials matching at all,
    whether the quantities line up, and whether the purity clears their
    minimum. Deliberately the same blend as composite_compatibility() in
    recommendation_service, so the MC-GNN is trained on and served the same
    quantity.
    """
    material_compat = 100.0  # listing and requirement share a material by construction
    score = 0.5 * material_compat + 0.3 * quantity_compat + 0.2 * quality_compat
    return max(0.0, min(score, 100.0))


def _parse_request_date(raw: str | None) -> datetime | None:
    """Parse a CSV request_date (YYYY-MM-DD) into a datetime, or None."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw.strip(), "%Y-%m-%d")
    except ValueError:
        return None


def _simulate_request_outcome(
    rng: random.Random,
    *,
    distance_km: float,
    quantity_compat: float,
    buyer_trust: float,
    price_ratio: float,
    prior_successes: int = 0,
) -> str:
    """
    Decide how a historical exchange request ended.

    The outcome follows from the deal's actual merits -- how close the plants
    are, how well the quantities and purity line up, how trusted the buyer is,
    and how attractive the offered price is -- passed through a logistic curve
    and then *sampled*. Sampling rather than thresholding matters: it leaves
    the label correlated with the features without being a deterministic
    function of them, because real negotiations also turn on things no dataset
    records. A model can therefore learn a genuine signal here, but can never
    reach a perfect score.

    Requests that were never resolved are drawn separately, since "still
    pending" is not a verdict on the match.
    """
    proximity = 1.0 - min(distance_km / 500.0, 1.0)
    # Driven by the quantity fit rather than the blended compatibility score:
    # the blend is half a constant (material match), which squeezes its range
    # too narrow to carry a usable signal. The stored compatibility score is a
    # linear function of this, so the model can still read it.
    quantity_fit = max(0.0, min(quantity_compat / 100.0, 1.0))
    trust = max(0.0, min(buyer_trust / 100.0, 1.0))
    # Offers land between 0.85x and 1.15x the asking price; map that to 0-1.
    price_appeal = max(0.0, min((price_ratio - 0.85) / 0.30, 1.0))
    # Established partners deal with each other more readily; saturates so a
    # long history helps but never guarantees the outcome.
    rapport = min(prior_successes / 3.0, 1.0)

    z = (
        ACCEPT_INTERCEPT
        + 2.60 * proximity
        + 2.40 * quantity_fit
        + 1.20 * trust
        + 1.60 * rapport  # relational effect, only visible via graph structure
        + 0.90 * price_appeal  # not visible to the model: irreducible noise
        + rng.gauss(0.0, 0.35)  # unobserved factors, keeps the label noisy
    )
    p_accept = 1.0 / (1.0 + math.exp(-z))

    if rng.random() < p_accept:
        return "Accepted"

    return rng.choices(
        ["Rejected", "Pending", "Cancelled"], weights=[55, 30, 15], k=1
    )[0]


def seed_database():
    """Main seeding function."""
    from app.core.database import engine
    from app.models import Base
    
    # Ensure all tables are created
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    try:
        # Check if already seeded
        existing_companies = db.query(Company).count()
        if existing_companies > 0:
            logger.info(f"Database already contains {existing_companies} companies. Skipping seed.")
            return

        logger.info("Starting database seed from CSV files...")

        # ID Mappings: CSV string ID -> UUID
        company_map: dict[str, uuid.UUID] = {}
        plant_map: dict[str, uuid.UUID] = {}
        company_first_plant: dict[str, uuid.UUID] = {}
        plant_coords: dict[uuid.UUID, tuple[float, float]] = {}
        company_trust: dict[str, float] = {}
        # csv request id -> (distance km, listing unit, material carbon factor)
        accepted_requests: dict[str, tuple[float, QuantityUnit, float | None]] = {}
        outcome_rng = random.Random(OUTCOME_SEED)
        material_map: dict[str, uuid.UUID] = {}
        waste_map: dict[str, uuid.UUID] = {}
        request_map: dict[str, uuid.UUID] = {}
        transaction_map: dict[str, uuid.UUID] = {}

        # ── 1. Seed Transport Rates ────────────────────
        logger.info("Seeding Transport Rates...")
        default_rates = [
            TransportRate(vehicle_type=VehicleType.MINI_TRUCK, max_capacity_tons=2.5, cost_per_km=15.0, carbon_emission_per_km=0.12, average_speed_kmph=40.0, fuel_type="Diesel"),
            TransportRate(vehicle_type=VehicleType.LIGHT_TRUCK, max_capacity_tons=5.0, cost_per_km=25.0, carbon_emission_per_km=0.22, average_speed_kmph=45.0, fuel_type="Diesel"),
            TransportRate(vehicle_type=VehicleType.MEDIUM_TRUCK, max_capacity_tons=10.0, cost_per_km=40.0, carbon_emission_per_km=0.38, average_speed_kmph=50.0, fuel_type="Diesel"),
            TransportRate(vehicle_type=VehicleType.HEAVY_TRUCK, max_capacity_tons=20.0, cost_per_km=65.0, carbon_emission_per_km=0.65, average_speed_kmph=45.0, fuel_type="Diesel"),
            TransportRate(vehicle_type=VehicleType.CONTAINER_TRUCK, max_capacity_tons=32.0, cost_per_km=90.0, carbon_emission_per_km=0.95, average_speed_kmph=40.0, fuel_type="Diesel"),
        ]
        db.add_all(default_rates)
        db.flush()

        # ── 2. Seed Admin User ─────────────────────────
        logger.info("Seeding Admin User...")
        admin_user = User(
            email="admin@ecolink.ai",
            hashed_password=hash_password("admin123"),
            full_name="EcoLinkAI Platform Admin",
            role=UserRole.ADMIN,
        )
        db.add(admin_user)
        db.flush()

        # ── 3. Seed Companies ──────────────────────────
        companies_file = CSV_DIR / "companies.csv"
        if companies_file.exists():
            logger.info("Seeding Companies...")
            with open(companies_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    c_id = row["company_id"]
                    u_id = uuid.uuid4()
                    company_map[c_id] = u_id

                    status_str = row.get("verification_status", "Pending")
                    status_enum = VerificationStatus.VERIFIED if status_str == "Verified" else VerificationStatus.PENDING

                    company_trust[c_id] = float(row.get("trust_score", 75.0))

                    comp = Company(
                        id=u_id,
                        company_name=row["company_name"],
                        industry_type=row["industry_sector"],
                        registration_number=row["registration_number"],
                        gst_number=f"32{row['registration_number'][-10:]}1Z5"[:15],
                        license_number=f"LIC-{c_id}",
                        verification_status=status_enum,
                        trust_score=float(row.get("trust_score", 75.0)),
                    )
                    db.add(comp)

                    # Create a default user for each seeded company
                    comp_user = User(
                        email=row["email"],
                        hashed_password=hash_password("password123"),
                        full_name=f"{row['company_name']} Manager",
                        role=UserRole.COMPANY,
                        company_id=u_id,
                    )
                    db.add(comp_user)
            db.flush()

        # ── 4. Seed Plants ─────────────────────────────
        plants_file = CSV_DIR / "plants.csv"
        if plants_file.exists():
            logger.info("Seeding Plants...")
            # Kerala district coordinates mapping for realistic plant lat/lon
            # All 14 Kerala districts. Any district missing here silently falls
            # back to a single default point, which stacks unrelated plants on
            # top of each other and corrupts distance -- the strongest feature
            # the recommender has.
            district_coords = {
                "Thiruvananthapuram": (8.5241, 76.9366),
                "Kollam": (8.8932, 76.6141),
                "Pathanamthitta": (9.2648, 76.7870),
                "Alappuzha": (9.4981, 76.3388),
                "Kottayam": (9.5916, 76.5222),
                "Idukki": (9.8497, 76.9681),
                "Ernakulam": (9.9816, 76.2999),
                "Thrissur": (10.5276, 76.2144),
                "Palakkad": (10.7867, 76.6548),
                "Malappuram": (11.0732, 76.0740),
                "Kozhikode": (11.2588, 75.7804),
                "Wayanad": (11.6854, 76.1320),
                "Kannur": (11.8745, 75.3704),
                "Kasaragod": (12.4996, 74.9869),
            }

            with open(plants_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader, 1):
                    p_id = row["plant_id"]
                    c_id = row["company_id"]
                    u_id = uuid.uuid4()
                    plant_map[p_id] = u_id

                    if c_id not in company_first_plant:
                        company_first_plant[c_id] = u_id

                    district = row.get("district", "Ernakulam")
                    base_lat, base_lon = district_coords.get(district, (10.0, 76.5))
                    # Add tiny jitter so plants in same district don't overlap exactly
                    lat = Decimal(str(round(base_lat + (idx % 10) * 0.015, 6)))
                    lon = Decimal(str(round(base_lon + (idx % 8) * 0.015, 6)))
                    plant_coords[u_id] = (float(lat), float(lon))

                    plant = Plant(
                        id=u_id,
                        company_id=company_map[c_id],
                        plant_name=row["plant_name"],
                        plant_type=row.get("plant_type", "Manufacturing Facility"),
                        address=f"Industrial Zone, {district}",
                        district=district,
                        state=row.get("state", "Kerala"),
                        country=row.get("country", "India"),
                        latitude=lat,
                        longitude=lon,
                    )
                    db.add(plant)

                    # Initialize analytics record for each plant
                    analytics = Analytics(plant_id=u_id)
                    db.add(analytics)

            db.flush()

        # ── 5. Seed Materials ──────────────────────────
        materials_file = CSV_DIR / "materials.csv"
        if materials_file.exists():
            logger.info("Seeding Materials...")
            with open(materials_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                seen_names = set()
                for row in reader:
                    m_id = row["material_id"]
                    name = row["material_name"]
                    if name in seen_names:
                        # Append unique suffix if material name repeats in CSV
                        name = f"{name} ({m_id})"
                    seen_names.add(name)

                    u_id = uuid.uuid4()
                    material_map[m_id] = u_id

                    cat_str = row.get("material_category", "Metal")
                    try:
                        cat_enum = MaterialCategory(cat_str)
                    except ValueError:
                        cat_enum = MaterialCategory.OTHER

                    haz_flag = row.get("hazardous", "No") == "Yes"
                    haz_enum = HazardClass.HAZARDOUS if haz_flag else HazardClass.NON_HAZARDOUS

                    mat = Material(
                        id=u_id,
                        material_name=name,
                        material_category=cat_enum,
                        hazard_class=haz_enum,
                        default_density=Decimal("1.2"),
                        carbon_factor=Decimal("1.85"),
                        description=f"{row.get('material_type', 'Raw Waste')} - {name}",
                    )
                    db.add(mat)
            db.flush()

        # ── 6. Seed Waste Listings & Requirements ────
        waste_file = CSV_DIR / "waste_listings.csv"
        if waste_file.exists():
            logger.info("Seeding Waste Listings & Requirements...")
            with open(waste_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    w_id = row["waste_id"]
                    u_id = uuid.uuid4()
                    waste_map[w_id] = u_id

                    p_id = row["plant_id"]
                    m_id = row["material_id"]
                    unit_str = row.get("unit", "kg").lower()
                    unit_enum = QuantityUnit.TON if unit_str in ("ton", "tons") else QuantityUnit.KG

                    status_str = row.get("listing_status", "Available")
                    status_map_dict = {
                        "Available": WasteStatus.AVAILABLE,
                        "Sold": WasteStatus.EXCHANGED,
                        "Exchanged": WasteStatus.EXCHANGED,
                        "Expired": WasteStatus.EXPIRED,
                        "Reserved": WasteStatus.RESERVED,
                    }
                    w_status = status_map_dict.get(status_str, WasteStatus.AVAILABLE)

                    w_list = WasteListing(
                        id=u_id,
                        plant_id=plant_map[p_id],
                        material_id=material_map[m_id],
                        quantity=Decimal(row["quantity"]),
                        unit=unit_enum,
                        purity_percentage=Decimal(row["purity_percentage"]) if row.get("purity_percentage") else Decimal("90.0"),
                        price_per_unit=Decimal(row["asking_price_per_unit"]) if row.get("asking_price_per_unit") else Decimal("100.0"),
                        available_from=datetime.strptime(row["availability_date"], "%Y-%m-%d").date() if row.get("availability_date") else date.today(),
                        available_until=datetime.strptime(row["expiry_date"], "%Y-%m-%d").date() if row.get("expiry_date") else date(2026, 12, 31),
                        quality_grade="Grade A",
                        status=w_status,
                    )
                    db.add(w_list)

                    # Also create corresponding Requirement on another plant every 3 listings to ensure demand
                    if idx % 3 == 0:
                        all_plant_ids = list(plant_map.values())
                        req_plant = [pid for pid in all_plant_ids if pid != plant_map[p_id]]
                        if req_plant:
                            target_plant_id = req_plant[idx % len(req_plant)]
                            req = Requirement(
                                plant_id=target_plant_id,
                                material_id=material_map[m_id],
                                quantity=Decimal(row["quantity"]),
                                unit=unit_enum,
                                minimum_purity=Decimal("80.0"),
                                maximum_budget_per_unit=Decimal(row.get("asking_price_per_unit", "120.0")),
                                required_before=date(2026, 12, 31),
                                preferred_max_distance_km=Decimal("300.0"),
                                status=RequirementStatus.OPEN,
                            )
                            db.add(req)

            db.flush()

        # ── 7. Seed Exchange Requests ──────────────────
        req_file = CSV_DIR / "exchange_requests.csv"
        seen_matches = set()
        if req_file.exists():
            logger.info("Seeding Exchange Requests...")
            # Successful trades between two companies make the next deal
            # between them likelier, so requests are replayed in date order and
            # the partnership history accumulates as we go. That relationship
            # effect lives in the graph's structure rather than in any single
            # row, which is precisely what a graph model can exploit and a
            # per-row formula cannot.
            partnership_history: dict[tuple[str, str], int] = {}
            with open(req_file, mode="r", encoding="utf-8") as f:
                rows = sorted(
                    csv.DictReader(f),
                    key=lambda r: r.get("request_date", ""),
                )
                for row in rows:
                    r_id = row["request_id"]
                    w_id = row["waste_id"]
                    u_id = uuid.uuid4()

                    sup_c_id = row["supplier_company_id"]
                    req_c_id = row["requester_company_id"]

                    sup_p_id = company_first_plant.get(sup_c_id)
                    buyer_p_id = company_first_plant.get(req_c_id)

                    if not sup_p_id or not buyer_p_id or sup_p_id == buyer_p_id or w_id not in waste_map:
                        continue

                    # Pick a requirement for this material if present
                    listing_obj = db.query(WasteListing).filter(WasteListing.id == waste_map[w_id]).first()
                    req_obj = None
                    if listing_obj:
                        req_obj = db.query(Requirement).filter(
                            Requirement.material_id == listing_obj.material_id,
                            Requirement.plant_id == buyer_p_id,
                        ).first()

                    if not req_obj and listing_obj:
                        # Auto-create requirement for buyer plant
                        req_obj = Requirement(
                            plant_id=buyer_p_id,
                            material_id=listing_obj.material_id,
                            quantity=Decimal(row["requested_quantity"]),
                            unit=listing_obj.unit,
                            required_before=date(2026, 12, 31),
                            status=RequirementStatus.OPEN,
                        )
                        db.add(req_obj)
                        db.flush()

                    match_key = (waste_map[w_id], req_obj.id if req_obj else None)
                    if req_obj and match_key in seen_matches:
                        continue
                    if req_obj:
                        seen_matches.add(match_key)

                    request_map[r_id] = u_id

                    # ── Real match features for this specific pairing ──
                    # Previously these were fixed placeholders, identical on
                    # every row, which left the MC-GNN with no variation to
                    # learn from. They are now derived from the actual plants,
                    # listing and requirement involved.
                    sup_lat, sup_lon = plant_coords[sup_p_id]
                    buy_lat, buy_lon = plant_coords[buyer_p_id]
                    distance_km = haversine_distance(sup_lat, sup_lon, buy_lat, buy_lon)

                    requested_qty = float(row["requested_quantity"])
                    listing_qty = float(listing_obj.quantity) if listing_obj else requested_qty
                    # requested_qty is generated (in generate_exchange_requests.py)
                    # as a fraction of this same listing's own quantity, so it is
                    # always already in the listing's unit -- there is no separate
                    # "requirement" unit to reconcile here.
                    listing_unit = listing_obj.unit if listing_obj else QuantityUnit.KG
                    listing_purity = (
                        float(listing_obj.purity_percentage)
                        if listing_obj and listing_obj.purity_percentage is not None
                        else None
                    )
                    required_purity = (
                        float(req_obj.minimum_purity)
                        if req_obj and req_obj.minimum_purity is not None
                        else None
                    )

                    quantity_compat = quantity_fit_pct(
                        listing_qty, listing_unit, requested_qty, listing_unit
                    )
                    quality_compat = _quality_fit(listing_purity, required_purity)
                    compatibility = _compatibility_score(quantity_compat, quality_compat)

                    quantity_tons = transport_tons(requested_qty, listing_unit)
                    transport_cost = estimate_transport_cost(distance_km, quantity_tons)
                    transport_emission = estimate_transport_emission(distance_km, quantity_tons)
                    carbon_factor = (
                        float(listing_obj.material.carbon_factor)
                        if listing_obj and listing_obj.material and listing_obj.material.carbon_factor
                        else None
                    )
                    carbon_saving = carbon_saving_for_quantity(requested_qty, listing_unit, carbon_factor)

                    # ── Outcome follows from those features ───────────
                    # The generated CSV assigns a status by fixed-weight random
                    # draw, with no bearing on the deal itself. Deriving it here
                    # instead is what gives the dataset a signal worth learning.
                    asking_price = (
                        float(listing_obj.price_per_unit)
                        if listing_obj and listing_obj.price_per_unit
                        else 0.0
                    )
                    offered_price = float(row.get("offered_price_per_unit") or 0.0)
                    price_ratio = (offered_price / asking_price) if asking_price > 0 else 1.0

                    pair_key = (sup_c_id, req_c_id)
                    status_str = _simulate_request_outcome(
                        outcome_rng,
                        distance_km=distance_km,
                        quantity_compat=quantity_compat,
                        buyer_trust=company_trust.get(req_c_id, 75.0),
                        price_ratio=price_ratio,
                        prior_successes=partnership_history.get(pair_key, 0),
                    )
                    if status_str == "Accepted":
                        partnership_history[pair_key] = (
                            partnership_history.get(pair_key, 0) + 1
                        )
                        # Only accepted requests can go on to become a real
                        # transaction; kept so the exchange's emissions and
                        # carbon saving can be computed from the actual route
                        # and correctly converted, not assumed to be kg.
                        accepted_requests[r_id] = (distance_km, listing_unit, carbon_factor)
                    status_dict = {
                        "Pending": ExchangeRequestStatus.PENDING,
                        "Accepted": ExchangeRequestStatus.ACCEPTED,
                        "Rejected": ExchangeRequestStatus.REJECTED,
                        "Cancelled": ExchangeRequestStatus.CANCELLED,
                    }
                    r_status = status_dict.get(status_str, ExchangeRequestStatus.PENDING)

                    # Persist the date the request was actually made, rather
                    # than letting created_at default to the moment of the bulk
                    # insert. Without it every row lands on one identical
                    # timestamp, and "which of these two deals came first" --
                    # the ordering this loop itself relies on, and the only
                    # thing that makes a prior-history feature computable
                    # without reading the future -- is lost on the way into the
                    # database.
                    requested_on = _parse_request_date(row.get("request_date"))

                    ex_req = ExchangeRequest(
                        id=u_id,
                        supplier_plant_id=sup_p_id,
                        buyer_plant_id=buyer_p_id,
                        waste_listing_id=waste_map[w_id],
                        requirement_id=req_obj.id if req_obj else None,
                        compatibility_score=Decimal(str(round(compatibility, 2))),
                        ai_confidence_score=Decimal(str(round(compatibility, 2))),
                        recommendation_rank=1,
                        distance_km=Decimal(str(round(distance_km, 2))),
                        estimated_transport_cost=Decimal(str(round(transport_cost, 2))),
                        estimated_carbon_emission=Decimal(str(round(transport_emission, 2))),
                        estimated_carbon_saving=Decimal(str(round(carbon_saving, 2))),
                        recommendation_reason=f"Historical request: {row.get('remarks', 'Direct exchange match')}",
                        status=r_status,
                    )
                    if requested_on is not None:
                        ex_req.created_at = requested_on
                    db.add(ex_req)
            db.flush()

        # ── 8. Seed Transactions (Exchanges) ───────────
        trx_file = CSV_DIR / "transactions.csv"
        if trx_file.exists():
            logger.info("Seeding Transactions (Exchanges)...")
            with open(trx_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t_id = row["transaction_id"]
                    r_id = row["request_id"]
                    u_id = uuid.uuid4()
                    transaction_map[t_id] = u_id

                    if r_id not in request_map:
                        continue

                    # A transaction can only exist where the request was
                    # accepted. The CSV carries its own status, but the accept
                    # decision is made during seeding, so without this check
                    # roughly half the exchanges end up attached to rejected or
                    # still-pending requests -- inflating every "completed
                    # exchange" count the model and the dashboards read.
                    if r_id not in accepted_requests:
                        continue

                    final_quantity = Decimal(row["final_quantity"])
                    # final_quantity is generated (in generate_transactions.py)
                    # as a fraction of the original request's quantity, so it is
                    # in the same listing unit that request was denominated in.
                    accepted_distance_km, accepted_unit, accepted_carbon_factor = accepted_requests[r_id]
                    quantity_tons = transport_tons(float(final_quantity), accepted_unit)
                    emission = estimate_transport_emission(accepted_distance_km, quantity_tons)
                    # Computed here rather than trusting the CSV's carbon_saving_kg
                    # column, which was generated as final_quantity * a random
                    # factor with no unit conversion -- the same bug this whole
                    # fix addresses, just one step further upstream.
                    carbon_saving = carbon_saving_for_quantity(
                        float(final_quantity), accepted_unit, accepted_carbon_factor
                    )

                    ex = Exchange(
                        id=u_id,
                        exchange_request_id=request_map[r_id],
                        exchange_status=ExchangeStatus.COMPLETED,
                        shipment_status=ShipmentStatus.DELIVERED,
                        agreed_price=Decimal(row["final_price_per_unit"]) * final_quantity,
                        transport_cost=Decimal(row["transport_cost"]),
                        actual_quantity=final_quantity,
                        actual_carbon_emission=Decimal(str(round(emission, 2))),
                        actual_carbon_saving=Decimal(str(round(carbon_saving, 2))),
                        delivered_at=datetime.strptime(row["transaction_date"], "%Y-%m-%d") if row.get("transaction_date") else datetime.now(),
                    )
                    db.add(ex)
            db.flush()

        # ── 9. Seed Reviews ─────────────────────────────
        rev_file = CSV_DIR / "reviews.csv"
        if rev_file.exists():
            logger.info("Seeding Reviews...")
            with open(rev_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t_id = row["transaction_id"]
                    if t_id not in transaction_map:
                        continue

                    ex_id = transaction_map[t_id]
                    # Verify exchange actually exists in session
                    ex_exists = db.query(Exchange).filter(Exchange.id == ex_id).first()
                    if not ex_exists:
                        continue

                    rating = int(row.get("rating", 5))

                    rev = Review(
                        exchange_id=ex_id,
                        supplier_rating=rating,
                        buyer_rating=rating,
                        supplier_feedback=row.get("review_comment", "Great partner!"),
                        buyer_feedback="Prompt delivery and good quality material.",
                    )
                    db.add(rev)

        db.commit()
        logger.info("Successfully seeded database with all datasets, admin user, and transport rates!")

    except Exception as e:
        db.rollback()
        logger.error(f"Error seeding database: {e}", exc_info=True)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    # This loads the synthetic CSVs only. The real KINFRA companies live in
    # datasets/kinfra_data.xlsx and are loaded separately, so a database built
    # from here alone is missing them.
    logger.warning(
        "Loading synthetic CSV data only. "
        "Use 'python -m scripts.database.rebuild' to load every dataset."
    )
    seed_database()
