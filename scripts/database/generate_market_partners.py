"""
Give the KINFRA companies a market to trade in.

The synthetic dataset and the real KINFRA companies were generated
independently, so their material catalogues barely overlap. Measured before
this existed: of the twelve materials KINFRA companies sell, exactly one had
a buyer asking for it by name, and "Mixed Recyclable Waste" had no buyer at
all. Every other KINFRA listing could only ever surface as a same-category
match, which carries an explicit compatibility penalty -- so the real
companies were structurally capped below the synthetic ones for a reason
having nothing to do with their merits.

This creates counterparties that deal in exactly those materials:

  buyers      want a named KINFRA material, so the real listings can be
              matched exactly rather than by category
  competitors sell the same materials, so a buyer has genuine alternatives
              and the ranking has something to choose between

Trading then proceeds in date order, and acceptance carries a referral term:
a buyer is likelier to accept a supplier its existing partners already deal
with. That is a documented feature of industrial networks and the one signal
a per-row model cannot read, since it lives in the neighbourhood rather than
the row. Without it the graph carries nothing a plain formula does not
already have.

Runs after the KINFRA ingest, so the materials it matches against are real.
"""

from __future__ import annotations

import csv
import logging
import math
import random
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

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
from app.utils.quantity import (
    carbon_saving_for_quantity,
    quantity_fit_pct,
    transport_tons,
)

logger = logging.getLogger(__name__)

SEED = 2026

# Sized to give message passing room to work. At 124 plants the accepted-deal
# graph averaged 46 neighbours per node -- three quarters of everyone else --
# so every neighbourhood looked alike and the embeddings collapsed toward a
# common value. Spreading the same kind of trading over several hundred plants
# keeps the average degree in the low teens, where a neighbourhood actually
# describes the plant it belongs to.
BUYERS_PER_MATERIAL = 30
COMPETITORS_PER_MATERIAL = 18

# Not every buyer approaches every seller: at this size that would be tens of
# thousands of enquiries and a graph as dense as the one being fixed. Each
# buyer contacts a random subset, which also leaves room for referrals to
# matter -- a buyer who has not met a seller can still hear about them.
SELLERS_CONTACTED_PER_BUYER = 8

# Kerala's industrial districts, so partners are spread across the state
# rather than stacked on the park.
DISTRICTS = [
    ("Thiruvananthapuram", 8.5241, 76.9366),
    ("Kollam", 8.8932, 76.6141),
    ("Ernakulam", 9.9816, 76.2999),
    ("Thrissur", 10.5276, 76.2144),
    ("Kozhikode", 11.2588, 75.7804),
    ("Kannur", 11.8745, 75.3704),
    ("Palakkad", 10.7867, 76.6548),
    ("Alappuzha", 9.4981, 76.3388),
]

# Three parts rather than two. Ten prefixes and six suffixes give sixty
# names, and this creates several hundred companies -- the rest fell through
# to "Kerala Materials 412", which is no use to anyone reading the data.
BUYER_PREFIXES = [
    "Southern", "Malabar", "Coastal", "Western Ghats", "Backwater",
    "Periyar", "Anamudi", "Vembanad", "Nilgiri", "Arabian",
    "Kuttanad", "Wayanad", "Bharata", "Pamba", "Chalakudy",
]
BUYER_MIDDLES = [
    "Polymer", "Metals", "Timber", "Composite", "Industrial",
    "Green", "Salvage", "Foundry", "Fibre", "Alloy", "Board", "Element",
]
BUYER_SUFFIXES = [
    "Recycling Works", "Reprocessing Ltd", "Materials Pvt Ltd",
    "Circular Industries", "Resource Recovery", "Reclamation Co",
    "Recovery Systems", "Processing LLP",
]

# How strongly a partner's endorsement counts toward a deal closing. Matched
# to the weight of a pair's own trading history: in this industry a bad
# consignment is expensive, so who vouches for a supplier matters about as
# much as having dealt with them before.
W_REFERRAL = 1.60
W_PROXIMITY = 2.60
W_QUANTITY = 2.40
W_TRUST = 1.20
W_DYADIC = 1.60
INTERCEPT = -4.10
NOISE_SD = 0.35


# Written out purely so the generated market can be read, diffed and quoted.
# Kept apart from datasets/csv, which holds authored source data: these are an
# output of each rebuild, not an input to it, and editing them changes nothing.
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "datasets" / "generated"


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _write_csv(filename: str, header: list[str], rows: list[list]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / filename, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _company_name(rng: random.Random, used: set[str]) -> str:
    """A distinct, plausible company name. 1,440 combinations before repeats."""
    for _ in range(500):
        name = (
            f"{rng.choice(BUYER_PREFIXES)} {rng.choice(BUYER_MIDDLES)} "
            f"{rng.choice(BUYER_SUFFIXES)}"
        )
        if name not in used:
            used.add(name)
            return name
    name = f"Kerala Materials {len(used)}"
    used.add(name)
    return name


def _make_company(db, rng, used_names, *, trust: float, serial: int) -> tuple[Company, Plant]:
    district, lat, lon = rng.choice(DISTRICTS)
    name = _company_name(rng, used_names)
    # Serial rather than random: registration, GST and licence numbers are all
    # unique columns, and random draws from a small range collide well before
    # the hundredth company.
    company = Company(
        id=uuid.uuid4(),
        company_name=name,
        industry_type="Recycling & Resource Recovery",
        registration_number=f"MKT{serial:06d}",
        gst_number=f"32MKT{serial:07d}Z",
        license_number=f"LIC-MKT-{serial:05d}",
        verification_status=VerificationStatus.VERIFIED,
        trust_score=Decimal(str(round(trust, 2))),
        is_active=True,
    )
    db.add(company)
    db.flush()

    db.add(User(
        id=uuid.uuid4(),
        email=f"market{serial:04d}@partners.in",
        hashed_password=hash_password("password123"),
        full_name=f"{name} Manager",
        role=UserRole.COMPANY,
        company_id=company.id,
        is_active=True,
    ))

    plant = Plant(
        id=uuid.uuid4(),
        company_id=company.id,
        plant_name=f"{name} Facility",
        plant_type="Processing Plant",
        address=f"Industrial Estate, {district}",
        district=district,
        state="Kerala",
        country="India",
        # Nudged so partners in one district do not sit on the same pixel.
        latitude=Decimal(str(round(lat + rng.uniform(-0.12, 0.12), 6))),
        longitude=Decimal(str(round(lon + rng.uniform(-0.12, 0.12), 6))),
        is_active=True,
    )
    db.add(plant)
    db.flush()
    return company, plant


def generate_market_partners() -> dict[str, int]:
    """Create counterparties for the KINFRA materials and let them trade."""
    rng = random.Random(SEED)
    db = SessionLocal()
    counts = defaultdict(int)

    try:
        kinfra = db.query(Company).filter(Company.company_name.like("KINFRA%")).all()
        if not kinfra:
            logger.warning("No KINFRA companies found; skipping market partners.")
            return dict(counts)

        kinfra_plants = [p for c in kinfra for p in c.plants]
        kinfra_plant_ids = {p.id for p in kinfra_plants}

        # The materials the real companies actually sell.
        materials: dict[uuid.UUID, Material] = {}
        for listing in db.query(WasteListing).filter(
            WasteListing.plant_id.in_(kinfra_plant_ids)
        ).all():
            materials[listing.material_id] = listing.material

        used_names: set[str] = {c.company_name for c in db.query(Company).all()}
        serial = 0
        # Collected as we go, then written to datasets/generated so the market
        # is inspectable rather than existing only inside the database.
        out_companies: list[list] = []
        out_requirements: list[list] = []
        out_listings: list[list] = []
        out_exchanges: list[list] = []
        today = date.today()
        buyer_plants: list[tuple[Plant, Material, Requirement]] = []

        for material in materials.values():
            # Buyers: they want this material by name, which is what lets a
            # KINFRA listing match exactly instead of by category.
            for _ in range(BUYERS_PER_MATERIAL):
                serial += 1
                _, plant = _make_company(
                    db, rng, used_names, trust=rng.uniform(55, 95), serial=serial
                )
                requirement = Requirement(
                    id=uuid.uuid4(),
                    plant_id=plant.id,
                    material_id=material.id,
                    quantity=Decimal(str(round(rng.uniform(800, 9000), 2))),
                    unit=QuantityUnit.KG,
                    minimum_purity=Decimal(str(round(rng.uniform(55, 80), 2))),
                    maximum_budget_per_unit=Decimal(str(round(rng.uniform(15, 140), 2))),
                    required_before=today + timedelta(days=rng.randint(60, 300)),
                    status=RequirementStatus.OPEN,
                    notes=f"Sources {material.material_name} for reprocessing.",
                )
                db.add(requirement)
                db.flush()
                buyer_plants.append((plant, material, requirement))
                counts["buyers"] += 1
                out_companies.append([
                    plant.company.company_name, "buyer", material.material_name,
                    plant.plant_name, plant.district,
                    float(plant.latitude), float(plant.longitude),
                    float(plant.company.trust_score),
                ])
                out_requirements.append([
                    plant.company.company_name, material.material_name,
                    float(requirement.quantity), requirement.unit.value,
                    float(requirement.minimum_purity),
                    float(requirement.maximum_budget_per_unit),
                    requirement.required_before.isoformat(),
                ])

            # Competitors: they sell the same material, so a buyer has real
            # alternatives and the ranking has to discriminate.
            for _ in range(COMPETITORS_PER_MATERIAL):
                serial += 1
                _, plant = _make_company(
                    db, rng, used_names, trust=rng.uniform(50, 90), serial=serial
                )
                db.add(WasteListing(
                    id=uuid.uuid4(),
                    plant_id=plant.id,
                    material_id=material.id,
                    description=f"{material.material_name} from processing operations.",
                    quantity=Decimal(str(round(rng.uniform(600, 8000), 2))),
                    unit=QuantityUnit.KG,
                    purity_percentage=Decimal(str(round(rng.uniform(60, 96), 2))),
                    price_per_unit=Decimal(str(round(rng.uniform(10, 130), 2))),
                    available_from=today - timedelta(days=rng.randint(0, 40)),
                    available_until=today + timedelta(days=rng.randint(90, 320)),
                    status=WasteStatus.AVAILABLE,
                ))
                counts["competitors"] += 1
                out_companies.append([
                    plant.company.company_name, "competitor", material.material_name,
                    plant.plant_name, plant.district,
                    float(plant.latitude), float(plant.longitude),
                    float(plant.company.trust_score),
                ])

        db.flush()

        # ── Trading, replayed in date order ───────────
        # Chronological because the referral term can only be computed from
        # what had already happened.
        sellers: dict[uuid.UUID, list[WasteListing]] = defaultdict(list)
        for listing in db.query(WasteListing).filter(
            WasteListing.material_id.in_(materials.keys()),
            WasteListing.status == WasteStatus.AVAILABLE,
        ).all():
            sellers[listing.material_id].append(listing)

        successes: dict[tuple[uuid.UUID, uuid.UUID], int] = defaultdict(int)
        partners: dict[uuid.UUID, set[uuid.UUID]] = defaultdict(set)
        # Who buys from each seller, so a buyer can be linked to its peers --
        # the other customers of its own suppliers.
        buyers_of: dict[uuid.UUID, set[uuid.UUID]] = defaultdict(set)

        attempts = []
        for plant, material, requirement in buyer_plants:
            candidates = [
                listing for listing in sellers.get(material.id, [])
                if listing.plant_id != plant.id
            ]
            # A subset, not everyone: approaching every seller of a material
            # would connect the market almost completely, which is the shape
            # this is meant to avoid.
            if len(candidates) > SELLERS_CONTACTED_PER_BUYER:
                candidates = rng.sample(candidates, SELLERS_CONTACTED_PER_BUYER)
            for listing in candidates:
                attempts.append((rng.uniform(0, 240), plant, requirement, listing))
        attempts.sort(key=lambda a: a[0])

        for day_offset, buyer_plant, requirement, listing in attempts:
            seller_plant = listing.plant
            if seller_plant is None:
                continue

            distance_km = haversine_distance(
                float(seller_plant.latitude), float(seller_plant.longitude),
                float(buyer_plant.latitude), float(buyer_plant.longitude),
            )
            proximity = 1.0 - min(distance_km / 500.0, 1.0)
            quantity_fit = quantity_fit_pct(
                float(listing.quantity), listing.unit,
                float(requirement.quantity), requirement.unit,
            ) / 100.0
            trust = float(seller_plant.company.trust_score or 60) / 100.0

            pair = (seller_plant.id, buyer_plant.id)
            dyadic = min(successes[pair] / 3.0, 1.0)

            # The referral term: of the buyers this one shares a supplier with,
            # how many already buy from this seller?
            #
            # The reach has to be to peers, not to partners. A buyer's partners
            # are sellers, and sellers never buy from sellers, so asking
            # whether a partner had bought from this seller was a condition
            # that could not be met -- it fired on 0 of 2,880 attempts, and the
            # network effect was simply absent from the data.
            #
            # Peers are reached at two hops: the buyer's suppliers, then the
            # other buyers those suppliers serve. That is how word of mouth
            # actually travels in a market.
            # Expressed as the share of peers endorsing, not how many.
            #
            # A raw count saturates as the market matures: by the last fifth
            # of the timeline 93% of enquiries had at least two endorsements,
            # so the term was near-constant exactly where the held-out split
            # measures it, and predicted acceptance at AUC 0.47 -- below
            # chance. A proportion keeps discriminating, because two
            # endorsements out of three says something two out of twenty does
            # not.
            referral = 0.0
            peers: set[uuid.UUID] = set()
            for supplier in partners[buyer_plant.id]:
                peers |= buyers_of[supplier]
            peers.discard(buyer_plant.id)
            if peers:
                endorsing = sum(
                    1 for peer in peers if successes[(seller_plant.id, peer)] > 0
                )
                referral = endorsing / len(peers)

            z = (
                INTERCEPT
                + W_PROXIMITY * proximity
                + W_QUANTITY * quantity_fit
                + W_TRUST * trust
                + W_DYADIC * dyadic
                + W_REFERRAL * referral
                + rng.gauss(0.0, NOISE_SD)
            )
            accepted = rng.random() < _sigmoid(z)

            traded = min(float(listing.quantity), float(requirement.quantity))
            tons = transport_tons(traded, QuantityUnit.KG)
            carbon_factor = (
                float(listing.material.carbon_factor)
                if listing.material.carbon_factor else None
            )

            request = ExchangeRequest(
                id=uuid.uuid4(),
                supplier_plant_id=seller_plant.id,
                buyer_plant_id=buyer_plant.id,
                waste_listing_id=listing.id,
                requirement_id=requirement.id,
                requested_quantity=Decimal(str(round(traded, 2))),
                compatibility_score=Decimal(str(round(50 + 30 * quantity_fit + 20, 2))),
                material_compatibility=Decimal("100.00"),
                quantity_compatibility=Decimal(str(round(quantity_fit * 100, 2))),
                quality_compatibility=Decimal("100.00"),
                ai_confidence_score=Decimal("50.00"),
                recommendation_rank=1,
                distance_km=Decimal(str(round(distance_km, 2))),
                estimated_transport_cost=Decimal(
                    str(round(estimate_transport_cost(distance_km, tons), 2))
                ),
                estimated_carbon_emission=Decimal(
                    str(round(estimate_transport_emission(distance_km, tons), 2))
                ),
                estimated_carbon_saving=Decimal(
                    str(round(carbon_saving_for_quantity(traded, QuantityUnit.KG, carbon_factor), 2))
                ),
                recommendation_reason="Market exchange between material counterparties.",
                status=(
                    ExchangeRequestStatus.ACCEPTED if accepted
                    else ExchangeRequestStatus.REJECTED
                ),
            )
            when = today - timedelta(days=int(240 - day_offset))
            request.created_at = datetime(when.year, when.month, when.day)
            db.add(request)
            counts["requests"] += 1
            out_exchanges.append([
                seller_plant.company.company_name,
                buyer_plant.company.company_name,
                listing.material.material_name,
                round(traded, 2), round(distance_km, 2),
                round(quantity_fit, 3), round(referral, 3),
                "Accepted" if accepted else "Rejected",
                request.created_at.date().isoformat(),
            ])

            if not accepted:
                continue

            db.flush()
            counts["accepted"] += 1
            successes[pair] += 1
            partners[buyer_plant.id].add(seller_plant.id)
            partners[seller_plant.id].add(buyer_plant.id)
            buyers_of[seller_plant.id].add(buyer_plant.id)

            exchange = Exchange(
                id=uuid.uuid4(),
                exchange_request_id=request.id,
                exchange_status=ExchangeStatus.COMPLETED,
                shipment_status=ShipmentStatus.DELIVERED,
                agreed_price=Decimal(str(round(traded * float(listing.price_per_unit or 0), 2))),
                transport_cost=request.estimated_transport_cost,
                actual_quantity=request.requested_quantity,
                actual_carbon_emission=request.estimated_carbon_emission,
                actual_carbon_saving=request.estimated_carbon_saving,
            )
            db.add(exchange)
            db.flush()

            rating = rng.choices([5, 4, 3, 2], weights=[45, 35, 15, 5], k=1)[0]
            db.add(Review(
                id=uuid.uuid4(),
                exchange_id=exchange.id,
                supplier_rating=rating,
                buyer_rating=rng.choices([5, 4, 3], weights=[50, 35, 15], k=1)[0],
                supplier_feedback="Material as described.",
                buyer_feedback="Collection and payment went smoothly.",
            ))

        _write_csv(
            "market_companies.csv",
            ["company_name", "role", "material", "plant_name", "district",
             "latitude", "longitude", "trust_score"],
            out_companies,
        )
        _write_csv(
            "market_requirements.csv",
            ["company_name", "material", "quantity_kg", "unit",
             "minimum_purity", "max_budget_per_kg_inr", "required_before"],
            out_requirements,
        )
        _write_csv(
            "market_exchanges.csv",
            ["supplier", "buyer", "material", "quantity_kg", "distance_km",
             "quantity_fit", "referral_strength", "outcome", "date"],
            out_exchanges,
        )

        db.commit()
        logger.info("Market written to %s", OUTPUT_DIR)
        logger.info(
            "Market partners: %d buyers, %d competitors, %d requests (%d accepted)",
            counts["buyers"], counts["competitors"], counts["requests"], counts["accepted"],
        )
        return dict(counts)
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_market_partners()
