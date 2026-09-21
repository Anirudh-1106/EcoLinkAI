"""
Recommendation service — the core AI-powered partner matching engine.

This service handles:
1. Candidate generation (filtering compatible demand/supply)
2. Feature computation for ranking
3. AI model inference (MC-GNN) when available
4. Fallback to baseline weighted scoring
5. Explainability layer
"""

from __future__ import annotations

import logging
import time
import uuid
from decimal import Decimal

from sqlalchemy import func, or_
from sqlalchemy.orm import Session, joinedload

from app.enums.common import QuantityUnit
from app.enums.exchange import (
    ExchangeRequestStatus,
    RequirementStatus,
    WasteStatus,
)
from app.models.company import Company
from app.models.exchange import Exchange
from app.models.exchange_request import ExchangeRequest
from app.models.material import Material
from app.models.plant import Plant
from app.models.requirement import Requirement
from app.models.review import Review
from app.models.waste_listing import WasteListing
from app.schemas.recommendation import (
    PartnerCard,
    PartnerExplanation,
    RecommendationRequest,
    RecommendationResponse,
    RequirementRecommendationRequest,
    RequirementRecommendationResponse,
    SearchRecommendationRequest,
    SearchRecommendationResponse,
)
from app.services import graph_cache
from app.utils.distance import (
    estimate_transport_cost,
    estimate_transport_emission,
    haversine_distance,
)
from app.utils.quantity import (
    carbon_saving_for_quantity,
    price_per_kg,
    quantity_fit_pct,
    transport_tons,
)

logger = logging.getLogger(__name__)

# ── AI model versions ────────────────────────────────
MC_GNN_VERSION = "mc-gnn-v1.0"
BASELINE_VERSION = "baseline-v1.0"

# Material compatibility awarded when a listing is not the material the buyer
# asked for but belongs to the same category. High enough for such sellers to
# be surfaced, low enough that any exact match outranks them.
CATEGORY_MATCH_COMPAT = 60.0

_model_version = BASELINE_VERSION


def _try_load_model():
    """
    Load the trained MC-GNN checkpoint at application startup.

    Only reports whether the checkpoint itself is loadable; the graph and
    node embeddings needed for inference are built lazily (or warmed) by
    app.services.graph_cache.
    """
    global _model_version
    model = graph_cache._load_model()
    _model_version = MC_GNN_VERSION if model is not None else BASELINE_VERSION
    return model is not None


def composite_compatibility(
    material_compat: float, quantity_compat: float, quality_compat: float
) -> float:
    """
    Blend the three compatibility signals into a single 0-100 score.

    This is the compatibility the MC-GNN consumes as an edge feature and that
    gets stored on an exchange request. The identical blend is applied when
    seeding historical requests (scripts/database/seed.py), so the model is
    trained on and served the same quantity -- feeding it a figure here that
    was computed differently during training would leave it reading a signal
    it never actually learned.
    """
    score = 0.5 * material_compat + 0.3 * quantity_compat + 0.2 * quality_compat
    return max(0.0, min(score, 100.0))


def _gnn_link_score(
    db: Session,
    *,
    seller_plant_id,
    buyer_plant_id,
    distance_km: float,
    compatibility: float,
    transport_cost: float,
    carbon_saving: float,
) -> float | None:
    """
    Score one seller -> buyer pair with the trained MC-GNN link predictor.

    Edges are directed supplier -> buyer during training, so the same
    orientation is used here. Edge features are normalised exactly as in
    ai/features/edge_features.py; any divergence would feed the model
    inputs it never saw during training.

    Returns a 0-1 probability, or None when GNN inference is unavailable
    (no checkpoint, torch missing, or a plant absent from the cached graph).
    """
    context = graph_cache.get_context(db)
    if context is None:
        return None

    plant_id_to_idx = context["plant_id_to_idx"]
    src = plant_id_to_idx.get(str(seller_plant_id))
    dst = plant_id_to_idx.get(str(buyer_plant_id))
    if src is None or dst is None:
        # Plant registered after the cached graph was built.
        return None

    try:
        import torch

        embeddings = context["embeddings"]
        edge_attr = torch.tensor(
            [[
                min(distance_km / 500.0, 1.0),
                compatibility / 100.0,
                transport_cost / 10000.0,
                carbon_saving / 1000.0,
            ]],
            dtype=torch.float,
        )

        with torch.no_grad():
            features = torch.cat(
                [embeddings[src].unsqueeze(0), embeddings[dst].unsqueeze(0), edge_attr],
                dim=-1,
            )
            return float(context["model"].link_predictor(features).item())
    except Exception as e:
        logger.warning("MC-GNN inference failed, falling back to baseline: %s", e)
        return None


def get_recommendations(
    db: Session,
    request: RecommendationRequest,
) -> RecommendationResponse:
    """
    Generate ranked partner recommendations for a waste listing.

    Pipeline:
    1. Load waste listing and supplier plant
    2. Find candidate buyer plants with matching requirements
    3. Compute features (distance, compatibility, trust, carbon)
    4. Score candidates (MC-GNN if available, else baseline)
    5. Rank and build explanation cards
    """
    start_time = time.time()

    # ── 1. Load the waste listing ─────────────────────
    listing = (
        db.query(WasteListing)
        .options(
            joinedload(WasteListing.plant).joinedload(Plant.company),
            joinedload(WasteListing.material),
        )
        .filter(WasteListing.id == request.waste_listing_id)
        .first()
    )

    if not listing:
        raise ValueError("Waste listing not found")

    supplier_plant = listing.plant
    material = listing.material

    # ── 2. Candidate generation ───────────────────────
    candidates = _generate_candidates(
        db,
        listing=listing,
        supplier_plant=supplier_plant,
        material=material,
        max_distance_km=float(request.max_distance_km) if request.max_distance_km else None,
    )

    # ── 3-4. Score candidates ─────────────────────────
    scored = []
    for candidate in candidates:
        score_data = _score_candidate(
            db,
            listing=listing,
            supplier_plant=supplier_plant,
            candidate=candidate,
        )
        scored.append(score_data)

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # ── 5. Build partner cards ────────────────────────
    recommendations = []
    for rank, item in enumerate(scored[: request.max_results], start=1):
        card = _build_partner_card(rank, item, _model_version)
        recommendations.append(card)

    elapsed_ms = (time.time() - start_time) * 1000

    return RecommendationResponse(
        waste_listing_id=listing.id,
        material_name=material.material_name,
        supplier_plant_name=supplier_plant.plant_name,
        supplier_plant_latitude=supplier_plant.latitude,
        supplier_plant_longitude=supplier_plant.longitude,
        total_candidates=len(candidates),
        recommendations=recommendations,
        model_version=_model_version,
        inference_time_ms=round(elapsed_ms, 2),
    )


def score_candidate_pair(
    db: Session,
    *,
    waste_listing_id: uuid.UUID,
    buyer_plant_id: uuid.UUID,
    requirement_id: uuid.UUID | None = None,
) -> dict:
    """
    Compute real match metrics for one specific supplier-buyer pair.

    Used when a user sends a direct exchange request, so the stored
    compatibility/distance/carbon figures reflect the same computation
    that produced the recommendation card they acted on, rather than
    trusting client-supplied numbers or falling back to placeholders.
    """
    listing = (
        db.query(WasteListing)
        .options(
            joinedload(WasteListing.plant).joinedload(Plant.company),
            joinedload(WasteListing.material),
        )
        .filter(WasteListing.id == waste_listing_id)
        .first()
    )
    if not listing:
        raise ValueError("Waste listing not found")

    supplier_plant = listing.plant

    buyer_plant = (
        db.query(Plant)
        .options(joinedload(Plant.company))
        .filter(Plant.id == buyer_plant_id)
        .first()
    )
    if not buyer_plant:
        raise ValueError("Buyer plant not found")

    requirement = None
    if requirement_id:
        requirement = (
            db.query(Requirement)
            .options(joinedload(Requirement.material))
            .filter(Requirement.id == requirement_id)
            .first()
        )

    distance_km = haversine_distance(
        float(supplier_plant.latitude),
        float(supplier_plant.longitude),
        float(buyer_plant.latitude),
        float(buyer_plant.longitude),
    )

    candidate = {
        "requirement": requirement,
        "buyer_plant": buyer_plant,
        "buyer_company": buyer_plant.company,
        "distance_km": distance_km,
    }

    scored = _score_candidate(db, listing=listing, supplier_plant=supplier_plant, candidate=candidate)
    features = scored["features"]
    return {
        "ai_score": round(scored["score"], 2),
        "compatibility_score": features["compatibility"],
        "distance_km": features["distance_km"],
        "estimated_transport_cost": features["transport_cost"],
        "estimated_carbon_emission": features["transport_emission"],
        "estimated_carbon_saving": features["carbon_saving"],
    }


def _generate_candidates(
    db: Session,
    *,
    listing: WasteListing,
    supplier_plant: Plant,
    material: Material,
    max_distance_km: float | None = None,
) -> list[dict]:
    """
    Generate candidate buyer plants based on:
    - Matching material requirements (same material_id)
    - Open requirement status
    - Different company (no self-exchange)
    - Geographic feasibility
    """
    # Find open requirements for the same material
    # Extract base material name by removing the (M...) suffix
    import re
    base_material_name = re.sub(r' \([A-Z0-9]+\)$', '', material.material_name)

    requirements = (
        db.query(Requirement)
        .options(
            joinedload(Requirement.plant).joinedload(Plant.company),
        )
        .join(Material, Requirement.material_id == Material.id)
        .filter(
            Material.material_name.like(base_material_name + '%'),
            Requirement.status == RequirementStatus.OPEN,
        )
        .all()
    )

    candidates = []
    for req in requirements:
        buyer_plant = req.plant
        # Skip same company
        if str(buyer_plant.company_id) == str(supplier_plant.company_id):
            continue

        # Compute distance
        dist = haversine_distance(
            float(supplier_plant.latitude),
            float(supplier_plant.longitude),
            float(buyer_plant.latitude),
            float(buyer_plant.longitude),
        )

        # Geographic filter
        if max_distance_km and dist > max_distance_km:
            continue

        candidates.append({
            "requirement": req,
            "buyer_plant": buyer_plant,
            "buyer_company": buyer_plant.company,
            "distance_km": dist,
        })

    # If no requirements match, find plants from other companies
    # that have historically used this material (broader search)
    if not candidates:
        other_plants = (
            db.query(Plant)
            .options(joinedload(Plant.company))
            .filter(
                Plant.company_id != supplier_plant.company_id,
                Plant.is_active.is_(True),
            )
            .all()
        )
        for plant in other_plants:
            dist = haversine_distance(
                float(supplier_plant.latitude),
                float(supplier_plant.longitude),
                float(plant.latitude),
                float(plant.longitude),
            )
            if max_distance_km and dist > max_distance_km:
                continue

            candidates.append({
                "requirement": None,
                "buyer_plant": plant,
                "buyer_company": plant.company,
                "distance_km": dist,
            })

    return candidates


def _score_candidate(
    db: Session,
    *,
    listing: WasteListing,
    supplier_plant: Plant,
    candidate: dict,
) -> dict:
    """
    Score a single candidate using baseline weighted scoring.
    When MC-GNN is loaded, this can be replaced with model inference.
    """
    buyer_plant = candidate["buyer_plant"]
    buyer_company = candidate["buyer_company"]
    requirement = candidate["requirement"]
    distance_km = candidate["distance_km"]

    # ── Feature computation ───────────────────────────

    # Material compatibility (exact match = 100, no req = 50)
    material_compat = 100.0 if requirement else 50.0

    # Quantity compatibility -- converted to a common unit first; a listing
    # in tons and a requirement in kg are not the same number.
    if requirement:
        quantity_compat = quantity_fit_pct(
            float(listing.quantity), listing.unit,
            float(requirement.quantity), requirement.unit,
        )
    else:
        quantity_compat = 50.0

    # Quality compatibility
    quality_compat = 100.0
    if requirement and requirement.minimum_purity and listing.purity_percentage:
        if float(listing.purity_percentage) >= float(requirement.minimum_purity):
            quality_compat = 100.0
        else:
            quality_compat = (
                float(listing.purity_percentage)
                / float(requirement.minimum_purity)
                * 100
            )

    # Distance score (closer = better, max 500km reference)
    distance_score = max(0, (1 - distance_km / 500.0)) * 100

    # Trust score
    trust = float(buyer_company.trust_score) if buyer_company.trust_score else 50.0

    # Historical exchange count between these companies
    hist_count = (
        db.query(func.count(ExchangeRequest.id))
        .filter(
            ExchangeRequest.status == ExchangeRequestStatus.ACCEPTED,
            (
                (ExchangeRequest.supplier_plant_id == supplier_plant.id)
                & (ExchangeRequest.buyer_plant_id == buyer_plant.id)
            )
            | (
                (ExchangeRequest.supplier_plant_id == buyer_plant.id)
                & (ExchangeRequest.buyer_plant_id == supplier_plant.id)
            ),
        )
        .scalar()
        or 0
    )
    history_score = min(hist_count * 10, 100)

    # Transport cost
    quantity_tons = transport_tons(float(listing.quantity), listing.unit)
    transport_cost = estimate_transport_cost(distance_km, quantity_tons)
    transport_emission = estimate_transport_emission(distance_km, quantity_tons)

    # Carbon saving
    carbon_factor = (
        float(listing.material.carbon_factor)
        if listing.material.carbon_factor
        else None
    )
    carbon_saving = carbon_saving_for_quantity(float(listing.quantity), listing.unit, carbon_factor)

    # ── Baseline weighted score ───────────────────────
    # Weights from constants/ai.py
    baseline_score = (
        0.30 * material_compat
        + 0.15 * quantity_compat
        + 0.10 * quality_compat
        + 0.15 * distance_score
        + 0.10 * trust
        + 0.10 * history_score
        + 0.10 * min(carbon_saving / 100, 100)
    )
    baseline_score = min(max(baseline_score, 0), 100)

    # ── MC-GNN inference (falls back to baseline) ─────
    compatibility = composite_compatibility(
        material_compat, quantity_compat, quality_compat
    )
    gnn_prob = _gnn_link_score(
        db,
        seller_plant_id=supplier_plant.id,
        buyer_plant_id=buyer_plant.id,
        distance_km=distance_km,
        compatibility=compatibility,
        transport_cost=transport_cost,
        carbon_saving=carbon_saving,
    )

    if gnn_prob is not None:
        score = gnn_prob * 100.0
        scoring_method = "mc_gnn"
    else:
        score = baseline_score
        scoring_method = "baseline"

    # Transport feasibility label
    if distance_km < 100:
        transport_feas = "Excellent"
    elif distance_km < 250:
        transport_feas = "High"
    elif distance_km < 500:
        transport_feas = "Moderate"
    else:
        transport_feas = "Low"

    return {
        "score": score,
        "candidate": candidate,
        "features": {
            "material_compat": material_compat,
            "quantity_compat": quantity_compat,
            "quality_compat": quality_compat,
            "compatibility": round(compatibility, 2),
            "distance_km": round(distance_km, 2),
            "distance_score": distance_score,
            "trust": trust,
            "history_score": history_score,
            "hist_count": hist_count,
            "transport_cost": round(transport_cost, 2),
            "transport_emission": round(transport_emission, 2),
            "carbon_saving": round(carbon_saving, 2),
            "transport_feasibility": transport_feas,
            "scoring_method": scoring_method,
            "baseline_score": round(baseline_score, 2),
        },
    }


def _build_partner_card(rank: int, scored: dict, model_version: str) -> PartnerCard:
    """Build a PartnerCard from scored candidate data."""
    candidate = scored["candidate"]
    features = scored["features"]
    buyer_plant = candidate["buyer_plant"]
    buyer_company = candidate["buyer_company"]
    requirement = candidate["requirement"]

    # Build explanation summary
    reasons = []
    if features["material_compat"] >= 90:
        reasons.append("excellent material compatibility")
    elif features["material_compat"] >= 70:
        reasons.append("good material compatibility")

    if features["distance_km"] < 100:
        reasons.append(f"close proximity ({features['distance_km']:.0f} km)")
    elif features["distance_km"] < 250:
        reasons.append(f"reasonable distance ({features['distance_km']:.0f} km)")

    if features["trust"] >= 80:
        reasons.append("high trust rating")

    if features["hist_count"] > 0:
        reasons.append(f"{features['hist_count']} previous exchanges")

    if features["carbon_saving"] > 500:
        reasons.append(f"significant carbon benefit ({features['carbon_saving']:.0f} kg CO₂e)")

    summary = (
        "Recommended because of "
        + ", ".join(reasons)
        + "."
        if reasons
        else "Potential partner identified based on material and location compatibility."
    )

    explanation = PartnerExplanation(
        material_compatibility=(
            "Exact Match" if features["material_compat"] >= 90
            else "Compatible" if features["material_compat"] >= 50
            else "Partial"
        ),
        quantity_match_pct=round(features["quantity_compat"], 1),
        quality_match_pct=round(features.get("quality_compat", 0), 1),
        distance_km=features["distance_km"],
        transport_feasibility=features["transport_feasibility"],
        estimated_transport_cost=features["transport_cost"],
        trust_score=features["trust"],
        carbon_benefit_kg=features["carbon_saving"],
        historical_exchange_count=features["hist_count"],
        recommendation_summary=summary,
    )

    return PartnerCard(
        rank=rank,
        ai_score=round(scored["score"], 2),
        model_type=features.get("scoring_method", "baseline"),
        company_id=buyer_company.id,
        company_name=buyer_company.company_name,
        plant_id=buyer_plant.id,
        plant_name=buyer_plant.plant_name,
        plant_district=buyer_plant.district,
        plant_state=buyer_plant.state,
        plant_latitude=buyer_plant.latitude,
        plant_longitude=buyer_plant.longitude,
        requirement_id=requirement.id if requirement else None,
        required_quantity=requirement.quantity if requirement else None,
        required_purity=requirement.minimum_purity if requirement else None,
        material_name=requirement.material.material_name if requirement and requirement.material else "N/A",
        compatibility_score=features["material_compat"],
        distance_km=features["distance_km"],
        estimated_transport_cost=features["transport_cost"],
        estimated_carbon_saving=features["carbon_saving"],
        explanation=explanation,
    )


# ══════════════════════════════════════════════════════════
# BUYER-INITIATED FLOW: Find sellers for a requirement
# ══════════════════════════════════════════════════════════


def get_recommendations_by_requirement(
    db: Session,
    request: RequirementRecommendationRequest,
) -> RequirementRecommendationResponse:
    """
    Generate ranked seller recommendations for a buyer's material requirement.

    Pipeline:
    1. Load requirement and buyer plant
    2. Find candidate waste listings with matching material
    3. Compute features (distance, compatibility, trust, carbon)
    4. Score candidates (MC-GNN if available, else baseline)
    5. Rank and build seller partner cards
    """
    start_time = time.time()

    # ── 1. Load the requirement ───────────────────────
    requirement = (
        db.query(Requirement)
        .options(
            joinedload(Requirement.plant).joinedload(Plant.company),
            joinedload(Requirement.material),
        )
        .filter(Requirement.id == request.requirement_id)
        .first()
    )

    if not requirement:
        raise ValueError("Requirement not found")

    buyer_plant = requirement.plant
    material = requirement.material

    # ── 2. Candidate generation ───────────────────────
    candidates = _generate_seller_candidates(
        db,
        requirement=requirement,
        buyer_plant=buyer_plant,
        material=material,
        max_distance_km=float(request.max_distance_km) if request.max_distance_km else None,
    )

    # ── 3-4. Score candidates ─────────────────────────
    scored = []
    for candidate in candidates:
        score_data = _score_seller_candidate(
            db,
            requirement=requirement,
            buyer_plant=buyer_plant,
            candidate=candidate,
        )
        scored.append(score_data)

    # Sort by score descending. Category-only matches already carry their
    # material penalty in the score, so they settle below comparable exact
    # matches without needing to be grouped separately.
    scored.sort(key=lambda x: x["score"], reverse=True)

    # ── 5. Build partner cards ────────────────────────
    recommendations = []
    for rank, item in enumerate(scored[: request.max_results], start=1):
        card = _build_seller_partner_card(rank, item, _model_version)
        recommendations.append(card)

    elapsed_ms = (time.time() - start_time) * 1000

    return RequirementRecommendationResponse(
        requirement_id=requirement.id,
        material_name=material.material_name,
        buyer_plant_name=buyer_plant.plant_name,
        buyer_plant_latitude=buyer_plant.latitude,
        buyer_plant_longitude=buyer_plant.longitude,
        total_candidates=len(candidates),
        recommendations=recommendations,
        model_version=_model_version,
        inference_time_ms=round(elapsed_ms, 2),
    )


def score_seller_candidate_pair(
    db: Session,
    *,
    requirement_id: uuid.UUID,
    waste_listing_id: uuid.UUID,
    seller_plant_id: uuid.UUID,
) -> dict:
    """
    Compute real match metrics for one specific buyer-seller pair.

    Used when a buyer sends a direct exchange request from the
    recommendation page, so stored compatibility/distance/carbon
    figures are accurate.
    """
    requirement = (
        db.query(Requirement)
        .options(
            joinedload(Requirement.plant).joinedload(Plant.company),
            joinedload(Requirement.material),
        )
        .filter(Requirement.id == requirement_id)
        .first()
    )
    if not requirement:
        raise ValueError("Requirement not found")

    buyer_plant = requirement.plant

    listing = (
        db.query(WasteListing)
        .options(
            joinedload(WasteListing.plant).joinedload(Plant.company),
            joinedload(WasteListing.material),
        )
        .filter(WasteListing.id == waste_listing_id)
        .first()
    )
    if not listing:
        raise ValueError("Waste listing not found")

    seller_plant = listing.plant

    distance_km = haversine_distance(
        float(buyer_plant.latitude),
        float(buyer_plant.longitude),
        float(seller_plant.latitude),
        float(seller_plant.longitude),
    )

    candidate = {
        "listing": listing,
        "seller_plant": seller_plant,
        "seller_company": seller_plant.company,
        "distance_km": distance_km,
    }

    scored = _score_seller_candidate(
        db, requirement=requirement, buyer_plant=buyer_plant, candidate=candidate
    )
    features = scored["features"]
    return {
        "ai_score": round(scored["score"], 2),
        "compatibility_score": features["compatibility"],
        "distance_km": features["distance_km"],
        "estimated_transport_cost": features["transport_cost"],
        "estimated_carbon_emission": features["transport_emission"],
        "estimated_carbon_saving": features["carbon_saving"],
    }


def _generate_seller_candidates(
    db: Session,
    *,
    requirement: Requirement,
    buyer_plant: Plant,
    material: Material,
    max_distance_km: float | None = None,
) -> list[dict]:
    """
    Generate candidate seller plants/waste listings based on:
    - Matching material (same material_id)
    - Available listing status
    - Different company (no self-exchange)
    - Geographic feasibility
    """
    # Extract base material name by removing the (M...) suffix
    import re
    base_material_name = re.sub(r' \([A-Z0-9]+\)$', '', material.material_name)

    # Match on the material's name first, but accept anything in the same
    # category too. A buyer needing "Metal Waste" should still be shown
    # "Metal Offcuts" -- the same kind of material under another name --
    # rather than nothing at all. Category-only matches are scored lower
    # below, so exact matches still rank above them.
    listings = (
        db.query(WasteListing)
        .options(
            joinedload(WasteListing.plant).joinedload(Plant.company),
            joinedload(WasteListing.material),
        )
        .join(Material, WasteListing.material_id == Material.id)
        .filter(
            or_(
                Material.material_name.like(base_material_name + '%'),
                Material.material_category == material.material_category,
            ),
            WasteListing.status == WasteStatus.AVAILABLE,
        )
        .all()
    )

    candidates = []
    for listing in listings:
        seller_plant = listing.plant
        # Skip same company
        if str(seller_plant.company_id) == str(buyer_plant.company_id):
            continue

        # Compute distance
        dist = haversine_distance(
            float(buyer_plant.latitude),
            float(buyer_plant.longitude),
            float(seller_plant.latitude),
            float(seller_plant.longitude),
        )

        # Geographic filter
        if max_distance_km and dist > max_distance_km:
            continue

        listing_name = listing.material.material_name if listing.material else ""
        candidates.append({
            "listing": listing,
            "seller_plant": seller_plant,
            "seller_company": seller_plant.company,
            "distance_km": dist,
            "material_match": (
                "exact" if listing_name.startswith(base_material_name) else "category"
            ),
        })

    return candidates


def _score_seller_candidate(
    db: Session,
    *,
    requirement: Requirement,
    buyer_plant: Plant,
    candidate: dict,
) -> dict:
    """
    Score a single seller candidate using baseline weighted scoring.
    Mirrors _score_candidate() but from the buyer's perspective.
    """
    listing = candidate["listing"]
    seller_plant = candidate["seller_plant"]
    seller_company = candidate["seller_company"]
    distance_km = candidate["distance_km"]

    # ── Feature computation ───────────────────────────

    # An exact material match is worth full marks; a different material in the
    # same category is a real but weaker option, so it scores lower and ranks
    # below the exact matches rather than being hidden entirely.
    material_compat = (
        100.0 if candidate.get("material_match", "exact") == "exact"
        else CATEGORY_MATCH_COMPAT
    )

    # Quantity compatibility -- converted to a common unit first; a listing
    # in tons and a requirement in kg are not the same number.
    quantity_compat = quantity_fit_pct(
        float(listing.quantity), listing.unit,
        float(requirement.quantity), requirement.unit,
    )

    # Quality compatibility
    quality_compat = 100.0
    if requirement.minimum_purity and listing.purity_percentage:
        if float(listing.purity_percentage) >= float(requirement.minimum_purity):
            quality_compat = 100.0
        else:
            quality_compat = (
                float(listing.purity_percentage)
                / float(requirement.minimum_purity)
                * 100
            )

    # Price compatibility (if buyer has a budget and seller has a price).
    # Both prices are "per {unit}", so they're normalised to price-per-kg
    # before comparing -- a ton-priced listing at Rs91/ton is Rs0.091/kg,
    # not 91x a kg-priced budget of Rs50/kg.
    price_compat = 100.0
    if requirement.maximum_budget_per_unit and listing.price_per_unit:
        listing_price_kg = price_per_kg(float(listing.price_per_unit), listing.unit)
        budget_price_kg = price_per_kg(
            float(requirement.maximum_budget_per_unit), requirement.unit
        )
        if listing_price_kg is not None and budget_price_kg is not None:
            if listing_price_kg <= budget_price_kg:
                price_compat = 100.0
            else:
                price_compat = (budget_price_kg / listing_price_kg) * 100

    # Distance score (closer = better, max 500km reference)
    distance_score = max(0, (1 - distance_km / 500.0)) * 100

    # Trust score
    trust = float(seller_company.trust_score) if seller_company.trust_score else 50.0

    # Historical exchange count between these companies
    hist_count = (
        db.query(func.count(ExchangeRequest.id))
        .filter(
            ExchangeRequest.status == ExchangeRequestStatus.ACCEPTED,
            (
                (ExchangeRequest.supplier_plant_id == seller_plant.id)
                & (ExchangeRequest.buyer_plant_id == buyer_plant.id)
            )
            | (
                (ExchangeRequest.supplier_plant_id == buyer_plant.id)
                & (ExchangeRequest.buyer_plant_id == seller_plant.id)
            ),
        )
        .scalar()
        or 0
    )
    history_score = min(hist_count * 10, 100)

    # Transport cost
    quantity_tons = transport_tons(float(listing.quantity), listing.unit)
    transport_cost = estimate_transport_cost(distance_km, quantity_tons)
    transport_emission = estimate_transport_emission(distance_km, quantity_tons)

    # Carbon saving
    carbon_factor = (
        float(listing.material.carbon_factor)
        if listing.material.carbon_factor
        else None
    )
    carbon_saving = carbon_saving_for_quantity(float(listing.quantity), listing.unit, carbon_factor)

    # ── Baseline weighted score ───────────────────────
    baseline_score = (
        0.25 * material_compat
        + 0.15 * quantity_compat
        + 0.10 * quality_compat
        + 0.10 * min(price_compat, 100)
        + 0.15 * distance_score
        + 0.10 * trust
        + 0.05 * history_score
        + 0.10 * min(carbon_saving / 100, 100)
    )
    baseline_score = min(max(baseline_score, 0), 100)

    # ── MC-GNN inference (falls back to baseline) ─────
    compatibility = composite_compatibility(
        material_compat, quantity_compat, quality_compat
    )
    gnn_prob = _gnn_link_score(
        db,
        seller_plant_id=seller_plant.id,
        buyer_plant_id=buyer_plant.id,
        distance_km=distance_km,
        compatibility=compatibility,
        transport_cost=transport_cost,
        carbon_saving=carbon_saving,
    )

    if gnn_prob is not None:
        score = gnn_prob * 100.0
        scoring_method = "mc_gnn"
        # Every request the model trained on was for an exactly matching
        # material, so it never learned what a weaker material fit should cost
        # a seller, and its score here is an extrapolation. Apply that penalty
        # explicitly rather than presenting an unreliable number as if it were
        # learned. The baseline already weighs material_compat in its own sum,
        # so it needs no such adjustment.
        score *= material_compat / 100.0
    else:
        score = baseline_score
        scoring_method = "baseline"

    # Transport feasibility label
    if distance_km < 100:
        transport_feas = "Excellent"
    elif distance_km < 250:
        transport_feas = "High"
    elif distance_km < 500:
        transport_feas = "Moderate"
    else:
        transport_feas = "Low"

    return {
        "score": score,
        "candidate": candidate,
        "features": {
            "material_compat": material_compat,
            "quantity_compat": quantity_compat,
            "quality_compat": quality_compat,
            "compatibility": round(compatibility, 2),
            "price_compat": price_compat,
            "distance_km": round(distance_km, 2),
            "distance_score": distance_score,
            "trust": trust,
            "history_score": history_score,
            "hist_count": hist_count,
            "transport_cost": round(transport_cost, 2),
            "transport_emission": round(transport_emission, 2),
            "carbon_saving": round(carbon_saving, 2),
            "transport_feasibility": transport_feas,
            "scoring_method": scoring_method,
            "baseline_score": round(baseline_score, 2),
        },
    }


def _build_seller_partner_card(rank: int, scored: dict, model_version: str) -> PartnerCard:
    """Build a PartnerCard from scored seller candidate data."""
    candidate = scored["candidate"]
    features = scored["features"]
    listing = candidate["listing"]
    seller_plant = candidate["seller_plant"]
    seller_company = candidate["seller_company"]

    # Build explanation summary
    reasons = []
    if features["material_compat"] >= 90:
        reasons.append("exact material match")
    else:
        # Say so plainly: this seller offers a related material, not the one
        # that was asked for, and the buyer needs to see that before acting.
        reasons.append(
            f"related material in the same category ({listing.material.material_category.value})"
            if listing.material
            else "related material in the same category"
        )

    if features["quantity_compat"] >= 80:
        reasons.append(f"strong quantity fit ({features['quantity_compat']:.0f}%)")

    if features.get("quality_compat", 0) >= 90:
        reasons.append("meets purity requirements")

    if features.get("price_compat", 0) >= 90:
        reasons.append("within budget")

    if features["distance_km"] < 100:
        reasons.append(f"close proximity ({features['distance_km']:.0f} km)")
    elif features["distance_km"] < 250:
        reasons.append(f"reasonable distance ({features['distance_km']:.0f} km)")

    if features["trust"] >= 80:
        reasons.append("high trust rating")

    if features["hist_count"] > 0:
        reasons.append(f"{features['hist_count']} previous exchanges")

    if features["carbon_saving"] > 500:
        reasons.append(f"significant carbon benefit ({features['carbon_saving']:.0f} kg CO₂e)")

    summary = (
        "Recommended because of "
        + ", ".join(reasons)
        + "."
        if reasons
        else "Potential seller identified based on material and location compatibility."
    )

    explanation = PartnerExplanation(
        material_compatibility=(
            "Exact Match" if features["material_compat"] >= 90
            else "Same Category" if features["material_compat"] >= 50
            else "Partial"
        ),
        quantity_match_pct=round(features["quantity_compat"], 1),
        quality_match_pct=round(features.get("quality_compat", 0), 1),
        distance_km=features["distance_km"],
        transport_feasibility=features["transport_feasibility"],
        estimated_transport_cost=features["transport_cost"],
        trust_score=features["trust"],
        carbon_benefit_kg=features["carbon_saving"],
        historical_exchange_count=features["hist_count"],
        recommendation_summary=summary,
    )

    return PartnerCard(
        rank=rank,
        ai_score=round(scored["score"], 2),
        model_type=features.get("scoring_method", "baseline"),
        company_id=seller_company.id,
        company_name=seller_company.company_name,
        plant_id=seller_plant.id,
        plant_name=seller_plant.plant_name,
        plant_district=seller_plant.district,
        plant_state=seller_plant.state,
        plant_latitude=seller_plant.latitude,
        plant_longitude=seller_plant.longitude,
        waste_listing_id=listing.id,
        listing_quantity=listing.quantity,
        listing_purity=listing.purity_percentage,
        listing_price_per_unit=listing.price_per_unit,
        listing_unit=listing.unit,
        listing_quality_grade=listing.quality_grade,
        material_name=listing.material.material_name if listing.material else "N/A",
        compatibility_score=features["material_compat"],
        distance_km=features["distance_km"],
        estimated_transport_cost=features["transport_cost"],
        estimated_carbon_saving=features["carbon_saving"],
        explanation=explanation,
    )


def get_recommendations_by_search(
    db: Session,
    request: SearchRecommendationRequest,
) -> SearchRecommendationResponse:
    """
    Generate ranked seller recommendations from a free-text material search.

    Creates a 'ghost requirement' in memory (not persisted) so the existing
    scoring pipeline can be reused. This powers the Discover Sellers search bar.
    """
    start_time = time.time()

    # ── 1. Load buyer plant ───────────────────────────
    buyer_plant = (
        db.query(Plant)
        .options(joinedload(Plant.company))
        .filter(Plant.id == request.buyer_plant_id)
        .first()
    )
    if not buyer_plant:
        raise ValueError("Buyer plant not found")

    # ── 2. Find matching waste listings via ILIKE ─────
    listings = (
        db.query(WasteListing)
        .options(
            joinedload(WasteListing.plant).joinedload(Plant.company),
            joinedload(WasteListing.material),
        )
        .join(Material, WasteListing.material_id == Material.id)
        .filter(
            Material.material_name.ilike(f"%{request.material_query}%"),
            WasteListing.status == WasteStatus.AVAILABLE,
        )
        .all()
    )

    # ── 3. Build candidates with ghost requirement ────
    candidates = []
    for listing in listings:
        seller_plant = listing.plant
        # Skip same company
        if str(seller_plant.company_id) == str(buyer_plant.company_id):
            continue

        dist = haversine_distance(
            float(buyer_plant.latitude),
            float(buyer_plant.longitude),
            float(seller_plant.latitude),
            float(seller_plant.longitude),
        )

        # Geographic filter
        if request.max_distance_km and dist > float(request.max_distance_km):
            continue

        candidates.append({
            "listing": listing,
            "seller_plant": seller_plant,
            "seller_company": seller_plant.company,
            "distance_km": dist,
        })

    # ── 4. Score using ghost requirement ──────────────
    # Build a lightweight ghost requirement object for scoring
    ghost_quantity = request.quantity_needed if request.quantity_needed else None

    scored = []
    for candidate in candidates:
        listing = candidate["listing"]
        seller_plant = candidate["seller_plant"]
        seller_company = candidate["seller_company"]
        distance_km = candidate["distance_km"]

        # Material compatibility is always 100 (matched by search)
        material_compat = 100.0

        # Quantity compatibility. The search filter's "quantity needed" has no
        # unit field of its own -- the UI labels it in kg -- so it's compared
        # against the listing's quantity converted to kg, not the raw number.
        if ghost_quantity:
            quantity_compat = quantity_fit_pct(
                float(listing.quantity), listing.unit,
                ghost_quantity, QuantityUnit.KG,
            )
        else:
            quantity_compat = 100.0  # User didn't specify, assume perfect match

        # Quality — accept anything
        quality_compat = 100.0

        # Price — no budget constraint from search
        price_compat = 100.0

        # Distance score
        distance_score = max(0, (1 - distance_km / 500.0)) * 100

        # Trust
        trust = float(seller_company.trust_score) if seller_company.trust_score else 50.0

        # Historical exchanges
        hist_count = (
            db.query(func.count(ExchangeRequest.id))
            .filter(
                ExchangeRequest.status == ExchangeRequestStatus.ACCEPTED,
                (
                    (ExchangeRequest.supplier_plant_id == seller_plant.id)
                    & (ExchangeRequest.buyer_plant_id == buyer_plant.id)
                )
                | (
                    (ExchangeRequest.supplier_plant_id == buyer_plant.id)
                    & (ExchangeRequest.buyer_plant_id == seller_plant.id)
                ),
            )
            .scalar()
            or 0
        )
        history_score = min(hist_count * 10, 100)

        # Transport
        quantity_tons = transport_tons(float(listing.quantity), listing.unit)
        transport_cost = estimate_transport_cost(distance_km, quantity_tons)
        transport_emission = estimate_transport_emission(distance_km, quantity_tons)

        # Carbon
        carbon_factor = (
            float(listing.material.carbon_factor)
            if listing.material.carbon_factor
            else None
        )
        carbon_saving = carbon_saving_for_quantity(float(listing.quantity), listing.unit, carbon_factor)

        # Weighted score (same weights as seller flow)
        baseline_score = (
            0.25 * material_compat
            + 0.15 * quantity_compat
            + 0.10 * quality_compat
            + 0.10 * min(price_compat, 100)
            + 0.15 * distance_score
            + 0.10 * trust
            + 0.05 * history_score
            + 0.10 * min(carbon_saving / 100, 100)
        )
        baseline_score = min(max(baseline_score, 0), 100)

        # ── MC-GNN inference (falls back to baseline) ─────
        compatibility = composite_compatibility(
            material_compat, quantity_compat, quality_compat
        )
        gnn_prob = _gnn_link_score(
            db,
            seller_plant_id=seller_plant.id,
            buyer_plant_id=buyer_plant.id,
            distance_km=distance_km,
            compatibility=compatibility,
            transport_cost=transport_cost,
            carbon_saving=carbon_saving,
        )

        if gnn_prob is not None:
            score = gnn_prob * 100.0
            scoring_method = "mc_gnn"
        else:
            score = baseline_score
            scoring_method = "baseline"

        if distance_km < 100:
            transport_feas = "Excellent"
        elif distance_km < 250:
            transport_feas = "High"
        elif distance_km < 500:
            transport_feas = "Moderate"
        else:
            transport_feas = "Low"

        scored.append({
            "score": score,
            "candidate": candidate,
            "features": {
                "material_compat": material_compat,
                "quantity_compat": quantity_compat,
                "quality_compat": quality_compat,
                "compatibility": round(compatibility, 2),
                "price_compat": price_compat,
                "distance_km": round(distance_km, 2),
                "distance_score": distance_score,
                "trust": trust,
                "history_score": history_score,
                "hist_count": hist_count,
                "transport_cost": round(transport_cost, 2),
                "transport_emission": round(transport_emission, 2),
                "carbon_saving": round(carbon_saving, 2),
                "transport_feasibility": transport_feas,
                "scoring_method": scoring_method,
                "baseline_score": round(baseline_score, 2),
            },
        })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Build partner cards (reuse existing builder)
    recommendations = []
    for rank, item in enumerate(scored[: request.max_results], start=1):
        card = _build_seller_partner_card(rank, item, _model_version)
        recommendations.append(card)

    elapsed_ms = (time.time() - start_time) * 1000

    # Determine the primary material name from results
    matched_material_name = (
        scored[0]["candidate"]["listing"].material.material_name
        if scored
        else request.material_query
    )

    return SearchRecommendationResponse(
        material_query=request.material_query,
        material_name=matched_material_name,
        buyer_plant_name=buyer_plant.plant_name,
        buyer_plant_latitude=buyer_plant.latitude,
        buyer_plant_longitude=buyer_plant.longitude,
        total_candidates=len(candidates),
        recommendations=recommendations,
        model_version=_model_version,
        inference_time_ms=round(elapsed_ms, 2),
    )
