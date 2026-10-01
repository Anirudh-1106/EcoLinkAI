"""
Comprehensive unit and API test suite for EcoLinkAI.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

# Ensure backend/.env is loaded before app imports
from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

from app.main import app
from app.core.database import SessionLocal, engine
from app.models import Base
from app.models.company import Company
from app.models.plant import Plant
from app.models.material import Material
from app.models.waste_listing import WasteListing
from app.models.requirement import Requirement
from scripts.database.seed import seed_database

# Ensure database tables exist and are seeded for tests
Base.metadata.create_all(bind=engine)
db_check = SessionLocal()
try:
    if db_check.query(Company).count() == 0:
        db_check.close()
        seed_database()
    else:
        db_check.close()
except Exception:
    db_check.close()
    seed_database()

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["project"] == "EcoLinkAI"


def test_database_entities():
    db = SessionLocal()
    try:
        company_count = db.query(Company).count()
        plant_count = db.query(Plant).count()
        material_count = db.query(Material).count()
        waste_count = db.query(WasteListing).count()

        assert company_count > 0, "Companies table should be seeded"
        assert plant_count > 0, "Plants table should be seeded"
        assert material_count > 0, "Materials table should be seeded"
        assert waste_count > 0, "Waste listings table should be seeded"
    finally:
        db.close()


def test_list_companies_requires_admin():
    """The company directory exposes GST/registration numbers, so it is admin-only."""
    response = client.get("/api/v1/companies")
    assert response.status_code == 401


def test_list_materials_api():
    response = client.get("/api/v1/materials")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0


def test_recommendation_api_baseline():
    db = SessionLocal()
    try:
        listing = db.query(WasteListing).first()
        assert listing is not None, "Need at least one waste listing for testing"

        response = client.post(
            "/api/v1/recommendations",
            json={"waste_listing_id": str(listing.id), "max_results": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0
        first_rec = data["recommendations"][0]
        assert "ai_score" in first_rec
        assert "explanation" in first_rec
    finally:
        db.close()


def test_recommendations_are_ranked_and_scored_by_a_known_model():
    """Every card must declare which model scored it, and stay ranked by score."""
    db = SessionLocal()
    try:
        listing = db.query(WasteListing).first()
        response = client.post(
            "/api/v1/recommendations",
            json={"waste_listing_id": str(listing.id), "max_results": 5},
        )
        assert response.status_code == 200
        recs = response.json()["recommendations"]

        for rec in recs:
            assert rec["model_type"] in ("mc_gnn", "baseline")
            assert 0.0 <= rec["ai_score"] <= 100.0

        scores = [r["ai_score"] for r in recs]
        assert scores == sorted(scores, reverse=True), "Cards must be ranked by score"
    finally:
        db.close()


def test_ai_metrics_are_measured_not_hardcoded():
    """
    Metrics must come from a real evaluation against a real baseline.

    This deliberately does not assert that the MC-GNN outranks the baseline.
    Which model wins is a measurement, not an invariant, and a suite that
    fails unless the model wins pressures every future change toward making
    that number come out right rather than toward reporting it honestly. On a
    temporal split the baseline is currently marginally ahead (NDCG@5 0.687 vs
    0.678), and that has to be allowed to show rather than be treated as a
    broken build.

    What is asserted is that both numbers are genuinely produced: present,
    in range, and distinct from each other, which is what catches the
    hardcoded or copied-over metrics this test exists to prevent.
    """
    response = client.get("/api/v1/analytics/ai-metrics")
    assert response.status_code == 200
    data = response.json()

    assert data["training_samples"] > 0
    assert 0.0 <= data["ndcg_at_5"] <= 1.0
    assert 0.0 <= data["baseline_ndcg_at_5"] <= 1.0
    assert data["ndcg_at_5"] != data["baseline_ndcg_at_5"], (
        "Model and baseline NDCG are identical -- suggests one was copied "
        "from the other rather than separately evaluated"
    )


def _two_rival_companies():
    """A plant from one company plus a plant from a different company, with a token for the first."""
    from app.core.security import create_access_token
    from app.models.user import User

    db = SessionLocal()
    try:
        attacker_plant = db.query(Plant).first()
        victim_plant = (
            db.query(Plant).filter(Plant.company_id != attacker_plant.company_id).first()
        )
        user = db.query(User).filter(User.company_id == attacker_plant.company_id).first()
        token = create_access_token({"sub": str(user.id), "role": user.role.value})
        return (
            attacker_plant.id,
            attacker_plant.company_id,
            victim_plant.id,
            {"Authorization": f"Bearer {token}"},
        )
    finally:
        db.close()


def test_cannot_edit_another_companys_waste_listing():
    attacker_plant_id, _, victim_plant_id, headers = _two_rival_companies()
    db = SessionLocal()
    try:
        victim_listing = (
            db.query(WasteListing).filter(WasteListing.plant_id == victim_plant_id).first()
        )
        if victim_listing is None:
            pytest.skip("Victim company has no waste listing to target")
        listing_id = victim_listing.id
    finally:
        db.close()

    response = client.put(f"/api/v1/waste-listings/{listing_id}", headers=headers,
                          json={"price_per_unit": 1})
    assert response.status_code == 403


def test_cannot_create_waste_listing_on_another_companys_plant():
    _, _, victim_plant_id, headers = _two_rival_companies()
    db = SessionLocal()
    try:
        material_id = db.query(Material).first().id
    finally:
        db.close()

    response = client.post("/api/v1/waste-listings", headers=headers, json={
        "plant_id": str(victim_plant_id),
        "material_id": str(material_id),
        "quantity": 10,
        "unit": "kg",
        "available_from": "2026-01-01",
        "available_until": "2027-01-01",
    })
    assert response.status_code == 403


def test_cannot_edit_another_companys_requirement():
    _, _, victim_plant_id, headers = _two_rival_companies()
    db = SessionLocal()
    try:
        victim_req = (
            db.query(Requirement).filter(Requirement.plant_id == victim_plant_id).first()
        )
        if victim_req is None:
            pytest.skip("Victim company has no requirement to target")
        req_id = victim_req.id
    finally:
        db.close()

    response = client.put(f"/api/v1/requirements/{req_id}", headers=headers, json={"quantity": 1})
    assert response.status_code == 403


def test_cannot_raise_exchange_request_in_another_companys_name():
    """Buying on behalf of a plant you don't own would let you impersonate that company."""
    _, _, victim_plant_id, headers = _two_rival_companies()
    db = SessionLocal()
    try:
        listing = db.query(WasteListing).first()
        listing_id = listing.id
    finally:
        db.close()

    response = client.post("/api/v1/exchange-requests", headers=headers, json={
        "waste_listing_id": str(listing_id),
        "buyer_plant_id": str(victim_plant_id),
        "requested_quantity": 10,
    })
    assert response.status_code == 403


def test_cannot_review_an_exchange_you_were_not_party_to():
    """Reviews move trust scores, which feed recommendations - only participants may post one."""
    from app.models.exchange import Exchange
    from app.models.exchange_request import ExchangeRequest

    _, attacker_company_id, _, headers = _two_rival_companies()
    db = SessionLocal()
    try:
        own_plant_ids = [
            p.id for p in db.query(Plant).filter(Plant.company_id == attacker_company_id)
        ]
        outsider_exchange = (
            db.query(Exchange)
            .join(ExchangeRequest, Exchange.exchange_request_id == ExchangeRequest.id)
            .filter(
                ~ExchangeRequest.supplier_plant_id.in_(own_plant_ids),
                ~ExchangeRequest.buyer_plant_id.in_(own_plant_ids),
            )
            .first()
        )
        if outsider_exchange is None:
            pytest.skip("No third-party exchange available to target")
        exchange_id = outsider_exchange.id
    finally:
        db.close()

    response = client.post("/api/v1/reviews", headers=headers, json={
        "exchange_id": str(exchange_id),
        "rating": 1,
        "feedback": "x",
    })
    assert response.status_code == 403


def test_can_still_edit_own_waste_listing():
    """The ownership checks must not lock companies out of their own data."""
    attacker_plant_id, _, _, headers = _two_rival_companies()
    db = SessionLocal()
    try:
        own_listing = (
            db.query(WasteListing).filter(WasteListing.plant_id == attacker_plant_id).first()
        )
        if own_listing is None:
            pytest.skip("Company has no waste listing of its own")
        listing_id = own_listing.id
        price = float(own_listing.price_per_unit) if own_listing.price_per_unit else 50.0
    finally:
        db.close()

    response = client.put(f"/api/v1/waste-listings/{listing_id}", headers=headers,
                          json={"price_per_unit": price})
    assert response.status_code == 200


def test_cannot_review_an_exchange_that_is_not_completed():
    """A review rates an outcome, and it moves trust scores, so it needs one to exist."""
    from app.core.security import create_access_token
    from app.models.exchange import Exchange
    from app.models.exchange_request import ExchangeRequest
    from app.models.user import User

    from app.enums.exchange import ExchangeStatus

    db = SessionLocal()
    original_status = None
    exchange_id = None
    try:
        # Seed data completes every exchange, so put one back in transit for
        # the duration of the test rather than skipping and leaving the guard
        # unexercised. Restored in the finally block below.
        exchange = db.query(Exchange).first()
        if exchange is None:
            pytest.skip("No exchange available to target")

        req = db.query(ExchangeRequest).filter(
            ExchangeRequest.id == exchange.exchange_request_id
        ).first()
        buyer_plant = db.query(Plant).filter(Plant.id == req.buyer_plant_id).first()
        # A participant, so the request is rejected for its status rather than
        # for the caller lacking access.
        user = db.query(User).filter(User.company_id == buyer_plant.company_id).first()
        if user is None:
            pytest.skip("Participating company has no user account")

        token = create_access_token({"sub": str(user.id), "role": user.role.value})
        exchange_id = exchange.id
        original_status = exchange.exchange_status
        exchange.exchange_status = ExchangeStatus.IN_TRANSIT
        db.commit()

        response = client.post(
            "/api/v1/reviews",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "exchange_id": str(exchange_id),
                "rating": 1,
                "feedback": "x",
            },
        )
        assert response.status_code == 400
        assert "completed" in response.json()["detail"].lower()
    finally:
        if exchange_id is not None and original_status is not None:
            restore = db.query(Exchange).filter(Exchange.id == exchange_id).first()
            if restore is not None:
                restore.exchange_status = original_status
                db.commit()
        db.close()


def test_category_matches_are_offered_but_rank_below_exact_matches():
    """A related material is better than no seller at all, but must not outrank the real thing."""
    from app.enums.exchange import RequirementStatus
    from app.schemas.recommendation import RequirementRecommendationRequest
    from app.services import recommendation_service as rs

    db = SessionLocal()
    try:
        requirement = (
            db.query(Requirement)
            .join(Material, Requirement.material_id == Material.id)
            .filter(Requirement.status == RequirementStatus.OPEN)
            .first()
        )
        if requirement is None:
            pytest.skip("No open requirement to recommend against")

        response = rs.get_recommendations_by_requirement(
            db,
            RequirementRecommendationRequest(requirement_id=requirement.id, max_results=10),
        )
    finally:
        db.close()

    labels = [c.explanation.material_compatibility for c in response.recommendations]
    scores = [c.ai_score for c in response.recommendations]

    assert scores == sorted(scores, reverse=True), "Cards must stay ranked by score"

    # Wherever both kinds are present, no category match may sit above an exact one.
    if "Exact Match" in labels and "Same Category" in labels:
        assert labels.index("Same Category") > labels.index("Exact Match")


def test_gnn_falls_back_gracefully_for_unknown_plants():
    """A plant missing from the cached graph must degrade to baseline, not error."""
    import uuid as _uuid

    from app.models.plant import Plant
    from app.services import recommendation_service as rs

    db = SessionLocal()
    try:
        plant = db.query(Plant).first()
        score = rs._gnn_link_score(
            db,
            seller_plant_id=_uuid.uuid4(),  # never registered
            buyer_plant_id=plant.id,
            distance_km=50.0,
            material_compat=100.0,
            quantity_compat=100.0,
            quality_compat=100.0,
            transport_cost=1000.0,
            carbon_saving=500.0,
        )
        assert score is None
    finally:
        db.close()


# =====================================================
# Scoring inputs
#
# Everything below guards a bug that shipped and stayed hidden because the
# suite only covered authorization. Each one was found by hand, and without
# these a single careless edit puts it straight back.
# =====================================================


def test_kg_and_ton_convert_to_a_common_basis():
    """
    A ton listing against a kg requirement must not be compared as raw numbers.

    Treating 2458.16 ton as 2458.16 kg made a listing a thousand times too
    small, which fed transport cost, carbon saving and quantity fit alike.
    """
    from app.enums.common import QuantityUnit
    from app.utils.quantity import quantity_fit_pct, to_kg

    assert to_kg(1.0, QuantityUnit.TON) == 1000.0
    assert to_kg(1.0, QuantityUnit.KG) == 1.0

    # One tonne and a thousand kilograms are the same amount of material.
    assert quantity_fit_pct(1.0, QuantityUnit.TON, 1000.0, QuantityUnit.KG) == 100.0
    # And a tonne against a single kilogram is a terrible fit, not a perfect one.
    assert quantity_fit_pct(1.0, QuantityUnit.TON, 1.0, QuantityUnit.KG) < 1.0


def test_volume_and_count_units_are_not_guessed_as_mass():
    """
    Litres and pieces have no mass without a density this system never stores.

    Returning the raw number would silently claim a comparison nobody can
    make, so these report the "unknown" default instead.
    """
    from app.enums.common import QuantityUnit
    from app.utils.quantity import to_kg, quantity_fit_pct, carbon_saving_for_quantity

    for unit in (QuantityUnit.LITER, QuantityUnit.CUBIC_METER, QuantityUnit.PIECE, QuantityUnit.METER):
        assert to_kg(100.0, unit) is None
        assert quantity_fit_pct(100.0, unit, 100.0, QuantityUnit.KG, default=50.0) == 50.0
        # No mass means no defensible carbon claim.
        assert carbon_saving_for_quantity(100.0, unit, 1.2) == 0.0


def test_freight_cost_tapers_but_never_decreases():
    """
    Volume discount must stay continuous and monotonic.

    A flat per-ton rate priced a 2,458 t movement as 1,400 separate small
    deliveries. Tiering fixes that, but a bracket edge that jumps, or a rate
    that lets more tonnage cost less, would be worse than the flat rate.
    """
    from app.utils.distance import FREIGHT_BRACKETS, billable_ton_rate_units, estimate_transport_cost

    previous = -1.0
    for tenths in range(0, 30001):
        units = billable_ton_rate_units(tenths / 10.0)
        assert units >= previous - 1e-9, "more tonnage must never cost less"
        previous = units

    for upper, _ in FREIGHT_BRACKETS[:-1]:
        below = billable_ton_rate_units(upper - 1e-6)
        above = billable_ton_rate_units(upper + 1e-6)
        assert abs(above - below) < 1e-3, f"discontinuity at the {upper} t bracket edge"

    # Small consignments are unchanged, so the common case never regressed.
    for tons in (0.1, 1.0, 1.756, 5.0):
        assert abs(estimate_transport_cost(61.3, tons) - (61.3 * tons * 8.0)) < 1e-6

    # Large ones genuinely taper.
    assert estimate_transport_cost(50, 2458.16) < 50 * 2458.16 * 8.0 * 0.5
    assert billable_ton_rate_units(0.0) == 0.0
    assert billable_ton_rate_units(-5.0) == 0.0


def test_edge_features_stay_inside_zero_and_one():
    """
    No feature may dominate the model by raw magnitude.

    Transport cost and carbon saving were divided by flat constants and left
    unbounded, reaching 8,800 while distance and compatibility sat in 0-1.
    Shipment size then outweighed whether the quantity matched at all.
    """
    import itertools
    from app.utils.edge_encoding import EDGE_FEATURE_NAMES, encode_edge_features

    for dist, compat, cost, carbon, prior, seen in itertools.product(
        (0, 50, 500, 50_000), (0, 50, 100, 500), (0, 1e3, 5e6, 1e12),
        (0, 1e3, 1e7, 1e12), (0, 3, 9999), (True, False),
    ):
        vector = encode_edge_features(
            distance_km=dist, material_compatibility=compat,
            quantity_compatibility=compat, quality_compatibility=compat,
            transport_cost=cost, carbon_saving=carbon,
            prior_successes=prior, has_prior_interaction=seen,
        )
        assert len(vector) == len(EDGE_FEATURE_NAMES)
        assert all(0.0 <= v <= 1.0 for v in vector), (dist, compat, cost, carbon, vector)

    # Ordering must survive the compression, or the signal is destroyed.
    costs = [encode_edge_features(
        distance_km=50, material_compatibility=100, quantity_compatibility=50,
        quality_compatibility=100, transport_cost=c, carbon_saving=1000,
    )[4] for c in (0, 10, 1e3, 1e5, 5e6)]
    assert costs == sorted(costs)


def test_pair_history_never_counts_the_request_it_describes():
    """
    A pair's trading history must be built only from earlier deals.

    Counting the current request, or any later one, folds the answer into the
    input and inflates every metric downstream.
    """
    from app.core.database import SessionLocal
    from ai.graph.builder import build_industrial_graph
    from ai.evaluation.metrics import roc_auc
    import numpy as np

    db = SessionLocal()
    try:
        data, _, _ = build_industrial_graph(db)
        features = data.edge_attr.numpy()
        labels = data.y.numpy()

        prior_successes = features[:, 6]
        has_prior = features[:, 7]

        first_contact = has_prior == 0.0
        assert first_contact.sum() > 0, "expected some pairs meeting for the first time"
        assert (prior_successes[first_contact] == 0.0).all(), (
            "a pair with no prior interaction cannot have prior successes"
        )

        # A leak shows up as history alone predicting the outcome almost
        # perfectly. A weak, honest signal sits near chance.
        assert roc_auc(labels, prior_successes) < 0.75, "history alone is too predictive - likely leaking"
        assert roc_auc(labels, has_prior) < 0.75
    finally:
        db.close()


def test_calibration_changes_the_number_but_not_the_order():
    """
    Calibration must correct the displayed probability without re-ranking.

    If it moved candidates past one another it would be a silent change to
    the recommendations themselves, not a presentation fix.
    """
    import numpy as np
    from app.utils.calibration import apply_platt, expected_calibration_error, fit_platt

    rng = np.random.default_rng(0)
    truth = rng.random(400)
    # Scores that rank correctly but read systematically low.
    raw = truth * 0.6
    labels = (rng.random(400) < truth).astype(float)

    fitted = fit_platt(raw, labels)
    assert fitted is not None
    adjusted = np.array([apply_platt(float(p), *fitted) for p in raw])

    assert (np.argsort(raw) == np.argsort(adjusted)).all(), "calibration must preserve ranking"
    assert expected_calibration_error(adjusted, labels) < expected_calibration_error(raw, labels)

    # Too little data to fit is reported honestly rather than guessed at.
    assert fit_platt(np.array([0.5, 0.6]), np.array([1.0, 0.0])) is None
    assert fit_platt(rng.random(100), np.ones(100)) is None


# =====================================================
# Exchange lifecycle
# =====================================================


def _fresh_listing_and_buyer(db, *, priced=True, min_quantity=100):
    """An available listing plus a plant belonging to a different company."""
    from app.enums.exchange import WasteStatus

    query = db.query(WasteListing).filter(
        WasteListing.status == WasteStatus.AVAILABLE,
        WasteListing.quantity > min_quantity,
    )
    if priced:
        query = query.filter(WasteListing.price_per_unit.isnot(None))
    listing = query.first()
    if listing is None:
        pytest.skip("No suitable available listing in the seeded data")
    buyer = db.query(Plant).filter(Plant.company_id != listing.plant.company_id).first()
    return listing, buyer


def _raise_request(db, listing, buyer, quantity):
    from app.schemas.exchange import ExchangeRequestCreate
    from app.services import exchange_service, recommendation_service

    match = recommendation_service.score_candidate_pair(
        db,
        waste_listing_id=listing.id,
        buyer_plant_id=buyer.id,
        traded_quantity=float(quantity),
    )
    return exchange_service.create_exchange_request(
        db,
        ExchangeRequestCreate(
            waste_listing_id=listing.id,
            buyer_plant_id=buyer.id,
            requested_quantity=quantity,
        ),
        supplier_plant_id=listing.plant_id,
        compatibility_score=90,
        ai_confidence_score=50,
        recommendation_rank=1,
        distance_km=match["distance_km"],
        estimated_transport_cost=match["estimated_transport_cost"],
        estimated_carbon_emission=match["estimated_carbon_emission"],
        estimated_carbon_saving=match["estimated_carbon_saving"],
    )


def test_partial_purchase_leaves_the_remainder_on_the_market():
    """
    Buying part of a listing must not take the whole thing off the board.

    Accepting used to reserve the entire listing however little was asked
    for, so a buyer wanting a quarter removed the rest from every other
    buyer's reach.
    """
    from decimal import Decimal
    from app.enums.exchange import WasteStatus
    from app.services import exchange_service

    db = SessionLocal()
    try:
        listing, buyer = _fresh_listing_and_buyer(db)
        starting = listing.quantity
        # Quantised to the two decimals the column stores, so the assertion
        # tests the arithmetic rather than the rounding.
        quarter = (starting / 4).quantize(Decimal("0.01"))

        request = _raise_request(db, listing, buyer, quarter)
        exchange_service.accept_exchange_request(db, request.id)
        db.refresh(listing)

        assert listing.quantity == starting - quarter, "stock drawn down by the agreed amount"
        assert listing.status == WasteStatus.AVAILABLE, "the remainder stays purchasable"
    finally:
        db.rollback()
        db.close()


def test_a_listing_cannot_be_oversold():
    """Accepting more than remains must fail rather than drive stock negative."""
    from app.services import exchange_service

    db = SessionLocal()
    try:
        listing, buyer = _fresh_listing_and_buyer(db)
        request = _raise_request(db, listing, buyer, listing.quantity * 10)

        with pytest.raises(ValueError, match="only"):
            exchange_service.accept_exchange_request(db, request.id)
    finally:
        db.rollback()
        db.close()


def test_agreed_price_is_the_goods_not_the_lorry():
    """
    An exchange records what the material cost, not what the haulage cost.

    agreed_price was set to estimated_transport_cost, so every completed
    exchange -- and every analytic reading them -- carried a figure that was
    wrong by orders of magnitude.
    """
    from decimal import Decimal
    from app.models.exchange import Exchange
    from app.services import exchange_service

    db = SessionLocal()
    try:
        listing, buyer = _fresh_listing_and_buyer(db)
        quantity = (listing.quantity / 4).quantize(Decimal("0.01"))
        expected = quantity * listing.price_per_unit

        request = _raise_request(db, listing, buyer, quantity)
        exchange_service.accept_exchange_request(db, request.id)
        exchange = (
            db.query(Exchange).filter(Exchange.exchange_request_id == request.id).first()
        )

        # Within a rounding step of the stored precision.
        assert abs(exchange.agreed_price - expected) <= Decimal("0.01")
        assert exchange.agreed_price != exchange.transport_cost, (
            "agreed_price must not be a copy of the transport cost"
        )
    finally:
        db.rollback()
        db.close()


# =====================================================
# Reviews
# =====================================================


def _completed_exchange_without_review(db):
    from app.models.exchange import Exchange
    from app.models.review import Review
    from app.enums.exchange import ExchangeStatus

    exchange = (
        db.query(Exchange)
        .filter(Exchange.exchange_status == ExchangeStatus.COMPLETED)
        .first()
    )
    if exchange is None:
        pytest.skip("No completed exchange in the seeded data")
    db.query(Review).filter(Review.exchange_id == exchange.id).delete()
    db.commit()
    return exchange


def test_a_buyer_rates_the_supplier_and_never_itself():
    """
    Each party rates only the other.

    Both ratings were mandatory and came from whoever reviewed first, so a
    buyer set its own score and moved its own trust -- which then feeds the
    recommendations deciding who it gets shown to.
    """
    from app.schemas.review import ReviewCreate
    from app.services import review_service

    db = SessionLocal()
    try:
        exchange = _completed_exchange_without_review(db)
        request = exchange.exchange_request
        buyer_company_id = request.buyer_plant.company_id
        supplier_company_id = request.supplier_plant.company_id

        assert review_service.role_in_exchange(db, exchange.id, buyer_company_id) == "buyer"
        assert review_service.role_in_exchange(db, exchange.id, supplier_company_id) == "supplier"

        review = review_service.submit_review(
            db,
            ReviewCreate(exchange_id=exchange.id, rating=5, feedback="good material"),
            reviewer_company_id=buyer_company_id,
            role="buyer",
        )
        assert review.supplier_rating == 5, "the buyer rates the supplier"
        assert review.buyer_rating is None, "the buyer must not have rated itself"

        review = review_service.submit_review(
            db,
            ReviewCreate(exchange_id=exchange.id, rating=4, feedback="paid promptly"),
            reviewer_company_id=supplier_company_id,
            role="supplier",
        )
        assert review.supplier_rating == 5, "the buyer's rating must survive"
        assert review.buyer_rating == 4, "the supplier fills the other half"
    finally:
        db.rollback()
        db.close()


def test_a_company_outside_the_exchange_has_no_role_in_it():
    """Only the two parties may review, and role lookup is how that is decided."""
    from app.models.company import Company
    from app.services import review_service

    db = SessionLocal()
    try:
        exchange = _completed_exchange_without_review(db)
        request = exchange.exchange_request
        involved = {request.buyer_plant.company_id, request.supplier_plant.company_id}
        outsider = db.query(Company).filter(Company.id.notin_(involved)).first()

        assert review_service.role_in_exchange(db, exchange.id, outsider.id) is None
    finally:
        db.rollback()
        db.close()


def test_trust_counts_ratings_earned_on_both_sides():
    """
    A company that mostly buys must still be able to earn trust.

    Only supplier-side ratings counted, so a buyer's conduct never moved its
    score however well it behaved -- and trust feeds the recommender, leaving
    it unrecommendable for a reason unconnected to its behaviour.
    """
    from decimal import Decimal
    from app.models.exchange import Exchange
    from app.models.exchange_request import ExchangeRequest
    from app.models.review import Review
    from app.schemas.review import ReviewCreate
    from app.services import company_service, review_service

    db = SessionLocal()
    try:
        exchange = _completed_exchange_without_review(db)
        request = exchange.exchange_request
        buyer_company = request.buyer_plant.company

        # Strip every rating this company already holds, so the score that
        # follows can only have come from the one rating added below. Without
        # this the company keeps its supplier-side history and stays above
        # zero whether or not buyer-side ratings are counted -- which is
        # exactly the bug, passing unnoticed.
        plant_ids = [p.id for p in buyer_company.plants]
        stale = (
            db.query(Review.id)
            .join(Exchange, Review.exchange_id == Exchange.id)
            .join(ExchangeRequest, Exchange.exchange_request_id == ExchangeRequest.id)
            .filter(
                (ExchangeRequest.supplier_plant_id.in_(plant_ids))
                | (ExchangeRequest.buyer_plant_id.in_(plant_ids))
            )
            .all()
        )
        db.query(Review).filter(Review.id.in_([r[0] for r in stale])).delete(
            synchronize_session=False
        )

        # Parked at a value the correct answer cannot coincide with. Seeded
        # companies often already sit at 100, so asserting the expected score
        # without this passes whether or not anything was recalculated.
        buyer_company.trust_score = Decimal("1")
        db.commit()

        # A rating given *by* the supplier is one earned by the buyer, and is
        # now this company's only rating anywhere.
        review_service.submit_review(
            db,
            ReviewCreate(exchange_id=exchange.id, rating=5, feedback="paid promptly"),
            reviewer_company_id=request.supplier_plant.company_id,
            role="supplier",
        )
        company_service.update_trust_score(db, buyer_company.id)
        db.refresh(buyer_company)

        # 5 stars, scaled to the 0-100 trust range, and nothing else in play.
        assert float(buyer_company.trust_score) == 100.0, (
            "a buyer-side rating must be the thing that set this score"
        )
    finally:
        db.rollback()
        db.close()


# =====================================================
# Real-data study (EPA TRI)
# =====================================================


def _tri_toy_dataset():
    """Two senders, four receivers, one year of history and one target year."""
    from ai.experiments.tri_loader import Site, TRIDataset, Transfer

    sites = {
        k: Site(k, lat=40.0 + i * 0.1, lon=-80.0 - i * 0.1, naics=n)
        for i, (k, n) in enumerate([
            ("A", "331110"), ("B", "325412"), ("R1", "562211"),
            ("R2", "331492"), ("R3", ""), ("R4", "331492"),
        ])
    }
    transfers = [
        # history
        Transfer(2023, "A", "R1", "X", "M24", 1000.0),
        Transfer(2023, "B", "R2", "X", "M24", 800.0),
        Transfer(2023, "B", "R3", "Y", "M56", 500.0),
        Transfer(2023, "B", "R4", "X", "M24", 900.0),
        # the year being scored
        Transfer(2024, "A", "R1", "X", "M24", 1200.0),
        Transfer(2024, "A", "R2", "X", "M24", 1100.0),
    ]
    return TRIDataset(transfers, sites)


def test_tri_features_are_built_from_history_only():
    """
    Nothing from the year being scored may reach an input.

    The 2024 A->R1 transfer is the answer; if it were counted, A would have
    two past dealings with R1 instead of one and the model would be reading
    the label off its own input.
    """
    from app.utils.edge_encoding import EDGE_FEATURE_NAMES

    g = _tri_toy_dataset().build_graph(2024)
    idx = {k: i for i, k in enumerate(g["node_keys"])}
    prior = EDGE_FEATURE_NAMES.index("prior_successes")
    by_pair = {
        (g["node_keys"][s], g["node_keys"][r]): g["edge_attr"][e]
        for e, (s, r) in enumerate(g["edge_index"].T)
    }

    assert by_pair[("A", "R1")][prior] > 0, "one real past dealing"
    assert by_pair[("A", "R2")][prior] == 0, "A never dealt with R2 before 2024"

    history_links = {
        (g["node_keys"][s], g["node_keys"][r]) for s, r in g["message_edge_index"].T
    }
    assert history_links == {("A", "R1"), ("B", "R2"), ("B", "R3"), ("B", "R4")}

    exchanges_col = 5  # historical exchange count, as in node_features
    assert g["x"][idx["A"], exchanges_col] == pytest.approx(1 / 50.0), "one past transfer, not two"


def test_tri_negatives_never_include_a_real_partner():
    """A receiver the sender actually used that year is never sampled as 'did not happen'."""
    g = _tri_toy_dataset().build_graph(2024)
    keys = g["node_keys"]
    labelled = {
        (keys[s], keys[r]): y for (s, r), y in zip(g["edge_index"].T, g["y"])
    }
    assert labelled[("A", "R1")] == 1.0
    assert labelled[("A", "R2")] == 1.0
    assert labelled[("A", "R4")] == 0.0, "R4 took chemical X before and was not used"
    assert ("A", "R3") not in labelled, "R3 never handled X or its recovery method"


def test_tri_inputs_match_the_model_contract():
    """17 plant inputs and 8 pair inputs, every one inside 0-1."""
    from ai.experiments.tri_loader import EDGE_FEATURE_DIM, NODE_FEATURE_DIM

    g = _tri_toy_dataset().build_graph(2024)
    assert g["x"].shape[1] == NODE_FEATURE_DIM == 17
    assert g["edge_attr"].shape[1] == EDGE_FEATURE_DIM == 8
    assert (g["edge_attr"] >= 0).all() and (g["edge_attr"] <= 1).all()
    assert (g["x"][:, 2:] >= 0).all() and (g["x"][:, 2:] <= 1).all()


def test_tri_without_history_nothing_is_scored():
    """With no past to learn candidates from, the loader asks no questions rather than guessing."""
    g = _tri_toy_dataset().build_graph(2024, history_start=2024)
    assert g["stats"].get("queries_used", 0) == 0
    assert g["stats"]["history_transfers"] == 0


def test_tri_units_and_sector_mapping():
    from ai.experiments.tri_loader import naics_to_sector, to_kg

    assert to_kg(100, "Pounds") == pytest.approx(45.359237)
    assert to_kg(1000, "Grams") == pytest.approx(1.0)
    assert naics_to_sector("331110") == "Metal & Metallurgy"
    assert naics_to_sector("325412") == "Pharmaceuticals", "longest prefix wins over 325"
    assert naics_to_sector("325110") == "Chemicals & Petrochemicals"
    assert naics_to_sector("562211") == "General Manufacturing", "the platform's own fallback"
    assert naics_to_sector("") == "General Manufacturing"


def _tiny_graph():
    import torch

    torch.manual_seed(0)
    x = torch.rand(6, 17)
    x[:, 0] = torch.tensor([0.10, 0.11, 0.12, 0.13, 0.14, 0.15])  # lat / 90
    x[:, 1] = torch.tensor([0.42, 0.43, 0.41, 0.44, 0.42, 0.43])  # lon / 180
    edge_index = torch.tensor([[0, 1, 2, 3], [4, 5, 4, 5]])
    edge_attr = torch.rand(4, 8)
    message = torch.tensor([[0, 1], [4, 5]])
    return x, edge_index, edge_attr, message


def test_checkpoints_without_a_location_frame_read_coordinates_as_before():
    """Models saved before the setting existed must score exactly as they did."""
    import torch
    from ai.models.mc_gnn import MCGNN

    x, ei, ea, msg = _tiny_graph()
    model = MCGNN(in_features=17, edge_dim=8).eval()
    with torch.no_grad():
        current, _ = model(x, ei, ea, message_edge_index=msg)
        del model.location_frame  # as unpickled from an older checkpoint
        legacy, _ = model(x, ei, ea, message_edge_index=msg)
    assert torch.equal(current, legacy)


def test_market_frame_scores_do_not_depend_on_where_the_market_is():
    """Moving every plant by the same offset -- Kerala to the US -- changes nothing."""
    import torch
    from ai.models.mc_gnn import MCGNN

    x, ei, ea, msg = _tiny_graph()
    moved = x.clone()
    moved[:, 0] += 0.3
    moved[:, 1] -= 0.95

    market = MCGNN(in_features=17, edge_dim=8, location_frame="market").eval()
    absolute = MCGNN(in_features=17, edge_dim=8).eval()
    absolute.load_state_dict(market.state_dict())
    with torch.no_grad():
        here, _ = market(x, ei, ea, message_edge_index=msg)
        there, _ = market(moved, ei, ea, message_edge_index=msg)
        abs_here, _ = absolute(x, ei, ea, message_edge_index=msg)
        abs_there, _ = absolute(moved, ei, ea, message_edge_index=msg)
    assert torch.allclose(here, there, atol=1e-5)
    assert not torch.allclose(abs_here, abs_there, atol=1e-5), "absolute frame should notice the move"


def _tri_toy_with_newcomer():
    """The toy market plus C, a facility with no history that lists chemical X in 2024."""
    from ai.experiments.tri_loader import Site, TRIDataset, Transfer

    base = _tri_toy_dataset()
    sites = dict(base.sites)
    sites["C"] = Site("C", lat=40.3, lon=-80.3, naics="331110")
    transfers = base.transfers + [Transfer(2024, "C", "R4", "X", "M24", 700.0)]
    return TRIDataset(transfers, sites)


def test_tri_views_let_a_newcomer_join_from_its_listing():
    """No history, so no relationships -- but its listed waste links it to those who process it."""
    g = _tri_toy_with_newcomer().build_graph(2024, views=True)
    keys = g["node_keys"]
    c = keys.index("C")

    def neighbours(view):
        ei = g["views"][view]
        return {keys[b] for a, b in ei.T if a == c}

    assert neighbours("relationship") == set()
    assert {"R1", "R2", "R4"} <= neighbours("capability"), "C's chemical X is what R1, R2, R4 take in"
    assert neighbours("peer"), "same industry and waste stream as A and B"
    rows_from_c = g["edge_index"][0] == c
    assert g["sender_is_new"][rows_from_c].all()
    assert not g["sender_is_new"][~rows_from_c].any()


def test_tri_views_never_see_where_the_scored_year_went():
    """2024's A->R2 and C->R4 are answers; no network or pair column may know them."""
    g = _tri_toy_with_newcomer().build_graph(2024, views=True)
    keys = g["node_keys"]
    rel = {(keys[a], keys[b]) for a, b in g["views"]["relationship"].T}
    assert ("A", "R2") not in rel and ("C", "R4") not in rel

    votes = {
        (keys[s], keys[r]): v
        for (s, r), v in zip(g["edge_index"].T, g["pair_graph_features"][:, 1])
    }
    # Peer votes count history only. A and B are both peers of C; B sent to
    # R2 in 2023, A only in 2024 -- an answer -- so R2 has one vote, not two.
    from ai.experiments.tri_views import NEIGHBOURS_PER_FACILITY

    assert votes[("C", "R2")] == pytest.approx(1 / NEIGHBOURS_PER_FACILITY)
    assert votes[("C", "R4")] > 0, "B, a peer, sent to R4 in 2023"


def test_multiview_mcgnn_reads_three_networks_and_weighs_them():
    import torch
    from ai.models.mc_gnn import VIEW_NAMES, MultiViewMCGNN

    x, ei, ea, msg = _tiny_graph()
    views = [msg, torch.tensor([[0, 2], [5, 4]]), torch.tensor([[1, 3], [0, 2]])]
    model = MultiViewMCGNN(in_features=17, edge_dim=8).eval()
    with torch.no_grad():
        scores, channels = model(x, ei, ea, message_edge_index=views)
        weights = model.channel_weights(x, views)
    assert scores.shape == (4,) and len(channels) == 3
    assert weights.shape == (6, len(VIEW_NAMES))
    assert torch.allclose(weights.sum(dim=1), torch.ones(6))
    # Swapping in a different capability network must change the result:
    # each channel reads its own graph.
    other = [views[0], torch.tensor([[1, 3], [4, 5]]), views[2]]
    with torch.no_grad():
        changed, _ = model(x, ei, ea, message_edge_index=other)
    assert not torch.allclose(scores, changed)


def test_normalized_hsic_cannot_be_lowered_by_shrinking_embeddings():
    """The raw term falls with scale alone; the normalised one must not."""
    import torch
    from ai.models.hsic import hsic_loss, normalized_hsic

    torch.manual_seed(0)
    a, b = torch.randn(200, 16), torch.randn(200, 16)
    gram = hsic_loss(a, b) / torch.sqrt(hsic_loss(a, a) * hsic_loss(b, b))
    assert torch.allclose(normalized_hsic(a, b), gram, atol=1e-5), "same quantity as the Gram definition"
    assert torch.allclose(normalized_hsic(a, b), normalized_hsic(0.01 * a, 0.01 * b), atol=1e-6)
    assert hsic_loss(0.01 * a, 0.01 * b) < hsic_loss(a, b) * 1e-6
    assert normalized_hsic(a, a).item() == pytest.approx(1.0, abs=1e-6)


def test_tri_pair_sets_use_history_only():
    """A pair set may only hold facilities linked by earlier trade, never by the year scored."""
    from ai.experiments.tri_pairs import pair_neighbourhoods

    g = _tri_toy_with_newcomer().build_graph(2024, views=True)
    sets = pair_neighbourhoods(g)
    keys = g["node_keys"]
    rel = {(keys[a], keys[b]) for a, b in g["views"]["relationship"].T}
    for row, (s, b) in enumerate(g["edge_index"].T):
        for name in ("co_partners", "peers"):
            for member in sets[name][row]:
                if member >= 0:
                    assert (keys[member], keys[b]) in rel, f"{name} member must have traded with the recycler before"


def test_pair_aware_scorer_reads_shared_neighbours():
    """Changing a shared neighbour changes only the pairs that share it."""
    import torch
    from ai.models.mc_gnn import PairAwareMCGNN

    torch.manual_seed(0)
    x = torch.rand(8, 17)
    edges = torch.tensor([[0, 1, 2], [5, 6, 7]])
    attr = torch.rand(3, 10)
    views = [torch.tensor([[0, 5], [5, 0]]), torch.tensor([[1, 6], [6, 1]]), torch.tensor([[2, 3], [3, 2]])]
    sets = [torch.tensor([[3, -1], [-1, -1], [4, 3]]), torch.tensor([[-1, -1], [2, -1], [-1, -1]])]
    counts = torch.rand(3, 2)
    model = PairAwareMCGNN(17, edge_dim=10).eval()
    moved = x.clone()
    moved[4] += 1.0
    with torch.no_grad():
        before, _ = model(x, edges, attr, views, sets, counts)
        after, _ = model(moved, edges, attr, views, sets, counts)
    assert (before != after).tolist() == [False, False, True]
