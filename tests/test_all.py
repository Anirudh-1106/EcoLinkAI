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
        "supplier_rating": 1,
        "buyer_rating": 1,
        "supplier_feedback": "x",
        "buyer_feedback": "x",
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
                "supplier_rating": 1,
                "buyer_rating": 1,
                "supplier_feedback": "x",
                "buyer_feedback": "x",
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
