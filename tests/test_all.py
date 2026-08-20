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


def test_list_companies_api():
    response = client.get("/api/v1/companies")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) > 0


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
