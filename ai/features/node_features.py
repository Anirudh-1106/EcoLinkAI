"""
Node Feature Extraction for Plants.

Constructs dense numerical feature vectors for Plant nodes in the industrial graph:
- Location (Latitude, Longitude)
- Trust Score (0-100 normalized)
- Active Waste Listings Count
- Open Requirements Count
- Historical Exchanges Count
- Average Review Rating
- One-hot encoded Industry Sector
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.enums.exchange import ExchangeStatus, RequirementStatus, WasteStatus
from app.models.company import Company
from app.models.exchange import Exchange
from app.models.exchange_request import ExchangeRequest
from app.models.plant import Plant
from app.models.requirement import Requirement
from app.models.review import Review
from app.models.waste_listing import WasteListing

# Values a plant receives when the platform knows nothing about it yet: a
# company with no trust score, a seller with no reviews. Named so that any
# other data source fed to the model -- the EPA real-data study, for one --
# fills the same gaps with the same neutral values rather than its own guesses.
NEUTRAL_TRUST = 0.5
NEUTRAL_RATING = 0.7

# Counts are divided by these and capped at 1.0.
LISTINGS_SATURATION = 20.0
REQUIREMENTS_SATURATION = 20.0
EXCHANGES_SATURATION = 50.0

# Standard industry sectors for one-hot encoding
INDUSTRY_SECTORS = [
    "Chemicals & Petrochemicals",
    "Metal & Metallurgy",
    "Textiles & Apparel",
    "Food Processing",
    "Paper & Packaging",
    "Automotive",
    "Construction & Materials",
    "Electronics",
    "Pharmaceuticals",
    "General Manufacturing",
]


def extract_node_features(db: Session) -> tuple[list[Plant], np.ndarray, dict[str, int]]:
    """
    Extract feature matrix X for all active plants.

    Returns:
        plants: List of Plant objects
        features: 2D numpy array of shape (N_plants, N_features)
        plant_id_to_idx: Mapping from Plant UUID string to row index
    """
    plants = db.query(Plant).join(Company).filter(Plant.is_active.is_(True)).all()
    plant_id_to_idx = {str(p.id): idx for idx, p in enumerate(plants)}

    features_list = []

    for plant in plants:
        comp = plant.company

        # Location
        lat = float(plant.latitude) / 90.0
        lon = float(plant.longitude) / 180.0

        # Trust score
        trust = float(comp.trust_score) / 100.0 if comp and comp.trust_score else NEUTRAL_TRUST

        # Listing count
        active_listings = (
            db.query(func.count(WasteListing.id))
            .filter(WasteListing.plant_id == plant.id, WasteListing.status == WasteStatus.AVAILABLE)
            .scalar()
            or 0
        )
        norm_listings = min(active_listings / LISTINGS_SATURATION, 1.0)

        # Requirement count
        open_reqs = (
            db.query(func.count(Requirement.id))
            .filter(Requirement.plant_id == plant.id, Requirement.status == RequirementStatus.OPEN)
            .scalar()
            or 0
        )
        norm_reqs = min(open_reqs / REQUIREMENTS_SATURATION, 1.0)

        # Historical completed exchanges
        ex_count = (
            db.query(func.count(Exchange.id))
            .join(ExchangeRequest, Exchange.exchange_request_id == ExchangeRequest.id)
            .filter(
                (ExchangeRequest.supplier_plant_id == plant.id)
                | (ExchangeRequest.buyer_plant_id == plant.id),
                Exchange.exchange_status == ExchangeStatus.COMPLETED,
            )
            .scalar()
            or 0
        )
        norm_ex = min(ex_count / EXCHANGES_SATURATION, 1.0)

        # Average rating
        avg_rating = (
            db.query(func.avg(Review.supplier_rating))
            .join(Exchange, Review.exchange_id == Exchange.id)
            .join(ExchangeRequest, Exchange.exchange_request_id == ExchangeRequest.id)
            .filter(ExchangeRequest.supplier_plant_id == plant.id)
            .scalar()
        )
        norm_rating = (float(avg_rating) / 5.0) if avg_rating else NEUTRAL_RATING

        # Industry sector one-hot
        sector_vec = [0.0] * len(INDUSTRY_SECTORS)
        if comp and comp.industry_type in INDUSTRY_SECTORS:
            sector_vec[INDUSTRY_SECTORS.index(comp.industry_type)] = 1.0
        else:
            sector_vec[-1] = 1.0  # General Manufacturing fallback

        vec = [lat, lon, trust, norm_listings, norm_reqs, norm_ex, norm_rating] + sector_vec
        features_list.append(vec)

    return plants, np.array(features_list, dtype=np.float32), plant_id_to_idx
