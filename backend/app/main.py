"""
EcoLinkAI API Backend.

AI-Powered Industrial Waste Exchange and Industrial Symbiosis
Recommendation Platform using Multi-Channel Graph Neural Networks (MC-GNN).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import (
    analytics,
    auth,
    companies,
    exchange_requests,
    exchanges,
    materials,
    plants,
    recommendations,
    requirements,
    reviews,
    waste_listings,
)
from app.services.recommendation_service import _try_load_model

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ecolinkai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("Initializing EcoLinkAI Backend...")
    _try_load_model()
    yield
    logger.info("Shutting down EcoLinkAI Backend...")


app = FastAPI(
    title="EcoLinkAI API",
    version="1.0.0",
    description=(
        "AI-Powered Industrial Waste Exchange Platform using Multi-Channel "
        "Graph Neural Network (MC-GNN) for circular economy industrial symbiosis."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Router Registration ──────────────────────────
API_PREFIX = "/api/v1"

app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(companies.router, prefix=API_PREFIX)
app.include_router(plants.router, prefix=API_PREFIX)
app.include_router(materials.router, prefix=API_PREFIX)
app.include_router(waste_listings.router, prefix=API_PREFIX)
app.include_router(requirements.router, prefix=API_PREFIX)
app.include_router(exchange_requests.router, prefix=API_PREFIX)
app.include_router(exchanges.router, prefix=API_PREFIX)
app.include_router(reviews.router, prefix=API_PREFIX)
app.include_router(recommendations.router, prefix=API_PREFIX)
app.include_router(analytics.router, prefix=API_PREFIX)


@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "project": "EcoLinkAI",
        "version": "1.0.0",
        "description": "AI-Powered Circular Economy Waste Exchange Platform",
        "docs": "/docs",
    }