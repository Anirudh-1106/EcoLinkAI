"""Review schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class ReviewCreate(BaseModel):
    """
    One party's rating of the other, after an exchange completes.

    Deliberately carries a single rating with no indication of who it is
    about. The server decides that from the caller's side of the exchange --
    a buyer rates the supplier, a supplier rates the buyer. Letting the
    request name the direction would let a party rate itself, which is
    exactly what the previous shape required, since it demanded both ratings
    from whoever reviewed first.
    """
    exchange_id: uuid.UUID
    rating: int = Field(ge=1, le=5)
    feedback: str | None = Field(None, max_length=1000)


class ReviewResponse(BaseModel):
    """Review response schema."""
    id: uuid.UUID
    exchange_id: uuid.UUID
    supplier_rating: int | None = None
    buyer_rating: int | None = None
    supplier_feedback: str | None = None
    buyer_feedback: str | None = None
    created_at: datetime

    # Joined
    supplier_plant_name: str | None = None
    buyer_plant_name: str | None = None
    supplier_company_name: str | None = None
    buyer_company_name: str | None = None

    # What was traded. A rating is hard to interpret without knowing which
    # deal it refers to, and a company may have several with the same partner.
    material_name: str | None = None
    quantity: Decimal | None = None
    unit: str | None = None

    # Where the caller stands: "supplier" or "buyer", whether they have
    # already rated, and whether the other side has. Lets the interface offer
    # only the rating this viewer is entitled to give.
    my_role: str | None = None
    my_rating_submitted: bool = False
    counterparty_rating_submitted: bool = False

    model_config = {"from_attributes": True}


class ReviewListResponse(BaseModel):
    """Paginated review list."""
    items: list[ReviewResponse]
    total: int
    page: int
    page_size: int
