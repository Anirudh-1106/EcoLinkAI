"""Authentication schemas."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """Company registration request."""
    # User info
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=2, max_length=255)

    # Company info
    company_name: str = Field(min_length=2, max_length=255)
    industry_type: str = Field(min_length=2, max_length=100)
    registration_number: str = Field(min_length=5, max_length=100)
    gst_number: str = Field(min_length=15, max_length=15)
    license_number: str = Field(min_length=5, max_length=100)


class LoginRequest(BaseModel):
    """Login credentials."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user_id: uuid.UUID
    email: str
    full_name: str
    role: str
    company_id: uuid.UUID | None = None
    company_name: str | None = None


class UserResponse(BaseModel):
    """User profile response."""
    id: uuid.UUID
    email: str
    full_name: str
    role: str
    is_active: bool
    company_id: uuid.UUID | None = None

    model_config = {"from_attributes": True}
