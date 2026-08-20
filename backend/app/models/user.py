"""
User model for authentication.

Users belong to a company and have role-based access control.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING
from enum import Enum as PyEnum
from sqlalchemy import Boolean, Enum, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.company import Company


class UserRole(str, PyEnum):
    ADMIN = "admin"
    COMPANY = "company"
    PLANT_OPERATOR = "plant_operator"


class User(BaseModel):
    """
    Represents an authenticated user in the EcoLinkAI platform.

    Each user is associated with a company and has a role
    that determines their access level.
    """

    __tablename__ = "users"

    # ─── Authentication ───────────────────────────────
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    full_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    # ─── Role ─────────────────────────────────────────
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole),
        default=UserRole.COMPANY,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    # ─── Company Association ──────────────────────────
    company_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ─── Relationships ────────────────────────────────
    company: Mapped["Company | None"] = relationship(
        back_populates="users",
    )

    def __repr__(self) -> str:
        return (
            f"<User("
            f"id={self.id}, "
            f"email='{self.email}', "
            f"role='{self.role.value}'"
            f")>"
        )
