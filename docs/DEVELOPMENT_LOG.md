
# PROJECT LOG

## Project

**EcoLinkAI**

> AI-Powered Circular Economy Waste Exchange Network Using Multi-Channel Graph Neural Networks (MC-GNN)

---

# Development Log

## Date: 27 July 2026

### Phase

**Database Design & Architecture**

---

## 1. Completed the Remaining Database Models

Designed and finalized the following SQLAlchemy 2.0 models:

### Exchange

**Purpose**

* Represents an approved waste exchange between supplier and buyer after an AI recommendation is accepted.

**Implemented**

* Exchange lifecycle
* Shipment lifecycle
* Agreed price
* Transport cost
* Carbon emission tracking
* Carbon saving tracking
* Delivery information
* Completion notes
* One-to-one relationship with `ExchangeRequest`

---

### TransportRate

**Purpose**

* Stores transportation pricing information used by the AI recommendation engine.

**Implemented**

* Vehicle type
* Maximum carrying capacity
* Cost per kilometer
* Carbon emission per kilometer
* Fuel type
* Average vehicle speed
* Active/inactive status

**Architecture Decision**

* Transportation cost will **not** be stored in `WasteListing`.
* Transport cost will be dynamically calculated by the recommendation engine using:

  * Distance
  * Vehicle
  * Capacity
  * TransportRate table

---

### Review

**Purpose**

* Stores post-exchange feedback between companies.

**Implemented**

* Supplier rating
* Buyer rating
* Supplier feedback
* Buyer feedback
* Database constraints (ratings between 1–5)

**Architecture Decision**

* Each completed exchange allows:

  * Buyer → Supplier review
  * Supplier → Buyer review

These ratings will later contribute to company trust scores.

---

### Analytics

**Purpose**

* Stores aggregated statistics for each industrial plant.

**Implemented**

* Waste generated
* Waste exchanged
* Revenue generated
* Transport cost
* Carbon emission
* Carbon savings
* Average AI match score
* Average buyer rating
* Average supplier rating

---

# 2. Database Completion

The complete database schema has now been designed.

### Implemented Models

* BaseModel
* Company
* Plant
* IndustryContact
* Verification
* Material
* WasteListing
* Requirement
* ExchangeRequest
* Exchange
* TransportRate
* Review
* Analytics

**Total Models:** **13**

---

# 3. Project Structure Improvements

Created a new package:

```
app/constants/
```

Added:

```
constants/
│
├── __init__.py
├── ai.py
├── pagination.py
├── transport.py
└── validation.py
```

These files centralize:

* AI configuration
* Pagination limits
* Validation limits
* Transport-related configuration

instead of hardcoding values throughout the project.

---

# 4. Architecture Decisions

Several important architectural decisions were finalized.

### Company Hierarchy

```
Company
    │
    ├── IndustryContact
    ├── Plant
    └── Verification
```

A company may own multiple plants.

---

### Material Normalization

Instead of storing material names repeatedly:

```
WasteListing
material_name
```

the system now uses

```
Material
      │
      ├── WasteListing
      └── Requirement
```

This avoids duplicate entries and improves AI matching.

---

### Exchange Flow

```
WasteListing
        │
        ▼
ExchangeRequest
        ▲
Requirement
        │
        ▼
Exchange
        │
        ▼
Review
```

---

### Transport Cost Strategy

Transport cost is **not stored** inside waste listings.

Instead, it will be calculated dynamically using:

* Plant coordinates
* Distance
* Vehicle selection
* TransportRate
* Carbon factors

---

# 5. Code Quality Improvements

All models now follow a consistent production standard:

* SQLAlchemy 2.0 style
* UUID primary keys
* `TYPE_CHECKING` imports
* Proper `back_populates`
* Typed relationships
* `__repr__()` methods
* Centralized enums
* Normalized database design

---

# 6. Workflow Decision

The development workflow has been finalized as:

```
Database Models
        ↓
Relationship Audit
        ↓
Alembic Migration
        ↓
PostgreSQL Tables
        ↓
Pydantic Schemas
        ↓
Service Layer
        ↓
Routers
        ↓
Swagger Testing
        ↓
Frontend Integration
        ↓
MC-GNN Integration
```

---

# 7. Pending Tasks

## Immediate Next Task

Perform a complete relationship audit of all models.

This includes:

* Verify every `back_populates`
* Verify foreign keys
* Verify cascade rules
* Verify one-to-one relationships
* Check indexes
* Detect circular imports

---

## After Audit

* Configure Alembic
* Generate first migration
* Apply migration to PostgreSQL
* Verify tables in pgAdmin

---

# Current Project Status

| Component               | Status     |
| ----------------------- | ---------- |
| Database Design         | ✅ Complete |
| SQLAlchemy Models       | ✅ Complete |
| Constants Package       | ✅ Complete |
| Relationship Validation | ⏳ Pending  |
| Alembic                 | ⏳ Pending  |
| PostgreSQL Schema       | ⏳ Pending  |
| Pydantic Schemas        | ⏳ Pending  |
| Service Layer           | ⏳ Pending  |
| API Routers             | ⏳ Pending  |
| Frontend                | ⏳ Pending  |
| MC-GNN Integration      | ⏳ Pending  |

---

## Notes for Next Session

1. Perform a full audit of all SQLAlchemy models.
2. Fix any relationship inconsistencies before generating migrations.
3. Configure Alembic and generate the initial migration.
4. Create the PostgreSQL schema and verify all tables.
5. Begin implementing Pydantic schemas.

---

This log reflects the work completed on **27 July 2026** and establishes a clear checkpoint before transitioning from database design to implementation.
