
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

Here's a concise but detailed **Work Log for 28 July 2026**.

---

## **Date:** 28 July 2026

### **Objective**

Continue the development of the synthetic dataset generation pipeline for the **EcoLinkAI – AI-Powered Circular Economy Waste Exchange Network using MC-GNN** project by implementing the foundational relational datasets.

---

## **Work Completed**

### **1. Completed Companies Dataset**

* Finalized the `generate_companies.py` script.
* Generated synthetic company records with realistic attributes including:

  * Company ID
  * Company Name
  * Industry Sector
  * Registration Number
  * Contact Information
  * Headquarters Location
  * Trust Score
  * Verification Status
  * Company Size
* Performed data validation to ensure:

  * No duplicate Company IDs
  * No duplicate Registration Numbers
  * No duplicate Emails
  * No missing values
* Exported the dataset as:

  * `datasets/csv/companies.csv`

---

### **2. Developed Plants Dataset**

* Created metadata file:

  * `plant_types.json`
* Implemented `generate_plants.py`.
* Established a **one-to-many relationship** between companies and plants.
* Generated **1–3 plants for each company**.
* Included attributes such as:

  * Plant ID
  * Company ID
  * Plant Name
  * Plant Type
  * District
  * State
  * Country
* Validated generated data.
* Exported:

  * `datasets/csv/plants.csv`

---

### **3. Developed Materials Dataset**

* Created metadata files:

  * `materials.json`
  * `material_types.json`
  * `measurement_units.json`
  * `boolean_values.json`
* Implemented `generate_materials.py`.
* Generated multiple material records for every plant.
* Enhanced the dataset schema by adding:

  * Material Type
  * Measurement Unit
  * Recyclable Flag
  * Hazardous Flag
* Final material dataset consists of **9 attributes**, making it suitable for later MC-GNN feature engineering.
* Validated generated data and exported:

  * `datasets/csv/materials.csv`

---

### **4. Dataset Pipeline Enhancement**

* Strengthened the relational hierarchy of the project:

```text
Companies
    │
    ▼
Plants
    │
    ▼
Materials
```

* Established foreign-key relationships between datasets to support future modules.

---

### **5. Development Environment Improvements**

* Resolved Python virtual environment conflicts.
* Consolidated development into a single project virtual environment.
* Installed backend dependencies required for FastAPI and SQLAlchemy.
* Eliminated IDE import resolution issues.

---

### **6. Git Version Control**

* Added newly created dataset generators and metadata files to the repository.
* Prepared commits for the completed Companies, Plants, and Materials dataset generation modules.
* Repository is ready to be pushed after verification.

---

## **Deliverables Completed**

* `generate_companies.py`
* `generate_plants.py`
* `generate_materials.py`
* `companies.csv`
* `plants.csv`
* `materials.csv`
* Additional metadata JSON files for plants and materials.

---

## **Plan for Next Session (29 July 2026)**

* Begin implementation of the **Waste Listings Dataset**.
* Design realistic waste availability records linked to materials and plants.
* Include attributes such as quantity, purity, pricing, urgency, storage conditions, listing status, and environmental indicators.
* Prepare the dataset as the primary input for the MC-GNN recommendation model.
