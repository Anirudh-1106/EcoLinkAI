# EcoLinkAI — AI-Powered Industrial Waste Exchange & Symbiosis Network

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-emerald.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.6-blueviolet.svg)](https://pyg.org)
[![React](https://img.shields.io/badge/React-18-cyan.svg)](https://react.dev)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-v3-38bdf8.svg)](https://tailwindcss.com)

**EcoLinkAI** connects industrial plants generating reusable waste byproducts with manufacturing facilities that can consume them as raw material inputs.

Rather than relying on manually assigned static matching rules, EcoLinkAI models the industrial ecosystem as a multi-view graph and uses a **Multi-Channel Graph Neural Network (MC-GNN)** to rank exchange partners considering:
1. Material compatibility
2. Waste quantity feasibility
3. Quality / purity requirements
4. Geographic Haversine distance
5. Transportation costs & feasibility
6. Historical exchange interaction behavior
7. Trust & reputation scores
8. Carbon emission reduction & avoided virgin impact

---

## 🏛️ Architecture Overview

```
                  ECOLINKAI ARCHITECTURE

        ┌─────────────────────────────────────────┐
        │       React 18 + Vite Frontend          │
        │                                         │
        │ Landing Page | Dashboard | Map Route    │
        │ Waste Listings | Partner Recommendations│
        │ Exchange Requests | Transactions       │
        └────────────────────┬────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────┐
        │       FastAPI REST Service (v1)         │
        │                                         │
        │ Auth (JWT) | Companies | Plants | Waste │
        │ Requirements | Exchanges | Analytics   │
        └────────────────────┬────────────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
┌──────────────────┐                  ┌──────────────────┐
│   PostgreSQL     │                  │  MC-GNN Engine   │
│                  │                  │                  │
│ Companies (20)   │                  │ Feature Encoder  │
│ Plants (40)      │                  │ GCN/GAT/SAGE     │
│ Materials (141)  │                  │ HSIC Loss        │
│ Waste (346)      │                  │ Attention Fusion │
│ Requests (853)   │                  │ Link Predictor   │
│ Transactions(345)│                  └──────────────────┘
│ Reviews (345)    │
└──────────────────┘
```

---

## 🔬 MC-GNN Technical Implementation

The recommendation engine implements the novel **Multi-Channel Graph Neural Network (MC-GNN)** framework:
- **Over-Squashing Prevention**: Employs multiple parallel GNN backbones (**GCN**, **GAT**, **GraphSAGE**) as distinct channels to extract structural relations from multi-view perspectives.
- **Hilbert-Schmidt Independence Criterion (HSIC)**: Enforces feature diversity between channels via HSIC loss regularization:
  $$\text{HSIC}(Z_i, Z_j) = \frac{1}{(N-1)^2} \text{Tr}(K_i H K_j H)$$
- **Attention Fusion**: Adaptively weights and combines embeddings across channels.
- **Explainability Layer**: Converts neural embeddings and graph edge features into human-readable partner cards with explicit compatibility, distance, transport cost, trust, and carbon metrics.

---

## 🚀 Quickstart & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ & npm
- PostgreSQL database running on `localhost:5432` with database `ecolinkai`

### 1. Environment Configuration
Copy `.env.example` to `.env` in `backend/`:
```bash
DATABASE_URL=postgresql://postgres:postgres123@localhost:5432/ecolinkai
SECRET_KEY=ecolinkai-dev-secret-change-in-production
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

### 2. Database Migration & Seeding
```bash
# Run Alembic migrations to create PostgreSQL tables
cd backend
python -m alembic upgrade head

# Seed synthetic industrial ecosystem data (2,090 records)
cd ..
python -m scripts.database.seed
```

### 3. Train MC-GNN AI Model
```bash
python -m ai.training.train
```

### 4. Run Backend Server
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
Swagger API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Run Frontend Application
```bash
cd frontend
npm install
npm run dev
```
Open Browser: [http://localhost:5173](http://localhost:5173)

---

## 🧪 Testing

Run automated backend & AI recommendation tests:
```bash
python -m pytest tests/test_all.py -v
```

---

## 👤 Credentials for Demo
- **Admin Login**: `admin@ecolink.ai` / `admin123`
- **Company Login**: `saran-sankaran@industry.in` / `password123`
