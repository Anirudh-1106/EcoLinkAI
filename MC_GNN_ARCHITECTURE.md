# EcoLinkAI: System Architecture & AI Recommendation Engine

This document outlines the high-level architecture, entity relationships, the step-by-step transaction flow, and a deep-dive explanation of the Multi-Channel Graph Neural Network (MC-GNN) used for seller recommendations.

---

## 1. High-Level Project Architecture

*   **Backend (FastAPI - `backend/app/`)**:
    *   `models/`: Defines database tables (Company, Plant, WasteListing, Requirement) via SQLAlchemy.
    *   `routers/`: API endpoints that the frontend calls.
    *   `services/`: Core business logic, including `recommendation_service.py` where the MC-GNN AI matching algorithms live.
    *   `core/`: Setup for database connections and authentication.
*   **Frontend (React - `frontend/src/`)**: 
    *   Handles UI views (`RecommendationsPage`, `RequirementsPage`), components, and API client requests.
*   **Data & Scripts (`scripts/`, `datasets/`)**: 
    *   Python scripts that generate dummy CSV datasets and seed them into the PostgreSQL database.

---

## 2. Core Entities

*   **Companies**: Used for user accounts, overarching profiles, and calculating aggregate trust scores.
*   **Plants**: The physical locations of a company. Used heavily to calculate geographic distance and transport logistics.
*   **Materials**: Defines the actual byproduct being traded (e.g., PET Bottles, Glass Scrap) and its properties (hazardous, density).
*   **Exchange Requests**: Generated when a buyer decides they want a seller's waste. Tracks the status of the trade.
*   **Reviews**: Feedback left after a completed exchange, feeding back into the system to update a company's trust score.

---

## 3. Step-by-Step Transaction Flow

1.  **Listing Materials**: Sellers list their industrial byproducts as Waste Listings. Buyers post Requirements.
2.  **Finding a Match**: A Buyer selects their Requirement on the UI. The system searches the database for all Sellers who have that specific Material.
3.  **Calculating Current Trade Features**: The system calculates the logistics between the specific Buyer and Seller:
    *   *Distance & Transport Cost*: Calculated via Haversine formula between the two Plant GPS coordinates, factoring in vehicle rates.
    *   *Material Compatibility*: Delta between minimum required purity and actual listed purity.
    *   *Economic Viability*: Comparing Seller price vs Buyer budget.
    *   *Environmental Impact*: Estimated carbon emission (from transport) versus carbon saving (from recycling).
4.  **AI Prediction**: The MC-GNN processes the graph data (historical context) and the current trade features to output a final Match Percentage.
5.  **The Transaction**: The Buyer requests to buy, creating an Exchange Request. The physical waste is transported.
6.  **The Feedback Loop**: Once delivered, both companies leave Reviews. This updates the graph with a new successful edge, making the AI smarter for the next recommendation.

---

## 4. The Graph Input: Nodes and Edges

The AI models use a **unified graph** dynamically built straight from the PostgreSQL database:
*   **Nodes (Plants)**: Every physical facility is a node. Their raw **Node Features** are pulled from the DB (e.g., `[latitude, longitude, industry_type, company_trust_score, verification_status, company_size]`).
*   **Edges (Past Transactions)**: A connection is drawn between two plants if they have completed an exchange. The raw **Edge Features** are pulled from the DB (e.g., `[supplier_rating, buyer_rating, transport_cost, actual_carbon_saving]`).

---

## 5. The MC-GNN Layers

Once the features are gathered, the Multi-Channel Graph Neural Network (MC-GNN) uses three distinct "channels" simultaneously to evaluate the Seller's reputation and reliability.

### Channel 1: GCN (Graph Convolutional Network) – *The Reputation Aggregator*
*   **The Concept**: *"You are known by the company you keep."*
*   **Its Role**: GCN takes a Seller's baseline Node Features and mathematically mixes them with the features of every company they have traded with. If a Seller frequently trades with highly-verified, massive corporations, the GCN upgrades the Seller’s internal reputation. It maps the hierarchy of the entire industrial supply chain.

### Channel 2: GAT (Graph Attention Network) – *The Quality Filter*
*   **The Concept**: Uses **Attention** to weigh historical relationships based on quality.
*   **Its Role**: GAT uses the Edge Features (the actual transaction data and reviews) to calculate an attention multiplier. It multiplies successful 5-star trades by a high attention weight (e.g., 0.95) and failed 1-star trades by a low weight (e.g., 0.05). This forces the AI to "pay attention" only to the most relevant, reliable historical relationships when scoring a seller.

### Channel 3: GraphSAGE (Sample and Aggregate) – *The Cold-Start Solver*
*   **The Concept**: Handles brand new users who have 0 edges in the graph.
*   **Its Role**: Standard models fail on new companies due to the "cold start" problem. GraphSAGE is "inductive"—it doesn't need prior edges. It looks purely at the new company's raw Node Features (e.g., "Medium Textile Plant in Kerala") and mathematically samples the graph to find similar established peers. It infers how the new node will behave based on those peers, preventing new sellers from getting a 0% score.

### HSIC (Hilbert-Schmidt Independence Criterion)
*   **Where it is used**: During the **training** phase of the MC-GNN.
*   **Its Role**: Because we have three different channels (GCN, GAT, GraphSAGE) looking at the exact same graph, they might accidentally learn the exact same redundant patterns. HSIC is applied as a mathematical penalty (a regularization loss). The AI is forced to minimize the HSIC between the three channels, guaranteeing that GCN, GAT, and GraphSAGE all capture strictly **independent, non-redundant information**.

---

## 6. The Fusion Step & Final Output

The output of the GCN/GAT/GraphSAGE layers is called an **Embedding**—a mathematically condensed vector representing the Node and Edge features from the historical graph.

1.  **The Fusion**: The system concatenates (combines) this deep historical Embedding with the **Current Trade Features** (the math calculated for distance, compatibility, and cost in Step 3).
2.  **The Decision Layer**: This combined array of numbers is fed into a final Multi-Layer Perceptron (MLP) neural network, which weighs the current logistics against the historical trust.
3.  **The Score**: The neural network outputs a raw probability value (e.g., `0.9234`). We multiply this by 100 to give the final **92% AI Match**. Sellers are then ranked highest to lowest and displayed on the UI.

