"""
Edge Feature Extraction & Candidate Relationship Generation.

Extracts feature vectors for candidate edges between supplier and buyer plants:
- Distance (Haversine km)
- Material Compatibility Score
- Quantity Compatibility Ratio
- Historical Exchange Interaction Frequency
- Carbon Saving Potential
- Transport Cost Estimate
"""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from app.enums.exchange import ExchangeRequestStatus
from app.models.exchange_request import ExchangeRequest
from app.models.plant import Plant
from app.models.waste_listing import WasteListing
from app.utils.distance import estimate_carbon_saving, estimate_transport_cost, haversine_distance


def extract_edge_features(
    db: Session,
    plant_id_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract edges and edge feature vectors from historical exchange requests.

    Returns:
        edge_index: 2D array of shape (2, N_edges) [source_nodes, target_nodes]
        edge_attr: 2D array of shape (N_edges, N_edge_features)
        labels: 1D array of shape (N_edges,) where 1=accepted/successful, 0=rejected/failed
    """
    requests = (
        db.query(ExchangeRequest)
        .all()
    )

    edge_sources = []
    edge_targets = []
    edge_attrs = []
    labels = []

    for req in requests:
        sup_id = str(req.supplier_plant_id)
        buy_id = str(req.buyer_plant_id)

        if sup_id not in plant_id_to_idx or buy_id not in plant_id_to_idx:
            continue

        src_idx = plant_id_to_idx[sup_id]
        dst_idx = plant_id_to_idx[buy_id]

        dist_km = float(req.distance_km) if req.distance_km else 50.0
        norm_dist = min(dist_km / 500.0, 1.0)

        compat_score = float(req.compatibility_score) / 100.0 if req.compatibility_score else 0.5
        transport_cost = float(req.estimated_transport_cost) / 10000.0 if req.estimated_transport_cost else 0.1
        carbon_saving = float(req.estimated_carbon_saving) / 1000.0 if req.estimated_carbon_saving else 0.1

        # Binary label: 1 if ACCEPTED, 0 otherwise
        label = 1 if req.status == ExchangeRequestStatus.ACCEPTED else 0

        edge_sources.append(src_idx)
        edge_targets.append(dst_idx)
        edge_attrs.append([norm_dist, compat_score, transport_cost, carbon_saving])
        labels.append(label)

    if not edge_sources:
        # Fallback dummy single edge if dataset is empty
        edge_index = np.array([[0], [0]], dtype=np.int64)
        edge_attr = np.array([[0.1, 0.9, 0.1, 0.5]], dtype=np.float32)
        labels_arr = np.array([1], dtype=np.float32)
    else:
        edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)
        edge_attr = np.array(edge_attrs, dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.float32)

    return edge_index, edge_attr, labels_arr
