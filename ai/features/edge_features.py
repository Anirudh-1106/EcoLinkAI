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
from app.utils.edge_encoding import encode_edge_features


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
    # Ordered oldest first so each request's history can be accumulated from
    # the ones that genuinely preceded it. Requests seeded before created_at
    # was persisted all share one timestamp; id is a stable tie-break so the
    # ordering stays deterministic rather than whatever the database returns.
    requests = (
        db.query(ExchangeRequest)
        .order_by(ExchangeRequest.created_at.asc(), ExchangeRequest.id.asc())
        .all()
    )

    edge_sources = []
    edge_targets = []
    edge_attrs = []
    edge_times = []
    labels = []

    # Built up as the replay advances, so a request only ever sees deals that
    # closed before it. Counting the whole table instead would fold the
    # request's own outcome -- and every later one, including the test set's --
    # back into its input.
    prior_success_counts: dict[tuple[str, str], int] = {}
    seen_pairs: set[tuple[str, str]] = set()

    for req in requests:
        sup_id = str(req.supplier_plant_id)
        buy_id = str(req.buyer_plant_id)

        if sup_id not in plant_id_to_idx or buy_id not in plant_id_to_idx:
            continue

        src_idx = plant_id_to_idx[sup_id]
        dst_idx = plant_id_to_idx[buy_id]

        pair = (sup_id, buy_id)

        # Raw, unscaled values -- encode_edge_features owns all the scaling so
        # that training and serving cannot end up normalising differently.
        attrs = encode_edge_features(
            distance_km=float(req.distance_km) if req.distance_km else 50.0,
            material_compatibility=(
                float(req.material_compatibility)
                if req.material_compatibility is not None
                else 100.0
            ),
            quantity_compatibility=(
                float(req.quantity_compatibility)
                if req.quantity_compatibility is not None
                else 50.0
            ),
            quality_compatibility=(
                float(req.quality_compatibility)
                if req.quality_compatibility is not None
                else 100.0
            ),
            transport_cost=float(req.estimated_transport_cost) if req.estimated_transport_cost else 0.0,
            carbon_saving=float(req.estimated_carbon_saving) if req.estimated_carbon_saving else 0.0,
            prior_successes=prior_success_counts.get(pair, 0),
            has_prior_interaction=pair in seen_pairs,
        )

        # Binary label: 1 if ACCEPTED, 0 otherwise
        label = 1 if req.status == ExchangeRequestStatus.ACCEPTED else 0

        edge_sources.append(src_idx)
        edge_targets.append(dst_idx)
        edge_attrs.append(attrs)
        edge_times.append(req.created_at.timestamp() if req.created_at else 0.0)
        labels.append(label)

        # Record this request only after it has been encoded, so it never
        # contributes to its own history.
        if label == 1:
            prior_success_counts[pair] = prior_success_counts.get(pair, 0) + 1
        seen_pairs.add(pair)

    if not edge_sources:
        # Fallback dummy single edge if dataset is empty
        edge_index = np.array([[0], [0]], dtype=np.int64)
        edge_attr = np.array([[0.1, 1.0, 0.9, 1.0, 0.1, 0.5, 0.0, 0.0]], dtype=np.float32)
        labels_arr = np.array([1], dtype=np.float32)
        times_arr = np.array([0.0], dtype=np.float64)
    else:
        edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)
        edge_attr = np.array(edge_attrs, dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.float32)
        times_arr = np.array(edge_times, dtype=np.float64)

    return edge_index, edge_attr, labels_arr, times_arr
