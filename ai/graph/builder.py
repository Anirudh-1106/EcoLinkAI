"""
Graph Builder module.

Constructs PyTorch Geometric Data objects from PostgreSQL.
"""

from __future__ import annotations

import torch
from sqlalchemy.orm import Session

from ai.features.edge_features import extract_edge_features
from ai.features.node_features import extract_node_features


def build_industrial_graph(db: Session):
    """
    Build a PyTorch Geometric Data graph object from PostgreSQL.
    """
    from torch_geometric.data import Data

    plants, x_arr, plant_id_to_idx = extract_node_features(db)
    edge_index_arr, edge_attr_arr, labels_arr, edge_times_arr = extract_edge_features(
        db, plant_id_to_idx
    )

    x = torch.tensor(x_arr, dtype=torch.float)
    edge_index = torch.tensor(edge_index_arr, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_arr, dtype=torch.float)
    y = torch.tensor(labels_arr, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        # When each request was made, so the split can be taken over time
        # rather than at random.
        edge_time=torch.tensor(edge_times_arr, dtype=torch.double),
        num_nodes=x.size(0),
    )

    return data, plant_id_to_idx, plants
