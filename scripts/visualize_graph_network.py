"""
Network Graph Visualizer for EcoLinkAI Industrial Symbiosis Graph.

Generates a classic node-link graph diagram (circles as plant facilities,
directed arrows as waste exchange edges) colored by facility classification.
Saves the visualization to an image file.
"""

import sys
from pathlib import Path

# Add project root and backend to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from app.core.database import SessionLocal
from ai.graph.builder import build_industrial_graph

def visualize_network_graph():
    db = SessionLocal()
    try:
        data, plant_id_to_idx, plants = build_industrial_graph(db)

        # Create NetworkX Directed Graph
        G = nx.DiGraph()

        # Facility type color mapping based on substrings
        def get_node_color(ptype: str) -> str:
            ptype_lower = ptype.lower()
            if "processing" in ptype_lower:
                return "#00f2fe"      # Cyan
            elif "manufacturing" in ptype_lower or "assembly" in ptype_lower:
                return "#4facfe"      # Electric Blue
            elif "chemical" in ptype_lower:
                return "#ff0844"      # Vivid Red/Pink
            elif "energy" in ptype_lower:
                return "#f6d365"      # Warm Yellow
            elif "recycling" in ptype_lower:
                return "#43e97b"      # Emerald Green
            else:
                return "#e2e8f0"      # Light Slate Gray (Warehouse, Research Facility, etc.)

        # Add Nodes with Metadata
        for idx, plant in enumerate(plants):
            color = get_node_color(plant.plant_type)
            G.add_node(
                idx,
                name=plant.plant_name,
                type=plant.plant_type,
                district=plant.district,
                lat=float(plant.latitude),
                lon=float(plant.longitude),
                color=color
            )

        # Add Edges (Filter top 100 strongest edges for a clean, readable visual)
        assert data.edge_index is not None and data.edge_attr is not None
        edge_index = data.edge_index.cpu().numpy()
        edge_attr = data.edge_attr.cpu().numpy()

        # Sort edges by compatibility score (feature index 1)
        sorted_indices = np.argsort(edge_attr[:, 1])[::-1]
        top_edge_indices = sorted_indices[:100]

        for idx in top_edge_indices:
            src = int(edge_index[0, idx])
            dst = int(edge_index[1, idx])
            compat = float(edge_attr[idx, 1])
            dist = float(edge_attr[idx, 0]) * 500.0
            G.add_edge(src, dst, weight=compat, distance=dist)

        plt.figure(figsize=(16, 12), facecolor="#0B0F19")
        ax = plt.gca()
        ax.set_facecolor("#0B0F19")

        # Position nodes using spring layout to spread them out beautifully and prevent overlapping
        pos = nx.spring_layout(G, k=0.5, iterations=80, seed=42)

        # Draw Edges with transparency
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color="#38bdf8",
            alpha=0.25,
            arrows=True,
            arrowsize=10,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.15",
            width=1.0
        )

        # Draw Nodes using mapped colors from node attributes
        node_colors = [G.nodes[n]["color"] for n in G.nodes()]
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=320,
            edgecolors="#ffffff",
            linewidths=1.2
        )

        # Draw Node Labels (Facility Names - Abbreviated for readability)
        labels = {node: G.nodes[node]["name"].replace("Facility", "").replace("Plant", "").strip() for node in G.nodes()}
        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            font_size=7,
            font_color="#ffffff",
            font_weight="bold",
            verticalalignment="bottom"
        )

        # Create Custom Legend
        from matplotlib.patches import Patch
        type_colors = {
            "Processing": "#00f2fe",
            "Manufacturing / Assembly": "#4facfe",
            "Chemical": "#ff0844",
            "Energy": "#f6d365",
            "Recycling": "#43e97b",
            "Other (Warehouse/Lab)": "#e2e8f0",
        }
        legend_elements = [
            Patch(facecolor=color, edgecolor="#ffffff", label=ftype)
            for ftype, color in type_colors.items()
        ]
        legend = ax.legend(
            handles=legend_elements,
            loc="upper left",
            facecolor="#1e293b",
            edgecolor="#334155",
            labelcolor="#ffffff",
            fontsize=10,
            title="Facility Types",
            title_fontsize=11
        )
        if legend is not None:
            plt.setp(legend.get_title(), color='#38bdf8', weight='bold')

        plt.title("EcoLinkAI — Industrial Symbiosis Multi-Channel Graph (40 Plant Nodes, 853 Edges)",
                  color="#ffffff", fontsize=14, fontweight="bold", pad=20)
        plt.axis("off")
        plt.tight_layout()

        output_dir = PROJECT_ROOT / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "industrial_symbiosis_graph.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#0B0F19")
        plt.close()

        print(f"[+] Successfully generated graph visualization image at: {output_path}")

    finally:
        db.close()

if __name__ == "__main__":
    visualize_network_graph()
