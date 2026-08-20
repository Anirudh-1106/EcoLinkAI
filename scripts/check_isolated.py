import sys
from pathlib import Path

# Add project root and backend to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import networkx as nx
from app.core.database import SessionLocal
from ai.graph.builder import build_industrial_graph

def check_degrees():
    db = SessionLocal()
    try:
        data, plant_id_to_idx, plants = build_industrial_graph(db)
        
        # Build graph using all 853 edges
        G = nx.DiGraph()
        G.add_nodes_from(range(len(plants)))
        
        assert data.edge_index is not None, "Graph does not contain any edges (edge_index is None)"
        edge_index = data.edge_index.cpu().numpy()
        for idx in range(edge_index.shape[1]):
            src = int(edge_index[0, idx])
            dst = int(edge_index[1, idx])
            G.add_edge(src, dst)
            
        print("Total Nodes:", len(plants))
        print("Total Edges in NetworkX:", G.number_of_edges())
        
        isolated = list(nx.isolates(G))
        print("Truly Isolated Nodes (Degree = 0):", len(isolated))
        for idx in isolated:
            print(f"  - Node {idx}: {plants[idx].plant_name} (Type: {plants[idx].plant_type})")
            
        print("\nDegrees of all nodes:")
        for idx in range(len(plants)):
            print(f"  Node {idx:<2}: {plants[idx].plant_name:<30} -> Degree: {G.degree(idx)}")
            
    finally:
        db.close()

if __name__ == "__main__":
    check_degrees()
