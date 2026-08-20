"""
Graph visualization & connectivity inspection script for EcoLinkAI.
Prints node connections, edge attributes, and symbiosis relationships.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

from app.core.database import SessionLocal
from app.models import Plant, Company, WasteListing, Requirement, ExchangeRequest, Material
from ai.graph.builder import build_industrial_graph


def inspect_graph():
    db = SessionLocal()
    try:
        print("\n" + "=" * 80)
        print(" ECOLINKAI INDUSTRIAL GRAPH CONTEXT & EDGE INSPECTION")
        print("=" * 80)

        data, plant_id_to_idx, plants = build_industrial_graph(db)
        idx_to_plant = {idx: p for p, idx in plant_id_to_idx.items()}

        print(f"\nTotal Nodes (Facilities): {data.num_nodes}")
        print(f"Total Edges (Symbiosis Links): {data.num_edges}\n")

        print("-" * 80)
        print(" SAMPLE NODE PROPERTIES (FACILITIES)")
        print("-" * 80)
        for i in range(min(5, len(plants))):
            p = plants[i]
            comp_name = p.company.company_name if p.company else "Unknown"
            print(f" Node {i:2d} | ID: {str(p.id)[:8]}... | Plant: {p.plant_name:<30} | Company: {comp_name:<25} | Type: {p.plant_type}")

        print("\n" + "-" * 80)
        print(" SAMPLE EDGE CONNECTIONS & SYMBIOSIS PROPERTIES (SUPPLIER -> BUYER)")
        print("-" * 80)

        # Query top exchange requests representing edges in the graph
        requests = db.query(ExchangeRequest).limit(10).all()

        print(f"{'SUPPLIER PLANT':<25} -> {'BUYER PLANT':<25} | {'MATERIAL':<20} | {'DIST (KM)':<10} | {'COMPAT %':<9} | {'CARBON SAVED'}")
        print("-" * 110)

        for req in requests:
            sup_name = req.supplier_plant.plant_name if req.supplier_plant else "Unknown"
            buyer_name = req.buyer_plant.plant_name if req.buyer_plant else "Unknown"
            mat_name = req.waste_listing.material.material_name if (req.waste_listing and req.waste_listing.material) else "Industrial Byproduct"
            
            dist = f"{float(req.distance_km):.1f} km" if req.distance_km else "N/A"
            compat = f"{float(req.compatibility_score):.1f}%" if req.compatibility_score else "N/A"
            carbon = f"{float(req.estimated_carbon_saving):.1f} kg CO2" if req.estimated_carbon_saving else "N/A"

            print(f"{sup_name[:24]:<25} -> {buyer_name[:24]:<25} | {mat_name[:19]:<20} | {dist:<10} | {compat:<9} | {carbon}")

        print("=" * 110 + "\n")

    finally:
        db.close()


if __name__ == "__main__":
    inspect_graph()
