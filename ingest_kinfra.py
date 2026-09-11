import sys
import os
import openpyxl
import uuid
from datetime import date, timedelta

# Add backend to path so we can import models and database
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.core.database import SessionLocal
from app.models.company import Company
from app.models.plant import Plant
from app.models.material import Material
from app.models.waste_listing import WasteListing
from app.enums.exchange import WasteStatus
from app.enums.company import VerificationStatus
from app.enums.material import MaterialCategory, HazardClass

def ingest_kinfra_data(lat=8.554790, lon=76.858660):
    db = SessionLocal()
    
    wb = openpyxl.load_workbook('datasets/kinfra_data.xlsx')
    sheet = wb.active
    assert sheet is not None, "Workbook has no active sheet"
    
    # Read headers
    headers = [cell.value for cell in sheet[1]]
    
    count = 0
    for row_idx in range(2, sheet.max_row + 1):
        row_data = [sheet.cell(row=row_idx, column=col_idx).value for col_idx in range(1, len(headers) + 1)]
        row_dict = dict(zip(headers, row_data))
        
        company_name = row_dict.get("Company Name")
        if not company_name:
            continue
        company_name = str(company_name)
            
        material_name = str(row_dict.get("Waste Material") or "Unknown")
        quantity_kg = row_dict.get("Waste Generated (per month, kg)")
        price_per_kg = row_dict.get("Cost per kg (₹)")
        
        print(f"Ingesting: KINFRA - {company_name}")
        
        # 1. Create Company
        company_id = uuid.uuid4()
        company = Company(
            id=company_id,
            company_name=f"KINFRA - {company_name}",
            industry_type="Manufacturing",
            registration_number=f"KIN-{count}",
            gst_number=f"GST-{count}",
            license_number=f"LIC-{count}",
            verification_status=VerificationStatus.PENDING,
            trust_score=50.0, # Brand new user
            is_active=True
        )
        db.add(company)
        
        # 2. Create Plant
        plant_id = uuid.uuid4()
        plant = Plant(
            id=plant_id,
            company_id=company_id,
            plant_name=f"{company_name} - Main Facility",
            plant_type="Manufacturing Unit",
            latitude=lat, # Exact KINFRA coordinates
            longitude=lon,
            address="KINFRA Industrial Park",
            district="Trivandrum",
            state="Kerala",
            country="India",
        )
        db.add(plant)
        
        # 3. Create or Find Material
        material = db.query(Material).filter(Material.material_name == material_name).first()
        if not material:
            material = Material(
                id=uuid.uuid4(),
                material_name=material_name,
                material_category=MaterialCategory.OTHER,
                hazard_class=HazardClass.NON_HAZARDOUS,
            )
            db.add(material)
            db.commit() # Commit to generate ID
            db.refresh(material)
            
        # 4. Create Waste Listing
        waste_price = float(str(price_per_kg)) if price_per_kg is not None else 0.0
        waste_qty = float(str(quantity_kg)) if quantity_kg is not None else 0.0
        
        listing = WasteListing(
            id=uuid.uuid4(),
            plant_id=plant_id,
            material_id=material.id,
            quantity=waste_qty,
            unit="kg",
            purity_percentage=80.0, # Default purity
            price_per_unit=waste_price,
            available_from=date.today(),
            available_until=date.today() + timedelta(days=365),
            status=WasteStatus.AVAILABLE,
            description=f"Raw {material_name} generated at KINFRA facility."
        )
        db.add(listing)
        
        count += 1
        
    db.commit()
    db.close()
    print(f"\nSuccessfully ingested {count} KINFRA companies into the database!")

if __name__ == "__main__":
    ingest_kinfra_data()

