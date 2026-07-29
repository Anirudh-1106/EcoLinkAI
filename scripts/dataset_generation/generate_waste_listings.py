import random
from datetime import datetime, timedelta

import pandas as pd

from scripts.dataset_generation.utils.constants import CSV_DIR, METADATA_DIR
from scripts.dataset_generation.utils.json_loader import load_json
companies_df = pd.read_csv(CSV_DIR / "companies.csv")
plants_df = pd.read_csv(CSV_DIR / "plants.csv")
materials_df = pd.read_csv(CSV_DIR / "materials.csv")
storage_conditions = load_json(
    METADATA_DIR / "storage_conditions.json"
)

urgency_levels = load_json(
    METADATA_DIR / "urgency_levels.json"
)

listing_statuses = load_json(
    METADATA_DIR / "listing_status.json"
)
measurement_units = load_json(
    METADATA_DIR / "measurement_units.json"
)
def generate_waste_id(index):
    return f"WST{index:05d}"
waste_data = []
counter = 1

for _, material in materials_df.iterrows():

    num_listings = random.randint(2, 3)

    for _ in range(num_listings):

        availability = datetime.now() + timedelta(
            days=random.randint(0, 30)
        )

        expiry = availability + timedelta(
            days=random.randint(15, 60)
        )

        waste_data.append({

            "waste_id": generate_waste_id(counter),

            "material_id": material["material_id"],

            "plant_id": material["plant_id"],

            "company_id": material["company_id"],

            "quantity": round(random.uniform(100, 5000), 2),

            "unit": random.choice(measurement_units),

            "purity_percentage": round(
                random.uniform(60, 99),
                2
            ),

            "availability_date": availability.date(),

            "expiry_date": expiry.date(),

            "storage_condition": random.choice(storage_conditions),

            "urgency": random.choice(urgency_levels),

            "asking_price_per_unit": round(
                random.uniform(5, 500),
                2
            ),

            "listing_status": random.choice(listing_statuses),

            "carbon_saving_score": round(
                random.uniform(10, 100),
                2
            ),

            "created_at": datetime.now()
        })

        counter += 1
df = pd.DataFrame(waste_data)
print("\n========== Waste Listings Validation ==========\n")

print("Dataset Shape:")
print(df.shape)

print("\nDuplicate Waste IDs:")
print(df["waste_id"].duplicated().sum())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nListings per Material:")
print(df.groupby("material_id").size())

print("\n==============================================\n")
output_file = CSV_DIR / "waste_listings.csv"

df.to_csv(output_file, index=False)

print(f"Dataset successfully saved to:\n{output_file}")