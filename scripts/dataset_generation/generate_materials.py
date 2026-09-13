import random
import pandas as pd

from scripts.dataset_generation.utils.constants import (
    CSV_DIR,
    METADATA_DIR
)

from scripts.dataset_generation.utils.json_loader import load_json
plants = pd.read_csv(CSV_DIR / "plants.csv")
companies = pd.read_csv(CSV_DIR / "companies.csv")

materials = load_json(
    METADATA_DIR / "materials.json"
)
material_types = load_json(
    METADATA_DIR / "material_types.json"
)

measurement_units = load_json(
    METADATA_DIR / "measurement_units.json"
)

boolean_values = load_json(
    METADATA_DIR / "boolean_values.json"
)

# Explicit category per material. Deriving it from the first word instead
# ("Steel Scrap" -> "Steel", "PET Bottles" -> "PET") leaves most materials
# with a category the database doesn't recognise, so they all collapse into
# "Other" -- which would make Fly Ash and E-Waste look like related materials
# to the recommender's category matching.
MATERIAL_CATEGORY = {
    "Plastic Scrap": "Plastic",
    "PET Bottles": "Plastic",
    "HDPE Plastic": "Plastic",
    "LDPE Plastic": "Plastic",
    "Paper Waste": "Paper",
    "Cardboard": "Paper",
    "Glass Scrap": "Glass",
    "Metal Scrap": "Metal",
    "Steel Scrap": "Metal",
    "Aluminium Scrap": "Metal",
    "Copper Scrap": "Metal",
    "Rubber Waste": "Rubber",
    "Textile Waste": "Textile",
    "Cotton Fiber": "Textile",
    "Wood Chips": "Wood",
    "Food Waste": "Organic",
    "Organic Compost": "Organic",
    "Fly Ash": "Construction",
    "Construction Debris": "Construction",
    "E-Waste": "Electronic",
}

# Waste each industry plausibly produces. Picking materials at random instead
# produces nonsense like a food processor selling aerospace metal offcuts.
SECTOR_MATERIALS = {
    "Manufacturing": ["Metal Scrap", "Steel Scrap", "Plastic Scrap", "Wood Chips", "Cardboard"],
    "Food Processing": ["Food Waste", "Organic Compost", "Cardboard", "Plastic Scrap", "Glass Scrap"],
    "Textiles": ["Textile Waste", "Cotton Fiber", "Plastic Scrap", "Cardboard"],
    "Chemical": ["Plastic Scrap", "Glass Scrap", "Fly Ash", "Metal Scrap"],
    "Pharmaceutical": ["Glass Scrap", "Cardboard", "Plastic Scrap", "Paper Waste"],
    "Electronics": ["E-Waste", "Copper Scrap", "Plastic Scrap", "Metal Scrap"],
    "Plastic": ["Plastic Scrap", "PET Bottles", "HDPE Plastic", "LDPE Plastic"],
    "Paper": ["Paper Waste", "Cardboard", "Wood Chips"],
    "Automotive": ["Metal Scrap", "Steel Scrap", "Rubber Waste", "Aluminium Scrap", "Copper Scrap"],
    "Construction Materials": ["Construction Debris", "Fly Ash", "Steel Scrap", "Wood Chips"],
    "Rubber": ["Rubber Waste", "Plastic Scrap", "Textile Waste"],
    "Metal Processing": ["Metal Scrap", "Steel Scrap", "Aluminium Scrap", "Copper Scrap"],
    "Renewable Energy": ["Fly Ash", "E-Waste", "Metal Scrap", "Copper Scrap"],
    "Agriculture": ["Organic Compost", "Food Waste", "Wood Chips", "Cotton Fiber"],
    "Waste Management": ["Plastic Scrap", "Paper Waste", "Glass Scrap", "Metal Scrap", "E-Waste"],
}

company_sector = dict(zip(companies["company_id"], companies["industry_sector"]))

material_records = []

material_counter = 1

for _, plant in plants.iterrows():

    number_of_materials = random.randint(2, 5)

    # Draw from what this company's industry actually produces, falling back
    # to the full list for any sector without a mapping.
    sector = str(company_sector.get(plant["company_id"], ""))
    candidate_materials = SECTOR_MATERIALS.get(sector) or materials
    number_of_materials = min(number_of_materials, len(candidate_materials))
    selected_materials = random.sample(candidate_materials, number_of_materials)

    for material in selected_materials:

        material_records.append({

            "material_id": f"M{material_counter:03d}",

            "plant_id": plant["plant_id"],

            "company_id": plant["company_id"],

            "material_name": material,

            "material_category": MATERIAL_CATEGORY.get(material, "Other"),
            "material_type": random.choice(material_types),

            "unit": random.choice(measurement_units),

            "recyclable": random.choice(boolean_values),

            "hazardous": random.choice(boolean_values)

        })

        material_counter += 1
df = pd.DataFrame(material_records)

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())
print("\n========== DATA VALIDATION ==========")

print(f"Duplicate Material IDs : {df['material_id'].duplicated().sum()}")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nMaterials per Plant:")
print(df.groupby("plant_id").size())
# -------------------------------
# Export Materials Dataset
# -------------------------------

output_file = CSV_DIR / "materials.csv"

df.to_csv(output_file, index=False)

print(f"\nMaterials dataset exported successfully to:\n{output_file}")
