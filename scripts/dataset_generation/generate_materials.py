import random
import pandas as pd

from scripts.dataset_generation.utils.constants import (
    CSV_DIR,
    METADATA_DIR
)

from scripts.dataset_generation.utils.json_loader import load_json
plants = pd.read_csv(CSV_DIR / "plants.csv")

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
material_records = []

material_counter = 1

for _, plant in plants.iterrows():

    number_of_materials = random.randint(2, 5)

    selected_materials = random.sample(materials, number_of_materials)

    for material in selected_materials:

        material_records.append({

            "material_id": f"M{material_counter:03d}",

            "plant_id": plant["plant_id"],

            "company_id": plant["company_id"],

            "material_name": material,

            "material_category": material.split()[0],
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
