import random
import pandas as pd

from scripts.dataset_generation.utils.constants import (
    CSV_DIR,
    METADATA_DIR
)

from scripts.dataset_generation.utils.json_loader import load_json
from scripts.dataset_generation.utils.generators import *
companies = pd.read_csv(CSV_DIR / "companies.csv")

plant_types = load_json(
    METADATA_DIR / "plant_types.json"
)
plants = []

plant_counter = 1

for _, company in companies.iterrows():

    number_of_plants = random.randint(1, 3)

    for _ in range(number_of_plants):

        plant = {

            "plant_id": f"P{plant_counter:03d}",

            "company_id": company["company_id"],

            "plant_name": f"{company['company_name']} Plant {plant_counter}",

            "plant_type": random.choice(plant_types),

            "district": company["district"],

            "state": company["state"],

            "country": company["country"]

        }

        plants.append(plant)

        plant_counter += 1
df = pd.DataFrame(plants)

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())
print("\n========== DATA VALIDATION ==========")

print(f"Duplicate Plant IDs : {df['plant_id'].duplicated().sum()}")

print(f"Missing Values      :")
print(df.isnull().sum())

print("\nPlants per Company:")

print(df.groupby("company_id").size())
# -------------------------------
# Export Plants Dataset
# -------------------------------

output_file = CSV_DIR / "plants.csv"

df.to_csv(output_file, index=False)

print(f"\nPlants dataset exported successfully to:\n{output_file}")