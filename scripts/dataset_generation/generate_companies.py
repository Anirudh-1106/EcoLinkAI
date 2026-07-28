import random
import pandas as pd

from scripts.dataset_generation.utils.constants import (
    CSV_DIR,
    METADATA_DIR,
)

from scripts.dataset_generation.utils.json_loader import load_json

from scripts.dataset_generation.utils.generators import *

from scripts.dataset_generation.utils.validators import *
industry_sectors = load_json(
    METADATA_DIR / "industry_sectors.json"
)

districts = load_json(
    METADATA_DIR / "kerala_districts.json"
)

company_sizes = load_json(
    METADATA_DIR / "company_sizes.json"
)

verification_status = load_json(
    METADATA_DIR / "verification_status.json"
)
print("Loading industry_sectors...")
industry_sectors = load_json(METADATA_DIR / "industry_sectors.json")

print("Loading districts...")
districts = load_json(METADATA_DIR / "kerala_districts.json")

print("Loading company_sizes...")
company_sizes = load_json(METADATA_DIR / "company_sizes.json")

print("Loading verification_status...")
verification_status = load_json(METADATA_DIR / "verification_status.json")
companies = []
for i in range(1, 21):

    company_name = fake.company()

    company = {

        "company_id": generate_company_id(i),

        "company_name": company_name,

        "industry_sector": random.choice(industry_sectors),

        "industry_subsector": "General",

        "registration_number": generate_registration_number(),

        "email": generate_email(company_name),

        "phone": generate_phone(),

        "website": generate_website(company_name),

        "head_office": fake.city(),

        "district": random.choice(districts),

        "state": "Kerala",

        "country": "India",

        "trust_score": generate_trust_score(),

        "verification_status": random.choice(verification_status),

        "established_year": generate_established_year(),

        "company_size": random.choice(company_sizes)

    }

    companies.append(company)
    df = pd.DataFrame(companies)

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())
print("\nChecking for duplicate Company IDs...")
print(df["company_id"].duplicated().sum())

print("\nChecking for duplicate Registration Numbers...")
print(df["registration_number"].duplicated().sum())

print("\nChecking for duplicate Emails...")
print(df["email"].duplicated().sum())

print("\nChecking for Missing Values...")
print(df.isnull().sum())