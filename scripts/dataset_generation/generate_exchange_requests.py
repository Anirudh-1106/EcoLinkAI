import random
from datetime import datetime, timedelta

import pandas as pd

from scripts.dataset_generation.utils.constants import CSV_DIR, METADATA_DIR
from scripts.dataset_generation.utils.json_loader import load_json
companies_df = pd.read_csv(CSV_DIR / "companies.csv")
waste_df = pd.read_csv(CSV_DIR / "waste_listings.csv")
request_statuses = load_json(METADATA_DIR / "request_statuses.json")
transport_modes = load_json(METADATA_DIR / "transportation_modes.json")
priorities = load_json(METADATA_DIR / "request_priorities.json")
remarks_list = load_json(METADATA_DIR / "request_remarks.json")
def generate_request_id(index):
    return f"REQ{index:05d}"
exchange_requests = []
request_counter = 1
for _, waste in waste_df.iterrows():

    supplier_company = waste["company_id"]

    asking_price = waste["asking_price_per_unit"]

    available_quantity = waste["quantity"]

    # Each listing receives 1–4 requests
    num_requests = random.randint(1, 4)

    for _ in range(num_requests):

        # Requester cannot be supplier
        requester = random.choice(
            companies_df[
                companies_df["company_id"] != supplier_company
            ]["company_id"].tolist()
        )

        requested_quantity = round(
            random.uniform(
                available_quantity * 0.2,
                available_quantity
            ),
            2
        )

        request_date = datetime.now() - timedelta(
            days=random.randint(1, 120)
        )

        pickup_date = request_date + timedelta(
            days=random.randint(2, 20)
        )

        offered_price = round(
            asking_price * random.uniform(0.85, 1.15),
            2
        )

        status = random.choices(
            request_statuses,
            weights=[35, 40, 15, 10],
            k=1
        )[0]

        exchange_requests.append({

            "request_id": generate_request_id(request_counter),

            "waste_id": waste["waste_id"],

            "requester_company_id": requester,

            "supplier_company_id": supplier_company,

            "requested_quantity": requested_quantity,

            "request_date": request_date.date(),

            "expected_pickup_date": pickup_date.date(),

            "offered_price_per_unit": offered_price,

            "request_status": status,

            "transportation_mode": random.choices(
    transport_modes,
    weights=[70, 20, 10],
    k=1
)[0],

            "priority": random.choice(
                priorities
            ),

            "remarks": random.choice(
                remarks_list
            )

        })

        request_counter += 1
df = pd.DataFrame(exchange_requests)
print("\n========== Exchange Requests Dataset ==========")

print(f"\nDataset Shape: {df.shape}")

print("\nDuplicate Request IDs:")
print(df["request_id"].duplicated().sum())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nRequest Status Distribution:")
print(df["request_status"].value_counts())

print("\nTransportation Modes:")
print(df["transportation_mode"].value_counts())
output_file = CSV_DIR / "exchange_requests.csv"

df.to_csv(output_file, index=False)

print(f"\nDataset exported successfully to:\n{output_file}")
