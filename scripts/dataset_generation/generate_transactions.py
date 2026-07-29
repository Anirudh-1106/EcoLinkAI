import random
from datetime import datetime, timedelta

import pandas as pd

from scripts.dataset_generation.utils.constants import CSV_DIR, METADATA_DIR
from scripts.dataset_generation.utils.json_loader import load_json
requests_df = pd.read_csv(CSV_DIR / "exchange_requests.csv")
waste_df = pd.read_csv(CSV_DIR / "waste_listings.csv")
accepted_requests = requests_df[
    requests_df["request_status"] == "Accepted"
].copy()
delivery_statuses = load_json(
    METADATA_DIR / "delivery_statuses.json"
)

payment_statuses = load_json(
    METADATA_DIR / "payment_statuses.json"
)
def generate_transaction_id(index):
    return f"TRX{index:05d}"
transactions = []

transaction_counter = 1

for _, request in accepted_requests.iterrows():

    waste = waste_df[
        waste_df["waste_id"] == request["waste_id"]
    ].iloc[0]

    transaction_date = pd.to_datetime(
        request["expected_pickup_date"]
    ) + timedelta(
        days=random.randint(0, 5)
    )

    final_quantity = round(

        request["requested_quantity"] *
        random.uniform(0.95, 1.00),

        2
    )

    final_price = round(

        request["offered_price_per_unit"] *
        random.uniform(0.98, 1.03),

        2
    )

    transport_cost = round(

        final_quantity *
        random.uniform(8, 25),

        2
    )

    carbon_saving = round(

        final_quantity *
        random.uniform(0.4, 2.5),

        2
    )

    transactions.append({

        "transaction_id":
            generate_transaction_id(transaction_counter),

        "request_id":
            request["request_id"],

        "waste_id":
            request["waste_id"],

        "buyer_company_id":
            request["requester_company_id"],

        "seller_company_id":
            request["supplier_company_id"],

        "transaction_date":
            transaction_date.date(),

        "final_quantity":
            final_quantity,

        "final_price_per_unit":
            final_price,

        "transport_cost":
            transport_cost,

        "carbon_saving_kg":
            carbon_saving,

        "delivery_status":
            random.choices(
                delivery_statuses,
                weights=[15, 20, 65],
                k=1
            )[0],

        "payment_status":
            random.choices(
                payment_statuses,
                weights=[15, 75, 10],
                k=1
            )[0]

    })

    transaction_counter += 1
df = pd.DataFrame(transactions)
print("\n========== Transactions Dataset ==========")

print(f"\nDataset Shape: {df.shape}")

print("\nDuplicate Transaction IDs:")
print(df["transaction_id"].duplicated().sum())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDelivery Status:")
print(df["delivery_status"].value_counts())

print("\nPayment Status:")
print(df["payment_status"].value_counts())
output_file = CSV_DIR / "transactions.csv"

df.to_csv(output_file, index=False)

print(f"\nDataset exported successfully to:\n{output_file}")