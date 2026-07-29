import random
from datetime import timedelta

import pandas as pd

from scripts.dataset_generation.utils.constants import CSV_DIR, METADATA_DIR
from scripts.dataset_generation.utils.json_loader import load_json
transactions_df = pd.read_csv(CSV_DIR / "transactions.csv")
review_comments = load_json(
    METADATA_DIR / "review_comments.json"
)

recommendation_options = load_json(
    METADATA_DIR / "recommendation_options.json"
)
def generate_review_id(index):
    return f"REV{index:05d}"
reviews = []

review_counter = 1

for _, transaction in transactions_df.iterrows():

    transaction_date = pd.to_datetime(
        transaction["transaction_date"]
    )

    review_date = transaction_date + timedelta(
        days=random.randint(3, 20)
    )

    rating = random.choices(
        [5, 4, 3, 2, 1],
        weights=[35, 35, 15, 10, 5],
        k=1
    )[0]

    would_recommend = (
        "Yes"
        if rating >= 4
        else random.choices(
            recommendation_options,
            weights=[20, 80],
            k=1
        )[0]
    )

    reviews.append({

        "review_id":
            generate_review_id(review_counter),

        "transaction_id":
            transaction["transaction_id"],

        "buyer_company_id":
            transaction["buyer_company_id"],

        "seller_company_id":
            transaction["seller_company_id"],

        "rating":
            rating,

        "review_comment":
            random.choice(review_comments),

        "review_date":
            review_date.date(),

        "would_recommend":
            would_recommend

    })

    review_counter += 1
df = pd.DataFrame(reviews)
print("\n========== Reviews Dataset ==========")

print(f"\nDataset Shape: {df.shape}")

print("\nDuplicate Review IDs:")
print(df["review_id"].duplicated().sum())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nRating Distribution:")
print(df["rating"].value_counts().sort_index())

print("\nRecommendation Distribution:")
print(df["would_recommend"].value_counts())
output_file = CSV_DIR / "reviews.csv"

df.to_csv(output_file, index=False)

print(f"\nDataset exported successfully to:\n{output_file}")
