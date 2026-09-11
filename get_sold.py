import pandas as pd

# Load datasets
companies = pd.read_csv('datasets/csv/companies.csv')
plants = pd.read_csv('datasets/csv/plants.csv')
materials = pd.read_csv('datasets/csv/materials.csv')
waste = pd.read_csv('datasets/csv/waste_listings.csv')

# Merge
df = waste.merge(materials, on='material_id', how='left')
df = df.merge(plants, on='plant_id', how='left')
df = df.merge(companies, on='company_id', how='left')

# Filter for sold waste (either listing_status == 'Sold' or 'Exchanged')
sold = df[df['listing_status'].isin(['Sold', 'Exchanged'])]

# Select relevant columns
result = sold[['material_name', 'quantity', 'unit', 'company_name', 'plant_name']]

print(f'Total sold items: {len(result)}')
for _, row in result.iterrows():
    print(f"{row['material_name']} ({row['quantity']} {row['unit']}) - Sold by: {row['company_name']} ({row['plant_name']})")

