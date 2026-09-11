import csv

def main():
    # Load materials
    materials = {}
    with open('datasets/csv/materials.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            materials[row['material_id']] = row['material_name']

    # Load companies
    companies = {}
    with open('datasets/csv/companies.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            companies[row['company_id']] = row['company_name']
            
    # Load plants
    plants = {}
    with open('datasets/csv/plants.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            plants[row['plant_id']] = {
                'plant_name': row['plant_name'],
                'company_id': row['company_id']
            }

    sold_items = []
    
    # Check transactions to see what was sold
    with open('datasets/csv/waste_listings.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get('listing_status')
            if status in ['Sold', 'Exchanged']:
                mat_name = materials.get(row['material_id'], 'Unknown Material')
                p_info = plants.get(row['plant_id'], {})
                comp_id = p_info.get('company_id') or row.get('company_id')
                comp_name = companies.get(comp_id, 'Unknown Company')
                
                sold_items.append({
                    'material': mat_name,
                    'quantity': f"{row['quantity']} {row['unit']}",
                    'company': comp_name,
                    'price': row.get('asking_price_per_unit', 'N/A')
                })
                
    # Format markdown
    md = "# Sold Waste Listings\n\n"
    md += "| Material | Quantity | Selling Company | Asking Price/Unit |\n"
    md += "| --- | --- | --- | --- |\n"
    for item in sold_items:
        md += f"| {item['material']} | {item['quantity']} | **{item['company']}** | ₹{item['price']} |\n"
        
    with open('sold_waste.md', 'w', encoding='utf-8') as f:
        f.write(md)
        
    print(f"Generated markdown with {len(sold_items)} sold items.")

if __name__ == '__main__':
    main()
