from utils.constants import METADATA_DIR
from utils.json_loader import load_json

company_schema = load_json(METADATA_DIR / "companies_schema.json")

print(company_schema)