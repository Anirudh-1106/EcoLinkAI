import random
from faker import Faker

fake = Faker("en_IN")


def generate_company_id(index: int) -> str:
    """
    Generate Company IDs like C001, C002...
    """
    return f"C{index:03d}"


def generate_registration_number() -> str:
    """
    Generate a pseudo company registration number.
    """
    return f"CIN{random.randint(100000000, 999999999)}"


def generate_phone() -> str:
    """
    Generate an Indian phone number.
    """
    return fake.phone_number()


def generate_email(company_name: str) -> str:
    """
    Generate email from company name.
    """
    username = (
        company_name.lower()
        .replace("&", "")
        .replace(",", "")
        .replace(".", "")
        .replace(" ", "")
    )

    domains = [
        "gmail.com",
        "company.com",
        "industry.in",
        "ecolink.ai"
    ]

    return f"{username}@{random.choice(domains)}"


def generate_website(company_name: str) -> str:
    """
    Generate website URL.
    """
    domain = (
        company_name.lower()
        .replace("&", "")
        .replace(",", "")
        .replace(".", "")
        .replace(" ", "")
    )

    return f"https://www.{domain}.com"


def generate_trust_score():
    return random.randint(60, 100)


def generate_established_year():
    return random.randint(1990, 2024)