import re

EMAIL_REGEX = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"


def is_valid_email(email: str) -> bool:
    """
    Validate email format.
    """
    return re.match(EMAIL_REGEX, email) is not None


def is_valid_trust_score(score: int) -> bool:
    """
    Trust score must be between 60 and 100.
    """
    return 60 <= score <= 100


def is_valid_established_year(year: int) -> bool:
    """
    Companies should have a realistic establishment year.
    """
    return 1900 <= year <= 2026


def is_not_empty(value) -> bool:
    """
    Check for non-empty values.
    """
    return value is not None and str(value).strip() != ""