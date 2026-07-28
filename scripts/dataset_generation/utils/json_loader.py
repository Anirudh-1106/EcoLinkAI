import json
from pathlib import Path


def load_json(filepath: Path):
    """
    Load and return a JSON file.
    """

    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)