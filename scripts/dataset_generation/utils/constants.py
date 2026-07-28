from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Dataset Paths
DATASETS_DIR = PROJECT_ROOT / "datasets"

METADATA_DIR = DATASETS_DIR / "metadata"
CSV_DIR = DATASETS_DIR / "csv"
GRAPH_DIR = DATASETS_DIR / "graph"
TRAINING_DIR = DATASETS_DIR / "training"

MASTER_DATASET = DATASETS_DIR / "master_dataset.xlsx"