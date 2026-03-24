import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "datasets" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "datasets" / "processed"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"

FOOD101_DIR = RAW_DATA_DIR / "food101"
RECIPE1M_DIR = RAW_DATA_DIR / "recipe1m"
USDA_DIR = RAW_DATA_DIR / "usda"
PORTION_DIR = RAW_DATA_DIR / "portion"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10