"""Configuration centralisée (surchargeable par variables d'environnement)."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "model"))
MODEL_META_PATH = MODEL_DIR / "model_meta.json"
SAMPLE_CLIENTS_PATH = MODEL_DIR / "sample_clients.parquet"
REFERENCE_DATA_PATH = MODEL_DIR / "reference_data.parquet"

# "onnx" (optimisé, par défaut) ou "lightgbm" (booster natif model.txt)
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "onnx")
# Préparation des données en NumPy (optimisé) ; 0 = version pandas v1
FAST_FEATURES = os.getenv("FAST_FEATURES", "1") == "1"

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
PREDICTIONS_DB = Path(os.getenv("PREDICTIONS_DB", DATA_DIR / "predictions.db"))
PREDICTIONS_LOG = Path(os.getenv("PREDICTIONS_LOG", DATA_DIR / "predictions.jsonl"))
# Base PostgreSQL externe (ex. NeonDB). Vide = SQLite local (PREDICTIONS_DB).
DATABASE_URL = os.getenv("DATABASE_URL") or None

ENABLE_UI = os.getenv("ENABLE_UI", "1") == "1"
ENABLE_PREDICTION_LOGGING = os.getenv("ENABLE_PREDICTION_LOGGING", "1") == "1"
