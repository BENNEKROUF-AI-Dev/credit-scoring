"""Configuration centralisée (surchargeable par variables d'environnement)."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "model"))
MODEL_PATH = MODEL_DIR / "model.joblib"
ONNX_MODEL_PATH = MODEL_DIR / "model.onnx"
MODEL_META_PATH = MODEL_DIR / "model_meta.json"
REFERENCE_DATA_PATH = MODEL_DIR / "reference_data.parquet"

# "onnx" (optimisé, par défaut si le fichier existe) ou "lightgbm" (natif)
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "onnx")
# Feature engineering NumPy (optimisé) ; 0 = version pandas v1
FAST_FEATURES = os.getenv("FAST_FEATURES", "1") == "1"

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
PREDICTIONS_DB = Path(os.getenv("PREDICTIONS_DB", DATA_DIR / "predictions.db"))
PREDICTIONS_LOG = Path(os.getenv("PREDICTIONS_LOG", DATA_DIR / "predictions.jsonl"))

ENABLE_UI = os.getenv("ENABLE_UI", "1") == "1"
ENABLE_PREDICTION_LOGGING = os.getenv("ENABLE_PREDICTION_LOGGING", "1") == "1"
