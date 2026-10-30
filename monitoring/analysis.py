"""Chargement des données de production et calcul des indicateurs de monitoring.
Fonctions partagées par le script d'analyse automatique et le dashboard Streamlit."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "predictions.db"
REFERENCE = ROOT / "model" / "reference_data.parquet"
CATEGORICAL = ["CODE_GENDER", "NAME_EDUCATION_TYPE"]

# Seuils d'alerte (à ajuster avec le métier)
ALERTS = {
    "error_rate": 0.05,         # > 5 % de requêtes en erreur
    "latency_p95_ms": 200.0,    # p95 de latence API
    "score_psi": 0.2,           # PSI > 0.2 = dérive significative
    "drift_share": 0.3,         # > 30 % des variables en dérive
    "refusal_rate_delta": 0.10, # taux de refus +/- 10 pts vs référence
}


def load_logs(db_path: Path | str = DEFAULT_DB) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM predictions ORDER BY id", conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    return df


def expand_inputs(logs: pd.DataFrame) -> pd.DataFrame:
    """Données d'entrée + score des requêtes réussies, au format de la référence."""
    ok = logs[logs["status_code"] == 200].copy()
    inputs = pd.json_normalize(ok["inputs"].map(json.loads).tolist())
    inputs.index = ok.index
    inputs["score"] = ok["score"]
    inputs["timestamp"] = ok["timestamp"]
    return inputs


def load_reference() -> pd.DataFrame:
    return pd.read_parquet(REFERENCE)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index sur des déciles de la référence."""
    reference, current = reference[~np.isnan(reference)], current[~np.isnan(current)]
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / max(len(current), 1)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def operational_metrics(logs: pd.DataFrame) -> dict:
    ok = logs[logs["status_code"] == 200]
    lat = ok["latency_ms"].dropna()
    inf = ok["inference_ms"].dropna()
    return {
        "n_requests": int(len(logs)),
        "n_success": int(len(ok)),
        "error_rate": float((logs["status_code"] != 200).mean()) if len(logs) else 0.0,
        "latency_p50_ms": float(lat.quantile(0.50)) if len(lat) else None,
        "latency_p95_ms": float(lat.quantile(0.95)) if len(lat) else None,
        "latency_p99_ms": float(lat.quantile(0.99)) if len(lat) else None,
        "inference_p50_ms": float(inf.quantile(0.50)) if len(inf) else None,
        "inference_p95_ms": float(inf.quantile(0.95)) if len(inf) else None,
        "refusal_rate": float((ok["decision"] == "REFUSE").mean()) if len(ok) else None,
        "mean_score": float(ok["score"].mean()) if len(ok) else None,
    }


def latency_anomalies(logs: pd.DataFrame, z: float = 4.0) -> pd.DataFrame:
    """Requêtes dont la latence dépasse médiane + z * MAD (robuste aux outliers)."""
    ok = logs[(logs["status_code"] == 200) & logs["latency_ms"].notna()]
    med = ok["latency_ms"].median()
    mad = (ok["latency_ms"] - med).abs().median() * 1.4826 or 1e-9
    return ok[ok["latency_ms"] > med + z * mad]
