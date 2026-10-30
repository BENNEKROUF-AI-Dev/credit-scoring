"""Prépare les artefacts du modèle servis par l'API.

Trois sources possibles (par ordre de préférence pour la soutenance) :
  1. Votre modèle MLflow de la partie 1 :
       python -m scripts.prepare_model --mlflow-model-uri "models:/credit_scoring/Production" \
              --data data/application_train.csv
  2. Ré-entraînement sur les vraies données Home Credit :
       python -m scripts.prepare_model --data data/application_train.csv
  3. Données synthétiques (démonstration, aucune donnée requise) :
       python -m scripts.prepare_model

Produit dans model/ : model.joblib, model_meta.json, reference_data.parquet
(référence pour la détection de drift).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from app.features import MODEL_FEATURES, RAW_FEATURES, build_features
from scripts.synthetic import generate_clients

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
COST_FN, COST_FP = 10, 1  # un mauvais payeur accepté coûte 10x un bon client refusé


def load_home_credit(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=RAW_FEATURES + ["TARGET"])
    df = df[df["CODE_GENDER"].isin(["M", "F"])]
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)  # anomalie connue
    df = df.dropna(subset=["AMT_ANNUITY"])
    return df.reset_index(drop=True)


def business_cost(y_true, proba, threshold) -> float:
    pred = (proba >= threshold).astype(int)
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return (COST_FN * fn + COST_FP * fp) / len(y_true)


def best_threshold(y_true, proba) -> tuple[float, float]:
    grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    costs = [business_cost(y_true, proba, t) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="chemin vers application_train.csv (Home Credit)")
    p.add_argument("--mlflow-model-uri", help="URI MLflow du modèle de la partie 1")
    p.add_argument("--n-synthetic", type=int, default=60000)
    p.add_argument("--version", default=None)
    args = p.parse_args()

    if args.data:
        df = load_home_credit(args.data)
        source = f"home-credit:{Path(args.data).name}"
    else:
        df = generate_clients(args.n_synthetic, seed=42, with_target=True)
        source = "synthetic"
    print(f"Données : {len(df)} lignes, taux de défaut {df['TARGET'].mean():.2%} ({source})")

    train, valid = train_test_split(df, test_size=0.2, stratify=df["TARGET"], random_state=42)
    X_train, X_valid = build_features(train), build_features(valid)

    if args.mlflow_model_uri:
        import mlflow.lightgbm  # le modèle doit être un LGBMClassifier loggé avec mlflow
        model = mlflow.lightgbm.load_model(args.mlflow_model_uri)
        source += f" | mlflow:{args.mlflow_model_uri}"
    else:
        model = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               min_child_samples=50, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, class_weight="balanced",
                               random_state=42, verbose=-1)
        model.fit(X_train, train["TARGET"])

    proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(valid["TARGET"], proba)
    threshold, cost = best_threshold(valid["TARGET"].to_numpy(), proba)
    print(f"AUC validation = {auc:.4f} | seuil métier = {threshold} | coût = {cost:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")

    # Référence pour le drift : échantillon des données d'entraînement + scores
    ref = train[RAW_FEATURES].sample(min(5000, len(train)), random_state=0).copy()
    ref["score"] = model.predict_proba(build_features(ref))[:, 1]
    ref.to_parquet(MODEL_DIR / "reference_data.parquet", index=False)

    version = args.version or datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M")
    meta = {
        "version": version,
        "source": source,
        "threshold": threshold,
        "raw_features": RAW_FEATURES,
        "model_features": MODEL_FEATURES,
        "metrics": {"auc_valid": round(auc, 4), "business_cost_valid": round(cost, 4),
                    "cost_fn": COST_FN, "cost_fp": COST_FP},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Artefacts écrits dans {MODEL_DIR}")


if __name__ == "__main__":
    main()
