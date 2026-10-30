"""Feature engineering partagé entre l'entraînement et l'inférence.

Une seule fonction `build_features` garantit que l'API applique exactement
les mêmes transformations que lors de l'entraînement (pas de training/serving skew).
Adaptez cette liste aux variables de VOTRE modèle de la partie 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Variables brutes attendues en entrée de l'API (noms du jeu Home Credit)
RAW_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
]

NUMERIC_RAW = [f for f in RAW_FEATURES if f not in ("CODE_GENDER", "NAME_EDUCATION_TYPE")]
CATEGORICAL_RAW = ["CODE_GENDER", "NAME_EDUCATION_TYPE"]

EDUCATION_LEVELS = {
    "Lower secondary": 0,
    "Secondary / secondary special": 1,
    "Incomplete higher": 2,
    "Higher education": 3,
    "Academic degree": 4,
}

# Variables effectivement vues par le modèle (ordre figé)
MODEL_FEATURES = NUMERIC_RAW + [
    "GENDER_M",
    "EDUCATION_ORD",
    "CREDIT_INCOME_PERCENT",
    "ANNUITY_INCOME_PERCENT",
    "CREDIT_TERM",
    "DAYS_EMPLOYED_PERCENT",
    "EXT_SOURCE_MEAN",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme les variables brutes en matrice de features du modèle."""
    out = df[NUMERIC_RAW].astype("float64").copy()
    out["GENDER_M"] = (df["CODE_GENDER"] == "M").astype("float64")
    out["EDUCATION_ORD"] = df["NAME_EDUCATION_TYPE"].map(EDUCATION_LEVELS).astype("float64")
    out["CREDIT_INCOME_PERCENT"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"]
    out["ANNUITY_INCOME_PERCENT"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"]
    out["CREDIT_TERM"] = out["AMT_ANNUITY"] / out["AMT_CREDIT"]
    out["DAYS_EMPLOYED_PERCENT"] = out["DAYS_EMPLOYED"] / out["DAYS_BIRTH"]
    out["EXT_SOURCE_MEAN"] = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out[MODEL_FEATURES]


_NUM_IDX = {f: i for i, f in enumerate(NUMERIC_RAW)}


def build_features_fast(records: list[dict], dtype=np.float32) -> np.ndarray:
    """Version NumPy pure de `build_features` pour l'inférence en ligne.

    Optimisation post-déploiement : le profiling (cProfile) a montré que la
    construction des features avec pandas représentait ~99 % du temps
    d'inférence d'une requête unitaire. Cette version donne exactement le même
    résultat (vérifié par tests/test_model.py) sans passer par un DataFrame.
    """
    n = len(records)
    num = np.empty((n, len(NUMERIC_RAW)), dtype=np.float64)
    gender = np.empty(n)
    edu = np.empty(n)
    for i, r in enumerate(records):
        for f, j in _NUM_IDX.items():
            v = r.get(f)
            num[i, j] = np.nan if v is None else v
        gender[i] = 1.0 if r.get("CODE_GENDER") == "M" else 0.0
        edu[i] = EDUCATION_LEVELS.get(r.get("NAME_EDUCATION_TYPE"), np.nan)

    g = lambda f: num[:, _NUM_IDX[f]]  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        ext = num[:, [_NUM_IDX["EXT_SOURCE_1"], _NUM_IDX["EXT_SOURCE_2"], _NUM_IDX["EXT_SOURCE_3"]]]
        cnt = (~np.isnan(ext)).sum(axis=1)
        ext_mean = np.where(cnt > 0, np.nansum(ext, axis=1) / np.maximum(cnt, 1), np.nan)
        derived = np.column_stack([
            gender, edu,
            g("AMT_CREDIT") / g("AMT_INCOME_TOTAL"),
            g("AMT_ANNUITY") / g("AMT_INCOME_TOTAL"),
            g("AMT_ANNUITY") / g("AMT_CREDIT"),
            g("DAYS_EMPLOYED") / g("DAYS_BIRTH"),
            ext_mean,
        ])
    out = np.hstack([num, derived])
    out[np.isinf(out)] = np.nan
    return out.astype(dtype, copy=False)
