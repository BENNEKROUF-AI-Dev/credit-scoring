"""Générateur de clients synthétiques au format Home Credit.

Sert à (1) entraîner un modèle de remplacement si vous n'avez pas encore
branché celui de la partie 1, et (2) simuler du trafic de production, avec
ou sans dérive des données (drift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EDU = ["Lower secondary", "Secondary / secondary special", "Incomplete higher",
       "Higher education", "Academic degree"]
EDU_P = [0.012, 0.710, 0.033, 0.243, 0.002]


def generate_clients(n: int, seed: int = 0, drift: float = 0.0, with_target: bool = False) -> pd.DataFrame:
    """drift ∈ [0, 1] : 0 = même distribution que l'entraînement ;
    1 = clientèle nettement plus jeune, plus endettée, scores externes plus bas."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(21, 69, n) - drift * rng.uniform(0, 12, n)
    age = np.clip(age, 18.5, 99)
    years_emp = np.minimum(rng.exponential(6, n), age - 18)
    days_employed = -(years_emp * 365.25).astype(int)
    days_employed = np.where(rng.random(n) < 0.18, np.nan, days_employed)

    income = np.exp(rng.normal(12.0 - 0.15 * drift, 0.5, n))
    credit = income * np.exp(rng.normal(np.log(3.5) + 0.35 * drift, 0.5, n))
    annuity = credit * rng.uniform(0.03, 0.08, n)
    goods = np.where(rng.random(n) < 0.01, np.nan, credit * rng.uniform(0.8, 1.0, n))

    ext1 = np.where(rng.random(n) < 0.56, np.nan, rng.beta(5, 4.5, n))
    ext2 = np.where(rng.random(n) < 0.002, np.nan, rng.beta(5, 3.5, n))
    ext3 = np.where(rng.random(n) < 0.20, np.nan, rng.beta(5 - 1.5 * drift, 4 + 1.0 * drift, n))

    df = pd.DataFrame({
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "AMT_INCOME_TOTAL": income.round(0),
        "AMT_CREDIT": credit.round(0),
        "AMT_ANNUITY": annuity.round(0),
        "AMT_GOODS_PRICE": np.round(goods, 0),
        "DAYS_BIRTH": -(age * 365.25).astype(int),
        "DAYS_EMPLOYED": days_employed,
        "CNT_CHILDREN": np.minimum(rng.poisson(0.4 + 0.3 * drift, n), 10),
        "CODE_GENDER": rng.choice(["F", "M"], n, p=[0.66, 0.34]),
        "NAME_EDUCATION_TYPE": rng.choice(EDU, n, p=EDU_P),
    })

    if with_target:
        def c(x):
            return np.nan_to_num(x - 0.5, nan=0.0)
        z = (-3.25 - 4.0 * c(ext2) - 3.5 * c(ext3) - 2.0 * c(ext1)
             + 0.35 * (df["CODE_GENDER"] == "M") - 0.02 * (age - 45)
             + 6.0 * (annuity / income - 0.10)
             - 0.4 * df["NAME_EDUCATION_TYPE"].isin(EDU[3:])
             + rng.normal(0, 0.6, n))
        df["TARGET"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df
