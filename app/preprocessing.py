"""Préparation des données : du dossier client à la matrice du modèle.

Reproduit EXACTEMENT les transformations des notebooks du projet précédent :

Notebook 1 (variables dérivées, recalculées ici par l'API) :
  - DAYS_EMPLOYED == 365243 (sentinelle « sans emploi »)  -> DAYS_EMPLOYED_ANOM = 1, valeur -> NaN
  - OWN_CAR_AGE manquant (pas de voiture)                  -> -1
  - EXT_SOURCE_k manquant                                  -> EXT_SOURCE_k_MISSING = 1
  - BUREAU_COUNT manquant (pas d'historique bureau)        -> BUREAU_MISSING = 1, BUREAU_COUNT = 0
  - CC_COUNT manquant (pas de carte de crédit)             -> CC_MISSING = 1

Notebook 2 (encodage) :
  - pd.get_dummies sur les variables texte, puis noms_propres() sur les noms de colonnes.
  La correspondance exacte (nom d'origine -> colonne du modèle) est calculée une fois par
  scripts/export_model.py et stockée dans model/model_meta.json.

Deux implémentations :
  - transform_pandas : version « de référence », lisible, identique aux notebooks
  - transform        : version NumPy optimisée pour l'API (même résultat, testée)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

ANOMALY_DAYS_EMPLOYED = 365243

# Variables calculées par l'API : l'appelant ne doit pas les envoyer
DERIVED = ["DAYS_EMPLOYED_ANOM", "EXT_SOURCE_1_MISSING", "EXT_SOURCE_2_MISSING",
           "EXT_SOURCE_3_MISSING", "BUREAU_MISSING", "CC_MISSING"]


def noms_propres(columns) -> list[str]:
    """Copie de la fonction du notebook 2 (même algorithme, mêmes doublons)."""
    vus, sortie = {}, []
    for n in columns:
        p = re.sub(r'[^A-Za-z0-9_]+', '_', str(n)).strip('_') or 'col'
        if p[0].isdigit():
            p = 'c_' + p
        if p in vus:
            vus[p] += 1
            p = f"{p}_{vus[p]}"
        else:
            vus[p] = 0
        sortie.append(p)
    return sortie


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def derive(record: dict, raw_columns: set[str]) -> dict:
    """Recalcule les variables dérivées du notebook 1 pour UN dossier."""
    r = dict(record)
    if "DAYS_EMPLOYED_ANOM" in raw_columns:
        anom = r.get("DAYS_EMPLOYED") == ANOMALY_DAYS_EMPLOYED
        r["DAYS_EMPLOYED_ANOM"] = int(anom)
        if anom:
            r["DAYS_EMPLOYED"] = None
    if "OWN_CAR_AGE" in raw_columns and _missing(r.get("OWN_CAR_AGE")):
        r["OWN_CAR_AGE"] = -1
    for k in (1, 2, 3):
        if f"EXT_SOURCE_{k}_MISSING" in raw_columns:
            r[f"EXT_SOURCE_{k}_MISSING"] = int(_missing(r.get(f"EXT_SOURCE_{k}")))
    if "BUREAU_MISSING" in raw_columns:
        miss = _missing(r.get("BUREAU_COUNT"))
        r["BUREAU_MISSING"] = int(miss)
        if miss:
            r["BUREAU_COUNT"] = 0
    if "CC_MISSING" in raw_columns:
        r["CC_MISSING"] = int(_missing(r.get("CC_COUNT")))
    return r


def underive(row: dict) -> dict:
    """Inverse de `derive` : remet un dossier du format notebook 1 au format « saisie ».
    Utilisé uniquement pour préparer des exemples (export, tests)."""
    r = {k: (None if _missing(v) else v) for k, v in row.items()}
    if r.get("DAYS_EMPLOYED_ANOM") == 1:
        r["DAYS_EMPLOYED"] = ANOMALY_DAYS_EMPLOYED
    if r.get("OWN_CAR_AGE") == -1:
        r["OWN_CAR_AGE"] = None
    if r.get("BUREAU_MISSING") == 1:
        r["BUREAU_COUNT"] = None
    for c in DERIVED:
        r.pop(c, None)
    return r


class Preprocessor:
    def __init__(self, meta: dict):
        self.model_columns: list[str] = meta["model_columns"]
        self.numeric: list[str] = meta["raw_numeric"]
        self.categorical: dict[str, list[str]] = meta["raw_categorical"]
        self.rename_map: dict[str, str] = meta["rename_map"]  # nom après get_dummies -> colonne modèle
        self.raw_columns = set(self.numeric) | set(self.categorical)
        idx = {c: i for i, c in enumerate(self.model_columns)}

        self._num_idx = [(c, idx[self.rename_map[c]]) for c in self.numeric]
        self._cat_idx = {
            col: {v: idx[self.rename_map[f"{col}_{v}"]] for v in values}
            for col, values in self.categorical.items()
        }
        self._dummy_positions = np.array(
            sorted(i for m in self._cat_idx.values() for i in m.values()), dtype=int)

    @property
    def input_columns(self) -> list[str]:
        """Champs acceptés en entrée de l'API (variables dérivées exclues)."""
        return [c for c in self.numeric + list(self.categorical) if c not in DERIVED]

    # ------------------------------------------------------------- version NumPy
    def transform(self, records: list[dict], dtype=np.float32) -> np.ndarray:
        n, m = len(records), len(self.model_columns)
        out = np.full((n, m), np.nan, dtype=np.float64)
        out[:, self._dummy_positions] = 0.0
        for i, rec in enumerate(records):
            r = derive(rec, self.raw_columns)
            for col, j in self._num_idx:
                v = r.get(col)
                if v is not None:
                    out[i, j] = v
            for col, mapping in self._cat_idx.items():
                j = mapping.get(r.get(col))
                if j is not None:
                    out[i, j] = 1.0
        return out.astype(dtype, copy=False)

    # ------------------------------------------------------------- version pandas
    def transform_pandas(self, records: list[dict]) -> pd.DataFrame:
        """Version de référence, calquée sur les notebooks (lente sur 1 ligne)."""
        df = pd.DataFrame([derive(r, self.raw_columns) for r in records])
        for c in self.numeric:
            if c not in df:
                df[c] = np.nan
        for c in self.categorical:
            if c not in df:
                df[c] = None
        df = df[self.numeric + list(self.categorical)]
        df[self.numeric] = df[self.numeric].astype("float64")
        X = pd.get_dummies(df, columns=list(self.categorical), dtype=int)
        X = X.rename(columns=self.rename_map)
        X = X.reindex(columns=self.model_columns)
        dummy_cols = [self.model_columns[i] for i in self._dummy_positions]
        X[dummy_cols] = X[dummy_cols].fillna(0)
        return X
