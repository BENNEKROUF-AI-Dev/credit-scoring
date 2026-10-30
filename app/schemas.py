"""Validation des entrées de l'API.

Le schéma est construit automatiquement à partir de model/model_meta.json : chaque
variable du dossier client (format du notebook 1) devient un champ, avec :
- variables texte    -> valeurs autorisées = catégories vues à l'entraînement
- variables numériques -> nombre fini (texte, NaN, infini refusés)
- variables clés     -> règles métier (plages) ci-dessous ; certaines obligatoires
- tout champ inconnu -> refusé (faute de frappe, identifiant, injection)
- variables dérivées (DAYS_EMPLOYED_ANOM, *_MISSING…) -> calculées par l'API, refusées en entrée

Tout manquement renvoie une erreur 422 détaillée.
"""
from __future__ import annotations

import json
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from app import config
from app.preprocessing import ANOMALY_DAYS_EMPLOYED, Preprocessor

# Règles métier : (contraintes, obligatoire ?). Ignorées si la variable n'existe pas.
RULES: dict[str, tuple[dict, bool]] = {
    "AMT_INCOME_TOTAL": ({"gt": 0, "le": 1e9}, True),
    "AMT_CREDIT": ({"gt": 0, "le": 1e9}, True),
    "DAYS_BIRTH": ({"ge": -36500, "le": -6570}, True),   # entre 18 et 100 ans
    "AMT_ANNUITY": ({"gt": 0, "le": 1e8}, False),
    "AMT_GOODS_PRICE": ({"gt": 0, "le": 1e9}, False),
    "EXT_SOURCE_1": ({"ge": 0, "le": 1}, False),
    "EXT_SOURCE_2": ({"ge": 0, "le": 1}, False),
    "EXT_SOURCE_3": ({"ge": 0, "le": 1}, False),
    "CNT_CHILDREN": ({"ge": 0, "le": 20}, False),
    "CNT_FAM_MEMBERS": ({"ge": 1, "le": 25}, False),
    "OWN_CAR_AGE": ({"ge": 0, "le": 100}, False),
    "DAYS_EMPLOYED": ({"ge": -36500, "le": ANOMALY_DAYS_EMPLOYED}, False),
    "DAYS_REGISTRATION": ({"ge": -36500, "le": 0}, False),
    "DAYS_ID_PUBLISH": ({"ge": -36500, "le": 0}, False),
}
REQUIRED_CATEGORICAL = ["CODE_GENDER"]


class _ClientBase(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=False)

    @model_validator(mode="after")
    def check_consistency(self):
        d = self.model_dump(by_alias=True)
        annuity, credit = d.get("AMT_ANNUITY"), d.get("AMT_CREDIT")
        if annuity is not None and credit is not None and annuity > credit:
            raise ValueError("AMT_ANNUITY ne peut pas dépasser AMT_CREDIT")
        emp, birth = d.get("DAYS_EMPLOYED"), d.get("DAYS_BIRTH")
        if emp is not None and emp != ANOMALY_DAYS_EMPLOYED:
            if emp > 0:
                raise ValueError(f"DAYS_EMPLOYED doit être <= 0 (ou {ANOMALY_DAYS_EMPLOYED} = sans emploi)")
            if birth is not None and emp < birth:
                raise ValueError("DAYS_EMPLOYED ne peut pas précéder la naissance")
        return self

    def to_record(self) -> dict:
        """Dossier avec les noms d'origine, sans les champs non renseignés."""
        return self.model_dump(by_alias=True, exclude_none=True)


def _example(meta_dir) -> dict | None:
    try:
        df = pd.read_parquet(meta_dir / "sample_clients.parquet").drop(
            columns=["SK_ID_CURR", "expected_score"], errors="ignore")
        row = df.iloc[0].to_dict()
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items() if pd.notna(v)}
    except Exception:  # noqa: BLE001
        return None


def build_client_model(meta: dict) -> type[BaseModel]:
    prep = Preprocessor(meta)
    fields = {}
    for i, col in enumerate(prep.input_columns):
        if col in prep.categorical:
            typ = Literal[tuple(prep.categorical[col])] if prep.categorical[col] else str
            required = col in REQUIRED_CATEGORICAL
            fields[f"f{i}"] = ((typ if required else Optional[typ]),
                               Field(... if required else None, alias=col))
        else:
            rule, required = RULES.get(col, ({}, False))
            fields[f"f{i}"] = ((float if required else Optional[float]),
                               Field(... if required else None, alias=col, allow_inf_nan=False, **rule))
    model = create_model("ClientData", __base__=_ClientBase, **fields)
    example = _example(config.MODEL_DIR)
    if example:
        model.model_config["json_schema_extra"] = {"example": example}
        model.model_rebuild(force=True)
    return model


ClientData = build_client_model(json.loads(config.MODEL_META_PATH.read_text()))


class PredictionResponse(BaseModel):
    request_id: str
    score: float = Field(..., description="Probabilité de défaut (0-1)")
    threshold: float
    decision: Literal["ACCORDE", "REFUSE"]
    model_version: str
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str
    model_version: str
