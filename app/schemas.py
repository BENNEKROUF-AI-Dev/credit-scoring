"""Schémas Pydantic : validation stricte des entrées de l'API.

- champs obligatoires manquants  -> 422
- valeurs hors plage              -> 422
- mauvais type (texte au lieu d'un nombre) -> 422
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Education = Literal[
    "Lower secondary",
    "Secondary / secondary special",
    "Incomplete higher",
    "Higher education",
    "Academic degree",
]


class ClientData(BaseModel):
    """Données d'un client. Conventions Home Credit : DAYS_* sont négatifs
    (nombre de jours avant la demande)."""

    model_config = ConfigDict(
        extra="forbid",  # refuse les champs inconnus (faute de frappe, injection)
        json_schema_extra={
            "example": {
                "EXT_SOURCE_1": 0.45,
                "EXT_SOURCE_2": 0.62,
                "EXT_SOURCE_3": 0.51,
                "AMT_INCOME_TOTAL": 180000,
                "AMT_CREDIT": 600000,
                "AMT_ANNUITY": 28000,
                "AMT_GOODS_PRICE": 540000,
                "DAYS_BIRTH": -14000,
                "DAYS_EMPLOYED": -2500,
                "CNT_CHILDREN": 1,
                "CODE_GENDER": "F",
                "NAME_EDUCATION_TYPE": "Higher education",
            }
        },
    )

    # Scores externes : optionnels (souvent manquants dans Home Credit)
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)

    # Montants : obligatoires et strictement positifs
    AMT_INCOME_TOTAL: float = Field(..., gt=0, le=1e8, description="Revenu annuel")
    AMT_CREDIT: float = Field(..., gt=0, le=1e8, description="Montant du crédit")
    AMT_ANNUITY: float = Field(..., gt=0, le=1e7, description="Annuité")
    AMT_GOODS_PRICE: Optional[float] = Field(None, gt=0, le=1e8)

    # Âge entre 18 et 100 ans ; ancienneté d'emploi <= 0 (ou inconnue)
    DAYS_BIRTH: int = Field(..., ge=-36500, le=-6570, description="Âge en jours (négatif)")
    DAYS_EMPLOYED: Optional[int] = Field(None, ge=-36500, le=0)
    CNT_CHILDREN: int = Field(0, ge=0, le=20)

    CODE_GENDER: Literal["M", "F"]
    NAME_EDUCATION_TYPE: Education

    @model_validator(mode="after")
    def check_consistency(self):
        if self.AMT_ANNUITY > self.AMT_CREDIT:
            raise ValueError("AMT_ANNUITY ne peut pas dépasser AMT_CREDIT")
        if self.DAYS_EMPLOYED is not None and self.DAYS_EMPLOYED < self.DAYS_BIRTH:
            raise ValueError("DAYS_EMPLOYED ne peut pas précéder la naissance")
        return self


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
