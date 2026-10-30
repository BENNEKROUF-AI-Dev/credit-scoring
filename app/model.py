"""Chargement unique du modèle et inférence.

Le modèle est chargé UNE fois au démarrage de l'API (voir `lifespan` dans
main.py) puis réutilisé pour toutes les requêtes.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app import config
from app.features import MODEL_FEATURES, build_features, build_features_fast

logger = logging.getLogger(__name__)


class ScoringModel:
    def __init__(self, backend: str | None = None, model_dir: Path | None = None,
                 fast_features: bool | None = None):
        model_dir = Path(model_dir or config.MODEL_DIR)
        meta_path = model_dir / "model_meta.json"
        self.meta = json.loads(meta_path.read_text())
        self.threshold: float = float(self.meta["threshold"])
        self.version: str = self.meta.get("version", "unknown")

        backend = (backend or config.MODEL_BACKEND).lower()
        onnx_path = model_dir / "model.onnx"
        if backend == "onnx" and not onnx_path.exists():
            logger.warning("model.onnx absent, repli sur le backend lightgbm")
            backend = "lightgbm"
        self.backend = backend
        self.fast_features = config.FAST_FEATURES if fast_features is None else fast_features

        if backend == "onnx":
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1  # meilleure latence unitaire, pas de contention
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            # sortie 1 = probabilités (liste de dicts ou tenseur selon le convertisseur)
            self._output_name = self._session.get_outputs()[1].name
        else:
            self._model = joblib.load(model_dir / "model.joblib")
            # booster natif : évite l'overhead de validation sklearn à chaque appel
            self._booster = self._model.booster_

        logger.info("Modèle %s chargé (backend=%s)", self.version, self.backend)

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        """Retourne la probabilité de défaut pour chaque ligne."""
        X = build_features(raw)
        return self._predict_array(X[MODEL_FEATURES].to_numpy(dtype=np.float64))

    def _predict_array(self, values: np.ndarray) -> np.ndarray:
        if self.backend == "onnx":
            out = self._session.run([self._output_name], {self._input_name: values.astype(np.float32)})[0]
            return np.asarray(out)[:, 1].astype(float)
        return self._booster.predict(values.astype(np.float64))

    def predict_one(self, record: dict) -> tuple[float, float]:
        """Prédit un seul client ; retourne (score, temps d'inférence en ms).
        Le temps d'inférence inclut le feature engineering."""
        t0 = time.perf_counter()
        if self.fast_features:
            score = float(self._predict_array(build_features_fast([record]))[0])
        else:  # chemin v1 (pandas), conservé pour comparaison
            score = float(self.predict_proba(pd.DataFrame([record]))[0])
        return score, (time.perf_counter() - t0) * 1000

    def decide(self, score: float) -> str:
        return "REFUSE" if score >= self.threshold else "ACCORDE"
