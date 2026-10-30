"""Chargement unique du modèle et inférence.

Le modèle est chargé UNE fois au démarrage de l'API (voir `lifespan` dans
main.py) puis réutilisé pour toutes les requêtes.

Backends :
- "onnx"     : model.onnx via ONNX Runtime (par défaut, le plus rapide, aucune
               dépendance aux versions de scikit-learn/LightGBM de l'entraînement)
- "lightgbm" : model.txt (booster LightGBM natif), utile pour comparer
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from app import config
from app.preprocessing import Preprocessor

logger = logging.getLogger(__name__)


class ScoringModel:
    def __init__(self, backend: str | None = None, model_dir: Path | None = None,
                 fast_features: bool | None = None):
        model_dir = Path(model_dir or config.MODEL_DIR)
        self.meta = json.loads((model_dir / "model_meta.json").read_text())
        self.threshold: float = float(self.meta["threshold"])
        self.version: str = self.meta.get("version", "unknown")
        self.preprocessor = Preprocessor(self.meta)
        self.fast_features = config.FAST_FEATURES if fast_features is None else fast_features

        backend = (backend or config.MODEL_BACKEND).lower()
        if backend == "lightgbm" and not (model_dir / "model.txt").exists():
            logger.warning("model.txt absent, repli sur le backend onnx")
            backend = "onnx"
        self.backend = backend

        if backend == "onnx":
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1  # meilleure latence unitaire, pas de contention
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(model_dir / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[1].name  # probabilités
        else:
            import lightgbm as lgb

            self._booster = lgb.Booster(model_file=str(model_dir / "model.txt"))

        logger.info("Modèle %s chargé (backend=%s, %d colonnes)", self.version, self.backend,
                    len(self.preprocessor.model_columns))

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        """Probabilité de défaut à partir de la matrice déjà encodée."""
        if self.backend == "onnx":
            out = self._session.run([self._output_name], {self._input_name: X.astype(np.float32)})[0]
            return np.asarray(out)[:, 1].astype(float)
        return self._booster.predict(X.astype(np.float64))

    def predict_records(self, records: list[dict]) -> np.ndarray:
        if self.fast_features:
            X = self.preprocessor.transform(records)
        else:  # chemin v1 (pandas), conservé pour comparaison
            X = self.preprocessor.transform_pandas(records).to_numpy(dtype=np.float64)
        return self.predict_array(X)

    def predict_one(self, record: dict) -> tuple[float, float]:
        """Prédit un seul client ; retourne (score, temps d'inférence en ms).
        Le temps d'inférence inclut la préparation des données."""
        t0 = time.perf_counter()
        score = float(self.predict_records([record])[0])
        return score, (time.perf_counter() - t0) * 1000

    def decide(self, score: float) -> str:
        return "REFUSE" if score >= self.threshold else "ACCORDE"
