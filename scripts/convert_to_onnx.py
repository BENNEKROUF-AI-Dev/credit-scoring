"""Convertit le modèle LightGBM en ONNX et vérifie la non-régression.

    python -m scripts.convert_to_onnx
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import onnx
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

from app.features import MODEL_FEATURES, build_features
from scripts.synthetic import generate_clients

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def main():
    model = joblib.load(MODEL_DIR / "model.joblib")
    onx = convert_lightgbm(
        model,
        initial_types=[("input", FloatTensorType([None, len(MODEL_FEATURES)]))],
        zipmap=False,          # sortie tenseur (plus rapide que la liste de dicts)
        target_opset=15,
    )
    onnx.save(onx, MODEL_DIR / "model.onnx")

    # Vérification : mêmes scores que LightGBM natif
    import onnxruntime as ort
    X = build_features(generate_clients(5000, seed=123)).to_numpy(np.float32)
    ref = model.booster_.predict(X.astype(np.float64))
    sess = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": X})[1][:, 1]
    diff = float(np.max(np.abs(ref - out)))
    print(f"model.onnx écrit — écart max de score vs LightGBM : {diff:.2e}")
    assert diff < 1e-4, "Régression détectée après conversion ONNX"


if __name__ == "__main__":
    main()
